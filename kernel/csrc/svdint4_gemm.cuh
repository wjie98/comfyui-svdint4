#pragma once

#include "gemm_w4a4.cuh"

namespace svdint4::kernels {

template <typename Config>
class SVDInt4Gemm {
    using GEMM = GEMM_W4A4<Config>;
    using LoraUp = Lora<Config>;

    using packed_act_t = typename GEMM::packed_act_t;
    using packed_wgt_t = typename GEMM::packed_wgt_t;
    using packed_ascale_t = typename GEMM::packed_ascale_t;
    using packed_wscale_t = typename GEMM::packed_wscale_t;
    using packed_fpsum_t = typename GEMM::packed_fpsum_t;
    using half_t = typename GEMM::half_t;

public:
    static void gemm(Tensor act,
                     Tensor wgt,
                     Tensor out,
                     Tensor ascales,
                     Tensor wscales,
                     Tensor lora_act,
                     Tensor lora_up,
                     Tensor bias,
                     bool act_unsigned,
                     const std::vector<float> &lora_scales) {
        const int M = static_cast<int>(act.numel() / act.shape[-1]);
        const int N = wgt.shape[0];
        const int K = act.shape[-1] * 2;
        assert(K == wgt.shape[1] * 2);

        const int actualM = static_cast<int>(out.numel() / out.shape[-1]);
        const int actualN = out.shape[-1];
        assert(actualM <= M && M - actualM < GEMM::BLOCK_M);
        assert(actualN <= N && N - actualN < GEMM::BLOCK_N);

        auto launch = [&]<typename Epilogue>(typename Epilogue::Arguments args) {
            assert(M % GEMM::BLOCK_M == 0);
            assert(N % GEMM::BLOCK_N == 0);

            dim3 grid(M / GEMM::BLOCK_M, N / GEMM::BLOCK_N);
            bool swap_block_mn = M > N * 2;
            if (swap_block_mn) {
                std::swap(grid.x, grid.y);
            }

            dispatchBool(act_unsigned, [&]<bool ACT_UNSIGNED>() {
                auto func = invoke_kernel<typename GEMM::template gemm_w4a4_kernel<Epilogue, ACT_UNSIGNED>,
                                          const packed_act_t *,
                                          const packed_wgt_t *,
                                          const packed_ascale_t *,
                                          const packed_wscale_t *,
                                          int,
                                          int,
                                          int,
                                          typename Epilogue::Arguments,
                                          bool,
                                          bool>;

                func<<<grid, GEMM::WARP_SIZE * GEMM::NUM_WARPS, 0, getCurrentCUDAStream()>>>(
                    act.data_ptr<packed_act_t>(),
                    wgt.data_ptr<packed_wgt_t>(),
                    ascales.data_ptr<packed_ascale_t>(),
                    wscales.data_ptr<packed_wscale_t>(),
                    M,
                    N,
                    K,
                    args,
                    swap_block_mn,
                    false);
                checkCUDA(cudaGetLastError());
            });
        };

        auto launch_bias = [&]<typename NextEpilogue>(typename NextEpilogue::Arguments next_args) {
            assert(!bias.valid() || bias.numel() == static_cast<size_t>(N));

            dispatchBool(bias.valid(), [&]<bool USE_BIAS>() {
                using Bias = typename GEMM::template EpilogueBias<USE_BIAS, false>;
                using Epilogue =
                    typename GEMM::template EpilogueCombination<Bias, NextEpilogue, typename GEMM::EpilogueNop>;

                launch.template operator()<Epilogue>(
                    {typename Bias::Arguments{
                         .bias = USE_BIAS ? bias.data_ptr<packed_wscale_t>() : nullptr,
                         .scale = nullptr,
                     },
                     next_args,
                     {}});
            });
        };

        using Output = typename GEMM::EpilogueDefault;
        typename Output::Arguments output_args{
            .out = out.data_ptr<half_t>(),
            .actualM = actualM,
            .actualN = actualN,
        };

        if (!lora_up.valid()) {
            launch_bias.template operator()<Output>(output_args);
            return;
        }

        const int rank = lora_up.shape[1];
        assert(rank % 16 == 0);
        assert(lora_up.shape[0] == N);
        assert(lora_act.shape[0] == M);
        assert(lora_act.shape[1] == rank);

        typename LoraUp::scale_t scales{};
        for (size_t i = 0; i < scales.size(); ++i) {
            scales[i] = i < lora_scales.size() ? lora_scales[i] : 0.0f;
        }

        using WithLora = typename GEMM::template EpilogueCombination<typename LoraUp::EpilogueLoraUp,
                                                                     Output,
                                                                     typename GEMM::EpilogueNop>;
        launch_bias.template operator()<WithLora>(
            {typename LoraUp::EpilogueLoraUp::Arguments{
                 .lora_act = lora_act.data_ptr<float>(),
                 .lora_wgt_up = lora_up.data_ptr<packed_fpsum_t>(),
                 .rank = rank,
                 .scales = scales,
                 .alwaysfalse = false,
             },
             output_args,
             {}});
    }

    static void quantize_act_lora(Tensor input,
                                  Tensor output,
                                  Tensor oscales,
                                  Tensor lora_down,
                                  Tensor lora_act_out,
                                  Tensor smooth) {
        const int actualM = static_cast<int>(input.numel() / input.shape[-1]);
        const int actualN = input.shape[-1];
        const int M = ceilDiv(actualM, GEMM::BLOCK_M) * GEMM::BLOCK_M;
        const int N = ceilDiv(actualN, GEMM::BLOCK_N) * GEMM::BLOCK_N;

        assert(output.dtype() == Tensor::INT8);
        assert(output.numel() / output.shape[-1] == static_cast<size_t>(M));
        assert(output.shape[-1] == N / 2);
        assert(isTypeMatch<half_t>(oscales.dtype()));
        assert(oscales.numel() == static_cast<size_t>(M * N / GEMM::WARP_K));

        const int rank = lora_down.shape[1];
        assert(rank % 16 == 0);
        assert(lora_down.shape[0] == N);
        assert(lora_act_out.shape[0] == M);
        assert(lora_act_out.shape[1] == rank);

        lora_act_out.zero_();

        using Kernel = typename GEMM::template quantize_w4a4_fuse_lora_kernel<false, false>;
        auto func = invoke_kernel<Kernel, typename Kernel::Arguments>;
        checkCUDA(cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, Kernel::SHMEM_SIZE));

        dim3 grid(M / GEMM::BLOCK_M, N / GEMM::BLOCK_N);
        func<<<grid, GEMM::WARP_SIZE * GEMM::NUM_WARPS, Kernel::SHMEM_SIZE, getCurrentCUDAStream()>>>(
            typename Kernel::Arguments{
                .input = input.data_ptr<half_t>(),
                .smooth_factor = smooth.valid() ? smooth.data_ptr<packed_wscale_t>() : nullptr,
                .output = output.data_ptr<packed_act_t>(),
                .oscales = oscales.data_ptr<typename Kernel::oscales_t>(),
                .lora_wgt_down = lora_down.data_ptr<packed_fpsum_t>(),
                .lora_act = lora_act_out.data_ptr<float>(),
                .lora_rank = rank,
                .M = M,
                .N = N,
                .actualM = actualM,
                .actualN = actualN,
                .alwaysfalse = false,
            });
        checkCUDA(cudaGetLastError());
    }
};

extern template class SVDInt4Gemm<GEMMConfig_W4A4_FP16>;
extern template class SVDInt4Gemm<GEMMConfig_W4A4_FP16_FasterI2F>;
extern template class SVDInt4Gemm<GEMMConfig_W4A4_BF16>;

}  // namespace svdint4::kernels
