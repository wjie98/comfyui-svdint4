#include "kernel_api.h"
#include "svdint4_gemm.cuh"

namespace svdint4::kernels {

template <typename F>
static void dispatch_int4(Tensor::ScalarType dtype, bool faster_i2f, F &&func) {
    if (faster_i2f && dtype == Tensor::FP16) {
        func.template operator()<GEMMConfig_W4A4_FP16_FasterI2F>();
    } else if (dtype == Tensor::FP16) {
        func.template operator()<GEMMConfig_W4A4_FP16>();
    } else if (dtype == Tensor::BF16) {
        func.template operator()<GEMMConfig_W4A4_BF16>();
    } else {
        throw std::invalid_argument("svdint4 only supports fp16/bfloat16 scale tensors");
    }
}

static bool use_faster_i2f(bool act_unsigned) {
    auto *prop = getCurrentDeviceProperties();
    return prop->major == 7 && prop->minor == 5 && !act_unsigned;
}

void gemm_svd(Tensor act,
              Tensor wgt,
              Tensor out,
              Tensor ascales,
              Tensor wscales,
              Tensor lora_act,
              Tensor lora_up,
              Tensor bias,
              bool act_unsigned,
              std::vector<float> lora_scales) {
    dispatch_int4(ascales.dtype(), use_faster_i2f(act_unsigned), [&]<typename Config>() {
        SVDInt4Gemm<Config>::gemm(act, wgt, out, ascales, wscales, lora_act, lora_up, bias, act_unsigned, lora_scales);
    });
}

void quantize_act_lora(Tensor input,
                       Tensor output,
                       Tensor oscales,
                       Tensor lora_down,
                       Tensor lora_act_out,
                       Tensor smooth) {
    dispatch_int4(input.dtype(), false, [&]<typename Config>() {
        SVDInt4Gemm<Config>::quantize_act_lora(input, output, oscales, lora_down, lora_act_out, smooth);
    });
}

}  // namespace svdint4::kernels
