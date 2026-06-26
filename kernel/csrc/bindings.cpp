#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/utils/pybind.h>

#include <optional>
#include <tuple>
#include <vector>

#include "kernel_api.h"

namespace {

class TorchOpContext {
public:
    TorchOpContext() {
        stackCUDAStreams.push_back(at::cuda::getCurrentCUDAStream().stream());
    }

    TorchOpContext(const TorchOpContext &) = delete;
    TorchOpContext(TorchOpContext &&) = delete;

    ~TorchOpContext() {
        assert(!stackCUDAStreams.empty());
        assert(stackCUDAStreams.back() == at::cuda::getCurrentCUDAStream().stream());
        stackCUDAStreams.pop_back();
    }
};

int64_t ceil_div(int64_t x, int64_t y) {
    return (x + y - 1) / y;
}

template <typename To, typename From>
To int_cast(From value) {
    TORCH_CHECK(value >= static_cast<From>(std::numeric_limits<To>::min()) &&
                    value <= static_cast<From>(std::numeric_limits<To>::max()),
                "integer overflow while converting tensor metadata");
    return static_cast<To>(value);
}

Tensor from_torch(at::Tensor input) {
    Tensor result;

    const int ndims = int_cast<int>(input.dim());
    for (int i = 0; i < ndims; ++i) {
        result.shape.dataExtent.push_back(int_cast<int>(input.size(i)));
        result.shape.dataStride.push_back(int_cast<int>(input.stride(i)));
    }

    switch (input.scalar_type()) {
    case at::ScalarType::Char:
    case at::ScalarType::Byte:
        result.scalarType = Tensor::INT8;
        break;
    case at::ScalarType::Short:
        result.scalarType = Tensor::INT16;
        break;
    case at::ScalarType::Int:
        result.scalarType = Tensor::INT32;
        break;
    case at::ScalarType::Long:
        result.scalarType = Tensor::INT64;
        break;
    case at::ScalarType::Half:
        result.scalarType = Tensor::FP16;
        break;
    case at::ScalarType::Float:
        result.scalarType = Tensor::FP32;
        break;
    case at::ScalarType::BFloat16:
        result.scalarType = Tensor::BF16;
        break;
    case at::ScalarType::Float8_e4m3fn:
        result.scalarType = Tensor::FP8_E4M3;
        break;
    case at::ScalarType::Float8_e5m2:
        result.scalarType = Tensor::FP8_E5M2;
        break;
    default:
        TORCH_CHECK(false, "unsupported tensor dtype for svdint4");
    }

    result.ptr = input.data_ptr();
    result.dev = Device{input.is_cuda() ? Device::CUDA : Device::CPU, input.is_cuda() ? input.get_device() : 0};
    result.owner = std::make_shared<at::Tensor>(std::move(input));
    return result;
}

void check_cuda_2d(const at::Tensor &t, const char *name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dim() == 2, name, " must be 2D");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_half_like(const at::Tensor &t, const char *name) {
    TORCH_CHECK(t.scalar_type() == at::kHalf || t.scalar_type() == at::kBFloat16,
                name,
                " must be float16 or bfloat16");
}

Tensor maybe_tensor(const std::optional<at::Tensor> &value) {
    if (!value.has_value()) {
        return Tensor{};
    }
    return from_torch(value.value());
}

std::vector<float> normalize_lora_scales(std::vector<float> scales, int64_t rank) {
    const int64_t groups = ceil_div(rank, 16);
    if (scales.empty()) {
        scales.resize(groups, 1.0f);
    } else if (static_cast<int64_t>(scales.size()) < groups) {
        scales.resize(groups, 1.0f);
    }
    return scales;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
quantize_act_lora(at::Tensor input, at::Tensor lora_down, at::Tensor smooth, int64_t pad_size) {
    TORCH_CHECK(pad_size > 0, "pad_size must be positive");

    input = input.contiguous();
    lora_down = lora_down.contiguous();
    smooth = smooth.contiguous();

    check_cuda_2d(input, "input");
    check_cuda_2d(lora_down, "lora_down");
    TORCH_CHECK(smooth.is_cuda(), "smooth must be a CUDA tensor");
    TORCH_CHECK(smooth.is_contiguous(), "smooth must be contiguous");
    check_half_like(input, "input");
    TORCH_CHECK(lora_down.scalar_type() == input.scalar_type(), "lora_down dtype must match input dtype");
    TORCH_CHECK(smooth.scalar_type() == input.scalar_type(), "smooth dtype must match input dtype");

    const int64_t actual_m = input.size(0);
    const int64_t actual_k = input.size(1);
    const int64_t m_pad = ceil_div(actual_m, pad_size) * pad_size;
    const int64_t k_pad = ceil_div(actual_k, 128) * 128;
    const int64_t rank = lora_down.size(1);

    TORCH_CHECK(lora_down.size(0) == k_pad,
                "lora_down first dimension must equal ceil(input K / 128) * 128");
    TORCH_CHECK(rank % 16 == 0, "lora rank must be a multiple of 16");
    TORCH_CHECK(smooth.numel() == k_pad, "smooth must be packed per-channel scale with k_pad elements");

    auto byte_opts = input.options().dtype(at::kByte);
    auto scale_opts = input.options();
    auto f32_opts = input.options().dtype(at::kFloat);
    at::Tensor qact = at::empty({m_pad, k_pad / 2}, byte_opts);
    at::Tensor ascales = at::empty({k_pad / 64, m_pad}, scale_opts);
    at::Tensor lora_act = at::empty({m_pad, rank}, f32_opts);

    TorchOpContext ctx;
    svdint4::kernels::quantize_act_lora(from_torch(input),
                                         from_torch(qact),
                                         from_torch(ascales),
                                         from_torch(lora_down),
                                         from_torch(lora_act),
                                         from_torch(smooth));
    return {qact, ascales, lora_act};
}

at::Tensor gemm_svd(at::Tensor act,
                    at::Tensor qweight,
                    at::Tensor ascales,
                    at::Tensor wscales,
                    at::Tensor lora_act,
                    at::Tensor lora_up,
                    std::optional<at::Tensor> bias,
                    int64_t actual_m,
                    int64_t actual_n,
                    bool act_unsigned,
                    std::vector<float> lora_scales) {
    act = act.contiguous();
    qweight = qweight.contiguous();
    ascales = ascales.contiguous();
    wscales = wscales.contiguous();
    lora_act = lora_act.contiguous();
    lora_up = lora_up.contiguous();
    if (bias.has_value()) {
        bias = bias.value().contiguous();
    }

    check_cuda_2d(act, "act");
    check_cuda_2d(qweight, "qweight");
    check_cuda_2d(ascales, "ascales");
    check_cuda_2d(wscales, "wscales");
    check_cuda_2d(lora_act, "lora_act");
    check_cuda_2d(lora_up, "lora_up");
    TORCH_CHECK(act.scalar_type() == at::kByte || act.scalar_type() == at::kChar, "act must be int8/uint8 packed");
    TORCH_CHECK(qweight.scalar_type() == at::kByte || qweight.scalar_type() == at::kChar,
                "qweight must be int8/uint8 packed");
    check_half_like(ascales, "ascales");
    TORCH_CHECK(wscales.scalar_type() == ascales.scalar_type(), "wscales dtype must match ascales dtype");
    TORCH_CHECK(lora_up.scalar_type() == ascales.scalar_type(), "lora_up dtype must match ascales dtype");
    TORCH_CHECK(lora_act.scalar_type() == at::kFloat, "lora_act must be float32");

    const int64_t m_pad = act.size(0);
    const int64_t k_pad = act.size(1) * 2;
    const int64_t n_pad = qweight.size(0);
    const int64_t rank = lora_up.size(1);

    TORCH_CHECK(qweight.size(1) * 2 == k_pad, "qweight K must match act K");
    TORCH_CHECK(m_pad % 256 == 0, "act M must be padded to a multiple of 256");
    TORCH_CHECK(n_pad % 128 == 0, "qweight N must be padded to a multiple of 128");
    TORCH_CHECK(k_pad % 64 == 0, "K must be a multiple of 64");
    TORCH_CHECK(ascales.numel() == (k_pad / 64) * m_pad, "ascales shape/numel is invalid");
    TORCH_CHECK(wscales.numel() == (k_pad / 64) * n_pad, "wscales shape/numel is invalid");
    TORCH_CHECK(lora_act.size(0) == m_pad, "lora_act M must match act M");
    TORCH_CHECK(lora_act.size(1) == rank, "lora_act rank must match lora_up rank");
    TORCH_CHECK(lora_up.size(0) == n_pad, "lora_up N must match qweight N");
    TORCH_CHECK(rank % 16 == 0, "lora rank must be a multiple of 16");

    if (actual_m < 0) {
        actual_m = m_pad;
    }
    if (actual_n < 0) {
        actual_n = n_pad;
    }
    TORCH_CHECK(actual_m > 0 && actual_m <= m_pad, "actual_m is out of range");
    TORCH_CHECK(actual_n > 0 && actual_n <= n_pad, "actual_n is out of range");
    TORCH_CHECK(m_pad - actual_m < 256, "actual_m may only trim the final M block");
    TORCH_CHECK(n_pad - actual_n < 128, "actual_n may only trim the final N block");

    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda(), "bias must be CUDA");
        TORCH_CHECK(bias.value().is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias.value().scalar_type() == ascales.scalar_type(), "bias dtype must match ascales dtype");
        TORCH_CHECK(bias.value().numel() == n_pad, "bias must be packed/padded to qweight N");
    }

    at::Tensor out = at::empty({actual_m, actual_n}, ascales.options());
    lora_scales = normalize_lora_scales(std::move(lora_scales), rank);

    TorchOpContext ctx;
    svdint4::kernels::gemm_svd(from_torch(act),
                                from_torch(qweight),
                                from_torch(out),
                                from_torch(ascales),
                                from_torch(wscales),
                                from_torch(lora_act),
                                from_torch(lora_up),
                                maybe_tensor(bias),
                                act_unsigned,
                                std::move(lora_scales));
    return out;
}

at::Tensor linear_svd(at::Tensor input,
                      at::Tensor qweight,
                      at::Tensor wscales,
                      at::Tensor lora_down,
                      at::Tensor lora_up,
                      at::Tensor smooth,
                      std::optional<at::Tensor> bias,
                      int64_t actual_n,
                      bool act_unsigned,
                      std::vector<float> lora_scales,
                      int64_t pad_size) {
    const int64_t actual_m = input.size(0);
    auto [qact, ascales, lora_act] = quantize_act_lora(input, lora_down, smooth, pad_size);
    return gemm_svd(qact,
                    qweight,
                    ascales,
                    wscales,
                    lora_act,
                    lora_up,
                    bias,
                    actual_m,
                    actual_n,
                    act_unsigned,
                    std::move(lora_scales));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_act_lora",
          &quantize_act_lora,
          pybind11::arg("input"),
          pybind11::arg("lora_down"),
          pybind11::arg("smooth"),
          pybind11::arg("pad_size") = 256);
    m.def("gemm_svd",
          &gemm_svd,
          pybind11::arg("act"),
          pybind11::arg("qweight"),
          pybind11::arg("ascales"),
          pybind11::arg("wscales"),
          pybind11::arg("lora_act"),
          pybind11::arg("lora_up"),
          pybind11::arg("bias") = std::nullopt,
          pybind11::arg("actual_m") = -1,
          pybind11::arg("actual_n") = -1,
          pybind11::arg("act_unsigned") = false,
          pybind11::arg("lora_scales") = std::vector<float>{});
    m.def("linear_svd",
          &linear_svd,
          pybind11::arg("input"),
          pybind11::arg("qweight"),
          pybind11::arg("wscales"),
          pybind11::arg("lora_down"),
          pybind11::arg("lora_up"),
          pybind11::arg("smooth"),
          pybind11::arg("bias") = std::nullopt,
          pybind11::arg("actual_n") = -1,
          pybind11::arg("act_unsigned") = false,
          pybind11::arg("lora_scales") = std::vector<float>{},
          pybind11::arg("pad_size") = 256);
}
