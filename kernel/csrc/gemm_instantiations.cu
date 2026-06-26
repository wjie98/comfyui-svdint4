#include "svdint4_gemm.cuh"

namespace svdint4::kernels {

template class SVDInt4Gemm<GEMMConfig_W4A4_FP16>;
template class SVDInt4Gemm<GEMMConfig_W4A4_FP16_FasterI2F>;
template class SVDInt4Gemm<GEMMConfig_W4A4_BF16>;

}  // namespace svdint4::kernels
