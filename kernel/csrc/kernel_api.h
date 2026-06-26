#pragma once

#include "runtime.h"

namespace svdint4::kernels {

void gemm_svd(Tensor act,
              Tensor wgt,
              Tensor out,
              Tensor ascales,
              Tensor wscales,
              Tensor lora_act,
              Tensor lora_up,
              Tensor bias,
              bool act_unsigned,
              std::vector<float> lora_scales);

void quantize_act_lora(Tensor input,
                       Tensor output,
                       Tensor oscales,
                       Tensor lora_down,
                       Tensor lora_act_out,
                       Tensor smooth);

}  // namespace svdint4::kernels
