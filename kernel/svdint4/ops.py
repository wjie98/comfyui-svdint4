from __future__ import annotations

import os
from collections.abc import Iterable

import torch
from torch import nn

from . import _C
from .packing import PackedInt4Weight, pack_bias, pack_linear_weight, pack_smooth

CUDA_SYNC_ENV = "SVDINT4_CUDA_SYNC"


def _validate_cuda_kernel_runtime(x: torch.Tensor, dtype: torch.dtype) -> None:
    if x.device.type != "cuda":
        raise RuntimeError("SVDInt4 kernel inputs must be CUDA tensors")
    major, minor = torch.cuda.get_device_capability(x.device)
    sm = major * 10 + minor
    if sm < 75:
        raise RuntimeError(f"SVDInt4 requires NVIDIA Turing/sm75 or newer, got sm{sm}")
    if dtype == torch.bfloat16 and sm < 80:
        raise RuntimeError("SVDInt4 bfloat16 kernels require Ampere/sm80 or newer")


def _sync_if_requested(x: torch.Tensor) -> None:
    if os.environ.get(CUDA_SYNC_ENV) == "1":
        torch.cuda.synchronize(x.device)


def quantize_act_lora(
    x: torch.Tensor,
    svd_down_packed: torch.Tensor,
    smooth_packed: torch.Tensor | None = None,
    *,
    pad_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.ndim != 2:
        raise ValueError("x must be 2D [M, K]")
    _validate_cuda_kernel_runtime(x, x.dtype)
    if smooth_packed is None:
        smooth = torch.ones(svd_down_packed.shape[0], dtype=x.dtype, device=x.device)
        smooth_packed = pack_smooth(smooth, k_pad=svd_down_packed.shape[0])
    result = _C.quantize_act_lora(x.contiguous(), svd_down_packed.contiguous(), smooth_packed.contiguous(), pad_size)
    _sync_if_requested(x)
    return result


def gemm_svd(
    qact: torch.Tensor,
    qweight: torch.Tensor,
    ascales: torch.Tensor,
    wscales: torch.Tensor,
    svd_act: torch.Tensor,
    svd_up_packed: torch.Tensor,
    *,
    bias_packed: torch.Tensor | None = None,
    actual_m: int = -1,
    actual_n: int = -1,
    act_unsigned: bool = False,
    lora_scales: list[float] | None = None,
) -> torch.Tensor:
    _validate_cuda_kernel_runtime(qact, wscales.dtype)
    result = _C.gemm_svd(
        qact.contiguous(),
        qweight.contiguous(),
        ascales.contiguous(),
        wscales.contiguous(),
        svd_act.contiguous(),
        svd_up_packed.contiguous(),
        None if bias_packed is None else bias_packed.contiguous(),
        actual_m,
        actual_n,
        act_unsigned,
        [] if lora_scales is None else lora_scales,
    )
    _sync_if_requested(qact)
    return result


def linear_svd(
    x: torch.Tensor,
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    svd_down_packed: torch.Tensor,
    svd_up_packed: torch.Tensor,
    *,
    smooth_packed: torch.Tensor | None = None,
    bias_packed: torch.Tensor | None = None,
    actual_n: int = -1,
    act_unsigned: bool = False,
    lora_scales: list[float] | None = None,
    pad_size: int = 256,
) -> torch.Tensor:
    _validate_cuda_kernel_runtime(x, x.dtype)
    if smooth_packed is None:
        smooth = torch.ones(svd_down_packed.shape[0], dtype=x.dtype, device=x.device)
        smooth_packed = pack_smooth(smooth, k_pad=svd_down_packed.shape[0])
    result = _C.linear_svd(
        x.contiguous(),
        qweight.contiguous(),
        wscales.contiguous(),
        svd_down_packed.contiguous(),
        svd_up_packed.contiguous(),
        smooth_packed.contiguous(),
        None if bias_packed is None else bias_packed.contiguous(),
        actual_n,
        act_unsigned,
        [] if lora_scales is None else lora_scales,
        pad_size,
    )
    _sync_if_requested(x)
    return result


def _extra_lora_delta(
    x2d: torch.Tensor,
    down: torch.Tensor,
    up: torch.Tensor,
    scale: float,
    out_features: int,
) -> torch.Tensor:
    down = down.to(device=x2d.device, dtype=x2d.dtype)
    up = up.to(device=x2d.device, dtype=x2d.dtype)

    if down.ndim != 2 or up.ndim != 2:
        raise ValueError("extra LoRA tensors must be 2D")
    if down.shape[0] == x2d.shape[1]:
        hidden = x2d @ down
    elif down.shape[1] == x2d.shape[1]:
        hidden = x2d @ down.t()
    else:
        raise ValueError("extra LoRA down shape must be [K, R] or [R, K]")

    rank = hidden.shape[1]
    if up.shape == (out_features, rank):
        delta = hidden @ up.t()
    elif up.shape == (rank, out_features):
        delta = hidden @ up
    elif up.shape[1] == rank:
        delta = hidden @ up[:out_features].t()
    else:
        raise ValueError("extra LoRA up shape must be [N, R] or [R, N]")
    return delta * scale


def _iter_extra_loras(extra_loras: Iterable | None):
    if extra_loras is None:
        return
    for item in extra_loras:
        if isinstance(item, dict):
            yield item["down"], item["up"], float(item.get("scale", 1.0))
        else:
            if len(item) == 2:
                down, up = item
                scale = 1.0
            elif len(item) == 3:
                down, up, scale = item
            else:
                raise ValueError("extra LoRA tuple must be (down, up) or (down, up, scale)")
            yield down, up, float(scale)


def svd_int4_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    svd_down_packed: torch.Tensor,
    svd_up_packed: torch.Tensor,
    *,
    smooth_packed: torch.Tensor | None = None,
    bias_packed: torch.Tensor | None = None,
    out_features: int | None = None,
    extra_loras: Iterable | None = None,
    act_unsigned: bool = False,
    lora_scales: list[float] | None = None,
    pad_size: int = 256,
) -> torch.Tensor:
    if x.shape[-1] > qweight.shape[1] * 2:
        raise ValueError("x K is larger than packed qweight K")

    original_shape = x.shape[:-1]
    x2d = x.reshape(-1, x.shape[-1]).contiguous()
    if out_features is None:
        out_features = qweight.shape[0]

    out = linear_svd(
        x2d,
        qweight,
        wscales,
        svd_down_packed,
        svd_up_packed,
        smooth_packed=smooth_packed,
        bias_packed=bias_packed,
        actual_n=out_features,
        act_unsigned=act_unsigned,
        lora_scales=lora_scales,
        pad_size=pad_size,
    )

    for down, up, scale in _iter_extra_loras(extra_loras):
        out = out + _extra_lora_delta(x2d, down, up, scale, out_features).to(out.dtype)

    return out.reshape(*original_shape, out_features)


class SVDInt4Linear(nn.Module):
    def __init__(
        self,
        qweight: torch.Tensor,
        wscales: torch.Tensor,
        svd_down_packed: torch.Tensor,
        svd_up_packed: torch.Tensor,
        *,
        smooth_packed: torch.Tensor | None = None,
        bias_packed: torch.Tensor | None = None,
        out_features: int | None = None,
        act_unsigned: bool = False,
        lora_scales: list[float] | None = None,
    ):
        super().__init__()
        self.register_buffer("qweight", qweight.contiguous())
        self.register_buffer("wscales", wscales.contiguous())
        self.register_buffer("svd_down_packed", svd_down_packed.contiguous())
        self.register_buffer("svd_up_packed", svd_up_packed.contiguous())
        self.register_buffer("smooth_packed", None if smooth_packed is None else smooth_packed.contiguous())
        self.register_buffer("bias_packed", None if bias_packed is None else bias_packed.contiguous())
        self.out_features = int(out_features if out_features is not None else qweight.shape[0])
        self.act_unsigned = act_unsigned
        self.lora_scales = lora_scales

    @classmethod
    def from_dense(
        cls,
        weight: torch.Tensor,
        svd_down: torch.Tensor,
        svd_up: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
        smooth: torch.Tensor | None = None,
        return_dequant: bool = False,
    ):
        from .packing import pack_svd_down, pack_svd_up

        packed: PackedInt4Weight = pack_linear_weight(weight, smooth=smooth, return_dequant=return_dequant)
        svd_down_packed = pack_svd_down(svd_down, k_pad=packed.k_pad)
        svd_up_packed = pack_svd_up(svd_up, n_pad=packed.n_pad, rank_pad=svd_down_packed.shape[1])
        bias_packed = None if bias is None else pack_bias(bias.to(weight.dtype), n_pad=packed.n_pad)
        layer = cls(
            packed.qweight,
            packed.wscales,
            svd_down_packed,
            svd_up_packed,
            smooth_packed=packed.smooth,
            bias_packed=bias_packed,
            out_features=packed.n,
        )
        if return_dequant:
            layer.dequant_weight = packed.dequant_weight
        return layer

    def forward(self, x: torch.Tensor, extra_loras: Iterable | None = None) -> torch.Tensor:
        return svd_int4_linear(
            x,
            self.qweight,
            self.wscales,
            self.svd_down_packed,
            self.svd_up_packed,
            smooth_packed=self.smooth_packed,
            bias_packed=self.bias_packed,
            out_features=self.out_features,
            extra_loras=extra_loras,
            act_unsigned=self.act_unsigned,
            lora_scales=self.lora_scales,
        )
