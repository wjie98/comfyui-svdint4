from __future__ import annotations

from dataclasses import dataclass

import torch


def ceil_divide(x: int, y: int) -> int:
    return (x + y - 1) // y


def round_up(x: int, multiple: int) -> int:
    return ceil_divide(x, multiple) * multiple


def _pad_to_shape(tensor: torch.Tensor, shape: tuple[int, ...], fill_value: float = 0) -> torch.Tensor:
    if tuple(tensor.shape) == shape:
        return tensor.contiguous()
    out = torch.full(shape, fill_value, dtype=tensor.dtype, device=tensor.device)
    slices = tuple(slice(0, size) for size in tensor.shape)
    out[slices] = tensor
    return out


def _pad_dims(
    tensor: torch.Tensor,
    divisor: int | tuple[int, ...],
    dim: int | tuple[int, ...],
    fill_value: float = 0,
) -> torch.Tensor:
    dims = (dim,) if isinstance(dim, int) else dim
    divisors = (divisor,) if isinstance(divisor, int) else divisor
    if len(dims) != len(divisors):
        raise ValueError("dim and divisor must have the same length")

    shape = list(tensor.shape)
    for d, div in zip(dims, divisors):
        shape[d] = round_up(shape[d], div)
    return _pad_to_shape(tensor, tuple(shape), fill_value=fill_value)


def pack_int4_weight(qweight: torch.Tensor) -> torch.Tensor:
    """Pack signed int4 weights in the W4A4 tensor-core layout."""
    if qweight.dtype != torch.int32:
        raise TypeError(f"qweight must be torch.int32, got {qweight.dtype}")
    n, k = qweight.shape
    if n % 128 != 0 or k % 128 != 0:
        raise ValueError("qweight shape must be padded to multiples of (128, 128)")

    bits = 4
    comp_n = 16
    comp_k = 64
    num_n_lanes = 8
    num_k_lanes = 4
    reg_n = 1
    reg_k = 8
    n_pack_size = 2
    k_pack_size = 2
    warp_n = 128
    num_n_packs = warp_n // (n_pack_size * num_n_lanes * reg_n)
    num_k_packs = 1

    packed = qweight.reshape(
        n // warp_n,
        num_n_packs,
        n_pack_size,
        num_n_lanes,
        reg_n,
        k // comp_k,
        num_k_packs,
        k_pack_size,
        num_k_lanes,
        reg_k,
    )
    packed = packed.permute(0, 5, 6, 1, 3, 8, 2, 7, 4, 9).contiguous()
    packed = packed.bitwise_and(0xF)
    shift = torch.arange(0, 32, bits, dtype=torch.int32, device=packed.device)
    packed = packed.bitwise_left_shift(shift).sum(dim=-1, dtype=torch.int32)
    return packed.view(dtype=torch.int8).view(n, -1)


def pack_scale(scale: torch.Tensor, group_size: int) -> torch.Tensor:
    """Pack fp16/bf16 per-group or per-channel scale tensors."""
    if scale.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"scale must be fp16/bf16, got {scale.dtype}")

    warp_n = 128
    num_lanes = 32
    s_pack_size = min(max(warp_n // num_lanes, 2), 8)
    num_s_lanes = min(num_lanes, warp_n // s_pack_size)
    num_s_packs = warp_n // (s_pack_size * num_s_lanes)
    warp_s = num_s_packs * num_s_lanes * s_pack_size

    n = scale.shape[0]
    if n % warp_s != 0:
        raise ValueError("scale first dimension must be padded to a multiple of 128")

    scale = scale.reshape(n // warp_s, num_s_packs, num_s_lanes // 4, s_pack_size // 2, 4, 2, -1)
    scale = scale.permute(0, 6, 1, 2, 4, 3, 5).contiguous()
    return scale.view(-1) if group_size == -1 else scale.view(-1, n)


def pack_smooth(smooth: torch.Tensor, k_pad: int | None = None) -> torch.Tensor:
    """Pack per-input-channel SVDQuant smooth factors."""
    if smooth.ndim != 1:
        raise ValueError("smooth must be 1D")
    if k_pad is None:
        k_pad = round_up(smooth.shape[0], 128)
    smooth = _pad_to_shape(smooth, (k_pad,), fill_value=1)
    return pack_scale(smooth.contiguous(), group_size=-1)


def pack_bias(bias: torch.Tensor, n_pad: int | None = None) -> torch.Tensor:
    """Pack a normal bias vector for the GEMM epilogue."""
    if bias.ndim != 1:
        raise ValueError("bias must be 1D")
    if n_pad is None:
        n_pad = round_up(bias.shape[0], 128)
    bias = _pad_to_shape(bias, (n_pad,), fill_value=0)
    return pack_scale(bias.contiguous(), group_size=-1)


def _pack_lowrank_weight(weight: torch.Tensor, down: bool) -> torch.Tensor:
    if weight.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"low-rank weight must be fp16/bf16, got {weight.dtype}")

    lane_n, lane_k = 1, 2
    n_pack_size, k_pack_size = 2, 2
    num_n_lanes, num_k_lanes = 8, 4
    frag_n = n_pack_size * num_n_lanes * lane_n
    frag_k = k_pack_size * num_k_lanes * lane_k
    weight = _pad_dims(weight, divisor=(frag_n, frag_k), dim=(0, 1))

    if down:
        r, c = weight.shape
        r_frags, c_frags = r // frag_n, c // frag_k
        weight = weight.view(r_frags, frag_n, c_frags, frag_k).permute(2, 0, 1, 3)
    else:
        c, r = weight.shape
        c_frags, r_frags = c // frag_n, r // frag_k
        weight = weight.view(c_frags, frag_n, r_frags, frag_k).permute(0, 2, 1, 3)

    weight = weight.reshape(c_frags, r_frags, n_pack_size, num_n_lanes, k_pack_size, num_k_lanes, lane_k)
    weight = weight.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
    return weight.view(c, r)


def _unpack_lowrank_weight(weight: torch.Tensor, down: bool) -> torch.Tensor:
    c, r = weight.shape
    lane_n, lane_k = 1, 2
    n_pack_size, k_pack_size = 2, 2
    num_n_lanes, num_k_lanes = 8, 4
    frag_n = n_pack_size * num_n_lanes * lane_n
    frag_k = k_pack_size * num_k_lanes * lane_k

    if down:
        r_frags, c_frags = r // frag_n, c // frag_k
    else:
        c_frags, r_frags = c // frag_n, r // frag_k
    weight = weight.view(c_frags, r_frags, num_n_lanes, num_k_lanes, n_pack_size, k_pack_size, lane_k)
    weight = weight.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    weight = weight.view(c_frags, r_frags, frag_n, frag_k)

    if down:
        return weight.permute(1, 2, 0, 3).contiguous().view(r, c)
    return weight.permute(0, 2, 1, 3).contiguous().view(c, r)


def pack_svd_down(down: torch.Tensor, k_pad: int | None = None, rank_pad: int | None = None) -> torch.Tensor:
    """Pack dense SVD down matrix with logical shape [K, R]."""
    if down.ndim != 2:
        raise ValueError("down must be 2D [K, R]")
    k, rank = down.shape
    k_pad = round_up(k, 128) if k_pad is None else k_pad
    rank_pad = round_up(rank, 16) if rank_pad is None else rank_pad
    padded = _pad_to_shape(down, (k_pad, rank_pad), fill_value=0)
    return _pack_lowrank_weight(padded.t().contiguous(), down=True)


def pack_svd_up(up: torch.Tensor, n_pad: int | None = None, rank_pad: int | None = None) -> torch.Tensor:
    """Pack dense SVD up matrix with logical shape [N, R]."""
    if up.ndim != 2:
        raise ValueError("up must be 2D [N, R]")
    n, rank = up.shape
    n_pad = round_up(n, 128) if n_pad is None else n_pad
    rank_pad = round_up(rank, 16) if rank_pad is None else rank_pad
    padded = _pad_to_shape(up, (n_pad, rank_pad), fill_value=0)
    return _pack_lowrank_weight(padded.contiguous(), down=False)


def unpack_svd_down(packed: torch.Tensor) -> torch.Tensor:
    """Unpack SVD down matrix to padded logical shape [K, R]."""
    return _unpack_lowrank_weight(packed, down=True).t().contiguous()


def unpack_svd_up(packed: torch.Tensor) -> torch.Tensor:
    """Unpack SVD up matrix to padded logical shape [N, R]."""
    return _unpack_lowrank_weight(packed, down=False)


@dataclass(frozen=True)
class PackedInt4Weight:
    qweight: torch.Tensor
    wscales: torch.Tensor
    smooth: torch.Tensor
    n: int
    k: int
    n_pad: int
    k_pad: int
    dequant_weight: torch.Tensor | None = None


def pack_linear_weight(
    weight: torch.Tensor,
    *,
    smooth: torch.Tensor | None = None,
    group_size: int = 64,
    return_dequant: bool = False,
) -> PackedInt4Weight:
    """Symmetrically quantize and pack a dense [N, K] weight matrix."""
    if weight.ndim != 2:
        raise ValueError("weight must be 2D [N, K]")
    if weight.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"weight must be fp16/bf16, got {weight.dtype}")
    if group_size != 64:
        raise ValueError("this INT4 kernel currently uses group_size=64")

    n, k = weight.shape
    n_pad, k_pad = round_up(n, 128), round_up(k, 128)

    if smooth is None:
        smooth_dense = torch.ones(k, dtype=weight.dtype, device=weight.device)
    else:
        if smooth.shape != (k,):
            raise ValueError("smooth must have shape [K]")
        smooth_dense = smooth.to(device=weight.device, dtype=weight.dtype)

    smooth_padded = _pad_to_shape(smooth_dense, (k_pad,), fill_value=1)
    weight_padded = _pad_to_shape(weight, (n_pad, k_pad), fill_value=0)
    effective_weight = weight_padded * smooth_padded.view(1, k_pad)

    grouped = effective_weight.view(n_pad, k_pad // group_size, group_size)
    max_abs = grouped.abs().amax(dim=-1)
    scale = max_abs / 7.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.round(grouped / scale.unsqueeze(-1)).clamp(-7, 7).to(torch.int32)
    q = q.view(n_pad, k_pad)

    qweight = pack_int4_weight(q)
    wscales = pack_scale(scale.to(dtype=weight.dtype).contiguous(), group_size=group_size)
    smooth_packed = pack_smooth(smooth_padded, k_pad=k_pad)

    dequant = None
    if return_dequant:
        dequant = (q.to(torch.float32).view(n_pad, k_pad // group_size, group_size) * scale.float().unsqueeze(-1))
        dequant = dequant.view(n_pad, k_pad) / smooth_padded.float().view(1, k_pad)
        dequant = dequant[:n, :k].to(dtype=weight.dtype)

    return PackedInt4Weight(
        qweight=qweight,
        wscales=wscales,
        smooth=smooth_packed,
        n=n,
        k=k,
        n_pad=n_pad,
        k_pad=k_pad,
        dequant_weight=dequant,
    )
