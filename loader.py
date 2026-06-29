from __future__ import annotations

import dataclasses
import logging
import math
import uuid
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open

import comfy.lora
import comfy.ops
import comfy.sd
import comfy.weight_adapter
from comfy.patcher_extension import CallbacksMP

try:
    from comfy.quant_ops import QuantizedLayout, QuantizedTensor, register_layout_class, register_layout_op
    _HAS_COMFY_QUANTIZED_TENSOR = issubclass(QuantizedTensor, torch.Tensor)
except Exception:
    QuantizedLayout = object
    QuantizedTensor = None
    _HAS_COMFY_QUANTIZED_TENSOR = False

    def register_layout_class(name, cls):
        return None

    def register_layout_op(torch_op, layout_cls):
        def decorator(func):
            return func
        return decorator


LOG = logging.getLogger("comfyui-svdint4")

PACKED_FIELDS = {"qweight", "wscales", "smooth", "svd_down", "svd_up", "bias_packed"}
REQUIRED_FIELDS = {"qweight", "wscales", "svd_down", "svd_up"}
W4_STORAGE_FIELDS = {"w4_qweight", "w4_scales", "w4_zeros"}
W4_STORAGE_REQUIRED_FIELDS = {"w4_qweight", "w4_scales"}
W4_STORAGE_GROUP_SIZE = 64
SUPPORTED_FORMATS = {"svdint4-dit-single-v2"}
COMPUTE_DTYPE = torch.float16
SVDINT4_LAYOUT_NAME = "SVDInt4PackedLayout"
W4_STORAGE_LAYOUT_NAME = "SVDInt4W4StorageLayout"
SVDINT4_STATE_PREFIX = "svdint4_"


def _kernel_install_hint() -> str:
    local_kernel = Path(__file__).resolve().parent / "kernel"
    if local_kernel.is_dir():
        return f"python -m pip install -v --no-build-isolation -e {local_kernel}"
    return "python -m pip install --no-build-isolation svdint4-kernel"


def _load_svdint4_linear():
    try:
        from svdint4.ops import svd_int4_linear
    except Exception as exc:
        raise RuntimeError(
            "SVDInt4 kernel is not installed or failed to load in the ComfyUI environment. "
            "Install it with the ComfyUI environment's Torch, for example: "
            f"SVDINT4_ARCH_LIST='7.5;8.0;8.6;8.9' {_kernel_install_hint()}"
        ) from exc
    return svd_int4_linear


def _validate_cuda_kernel_runtime(x: torch.Tensor) -> None:
    if x.device.type != "cuda":
        raise RuntimeError("SVDInt4 kernel inputs must be CUDA tensors")
    major, minor = torch.cuda.get_device_capability(x.device)
    sm = major * 10 + minor
    if sm < 75:
        raise RuntimeError(f"SVDInt4 requires NVIDIA Turing/sm75 or newer, got sm{sm}")


def _split_packed_key(key: str) -> tuple[str, str] | None:
    if "." not in key:
        return None
    name, field = key.rsplit(".", 1)
    if field not in PACKED_FIELDS:
        return None
    return name, field


def _split_w4_storage_key(key: str) -> tuple[str, str] | None:
    if "." not in key:
        return None
    name, field = key.rsplit(".", 1)
    if field not in W4_STORAGE_FIELDS:
        return None
    return name, field


def _model_metadata(model_path: Path) -> dict[str, str]:
    with safe_open(model_path, framework="pt", device="cpu") as handle:
        return handle.metadata() or {}


def is_svdint4_file(model_path: str | Path) -> bool:
    try:
        return _model_metadata(Path(model_path)).get("format") in SUPPORTED_FORMATS
    except Exception:
        return False


def _validate_metadata(metadata: dict[str, str], model_path: Path) -> None:
    fmt = metadata.get("format")
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(
            f"{model_path} is not an SVDInt4 DiT file: format={fmt!r}; "
            f"expected one of {sorted(SUPPORTED_FORMATS)}"
        )


def _collect_packed_layers_from_handle(handle) -> dict[str, dict[str, tuple[str, tuple[int, ...]]]]:
    layers: dict[str, dict[str, tuple[str, tuple[int, ...]]]] = {}
    for key in handle.keys():
        split = _split_packed_key(key)
        if split is None:
            continue
        name, field = split
        layers.setdefault(name, {})[field] = (key, tuple(handle.get_slice(key).get_shape()))

    valid: dict[str, dict[str, tuple[str, tuple[int, ...]]]] = {}
    for name, fields in sorted(layers.items()):
        missing = REQUIRED_FIELDS - fields.keys()
        if missing:
            LOG.warning("Skipping incomplete SVDInt4 layer %s: missing %s", name, sorted(missing))
            continue
        valid[name] = fields
    return valid


def _collect_w4_storage_layers_from_handle(handle) -> dict[str, dict[str, tuple[str, tuple[int, ...]]]]:
    layers: dict[str, dict[str, tuple[str, tuple[int, ...]]]] = {}
    for key in handle.keys():
        split = _split_w4_storage_key(key)
        if split is None:
            continue
        name, field = split
        layers.setdefault(name, {})[field] = (key, tuple(handle.get_slice(key).get_shape()))

    valid: dict[str, dict[str, tuple[str, tuple[int, ...]]]] = {}
    for name, fields in sorted(layers.items()):
        missing = W4_STORAGE_REQUIRED_FIELDS - fields.keys()
        if missing:
            LOG.warning("Skipping incomplete W4 storage layer %s: missing %s", name, sorted(missing))
            continue
        valid[name] = fields
    return valid


def _shape_numel(shape: tuple[int, ...]) -> int:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel


def _validate_rank(name: str, field: str, shape: tuple[int, ...], rank: int) -> None:
    if len(shape) != rank:
        raise ValueError(f"SVDInt4 layer {name}.{field} must be {rank}D, got shape {shape}")
    if any(int(dim) <= 0 for dim in shape):
        raise ValueError(f"SVDInt4 layer {name}.{field} has an invalid empty dimension: {shape}")


def _validate_packed_layer_shapes(name: str, fields: dict[str, tuple[str, tuple[int, ...]]]) -> tuple[int, int]:
    qshape = fields["qweight"][1]
    _validate_rank(name, "qweight", qshape, 2)
    out_features = int(qshape[0])
    in_features = int(qshape[1]) * 2
    if out_features % 128 != 0:
        raise ValueError(
            f"SVDInt4 layer {name}.qweight out_features must be padded to a multiple of 128, got {out_features}"
        )
    if in_features % 64 != 0:
        raise ValueError(
            f"SVDInt4 layer {name}.qweight in_features must be padded to a multiple of 64, got {in_features}"
        )

    wshape = fields["wscales"][1]
    _validate_rank(name, "wscales", wshape, 2)
    expected_wscale_numel = (in_features // 64) * out_features
    if _shape_numel(wshape) != expected_wscale_numel:
        raise ValueError(
            f"SVDInt4 layer {name}.wscales has {wshape} ({_shape_numel(wshape)} values), "
            f"expected {(in_features // 64, out_features)} or any 2D shape with {expected_wscale_numel} values"
        )

    down_shape = fields["svd_down"][1]
    up_shape = fields["svd_up"][1]
    _validate_rank(name, "svd_down", down_shape, 2)
    _validate_rank(name, "svd_up", up_shape, 2)
    if int(down_shape[0]) != in_features:
        raise ValueError(f"SVDInt4 layer {name}.svd_down K must match qweight K {in_features}, got {down_shape}")
    if int(up_shape[0]) != out_features:
        raise ValueError(f"SVDInt4 layer {name}.svd_up N must match qweight N {out_features}, got {up_shape}")
    if int(down_shape[1]) != int(up_shape[1]):
        raise ValueError(f"SVDInt4 layer {name} SVD rank mismatch: svd_down={down_shape}, svd_up={up_shape}")
    if int(down_shape[1]) % 16 != 0:
        raise ValueError(f"SVDInt4 layer {name} SVD rank must be padded to a multiple of 16, got {down_shape[1]}")

    if "smooth" in fields:
        smooth_shape = fields["smooth"][1]
        _validate_rank(name, "smooth", smooth_shape, 1)
        if _shape_numel(smooth_shape) != in_features:
            raise ValueError(f"SVDInt4 layer {name}.smooth must contain {in_features} values, got shape {smooth_shape}")
    if "bias_packed" in fields:
        bias_shape = fields["bias_packed"][1]
        _validate_rank(name, "bias_packed", bias_shape, 1)
        if _shape_numel(bias_shape) != out_features:
            raise ValueError(
                f"SVDInt4 layer {name}.bias_packed must contain {out_features} values, got shape {bias_shape}"
            )

    return in_features, out_features


def _validate_packed_layer_tensors(
    name: str,
    tensors: dict[str, torch.Tensor],
    expected_in: int,
    expected_out: int,
) -> None:
    for field in REQUIRED_FIELDS:
        if field not in tensors:
            raise ValueError(f"SVDInt4 layer {name} is missing required tensor {field}")

    qweight = tensors["qweight"]
    if qweight.dtype not in (torch.int8, torch.uint8):
        raise TypeError(f"SVDInt4 layer {name}.qweight must be packed int8/uint8, got {qweight.dtype}")
    q_in = int(qweight.shape[1]) * 2 if qweight.ndim == 2 else -1
    q_out = int(qweight.shape[0]) if qweight.ndim == 2 else -1
    if q_in != int(expected_in) or q_out != int(expected_out):
        raise ValueError(
            f"SVDInt4 layer {name}.qweight shape changed while loading: got {tuple(qweight.shape)}, "
            f"expected packed shape ({expected_out}, {expected_in // 2})"
        )

    for field in ("wscales", "svd_down", "svd_up", "smooth", "bias_packed"):
        value = tensors.get(field)
        if value is None:
            continue
        if value.dtype != COMPUTE_DTYPE:
            raise TypeError(f"SVDInt4 layer {name}.{field} must be {COMPUTE_DTYPE}, got {value.dtype}")


def _validate_w4_storage_layer_tensors(
    name: str,
    tensors: dict[str, torch.Tensor],
    expected_in: int,
    expected_out: int,
) -> None:
    for field in W4_STORAGE_REQUIRED_FIELDS:
        if field not in tensors:
            raise ValueError(f"W4 storage layer {name} is missing required tensor {field}")

    qweight = tensors["w4_qweight"]
    if qweight.dtype not in (torch.uint8, torch.int8):
        raise TypeError(f"W4 storage layer {name}.w4_qweight must be uint8/int8, got {qweight.dtype}")
    if qweight.ndim != 2:
        raise ValueError(f"W4 storage layer {name}.w4_qweight must be 2D, got shape {tuple(qweight.shape)}")
    expected_packed_in = (int(expected_in) + 1) // 2
    if int(qweight.shape[0]) != int(expected_out) or int(qweight.shape[1]) != expected_packed_in:
        raise ValueError(
            f"W4 storage layer {name}.w4_qweight shape mismatch: got {tuple(qweight.shape)}, "
            f"expected ({expected_out}, {expected_packed_in}) for dense shape ({expected_out}, {expected_in})"
        )

    scales = tensors["w4_scales"]
    if scales.dtype != COMPUTE_DTYPE:
        raise TypeError(f"W4 storage layer {name}.w4_scales must be {COMPUTE_DTYPE}, got {scales.dtype}")
    if scales.ndim != 2:
        raise ValueError(f"W4 storage layer {name}.w4_scales must be 2D, got shape {tuple(scales.shape)}")
    if int(scales.shape[0]) != int(expected_out):
        raise ValueError(
            f"W4 storage layer {name}.w4_scales shape mismatch: got {tuple(scales.shape)}, "
            f"expected out_features {expected_out}"
        )
    _infer_w4_storage_group_size(name, int(expected_in), int(scales.shape[1]))

    zeros = tensors.get("w4_zeros")
    if zeros is not None:
        if zeros.dtype != COMPUTE_DTYPE:
            raise TypeError(f"W4 storage layer {name}.w4_zeros must be {COMPUTE_DTYPE}, got {zeros.dtype}")
        if zeros.ndim != 2:
            raise ValueError(f"W4 storage layer {name}.w4_zeros must be 2D, got shape {tuple(zeros.shape)}")
        if tuple(zeros.shape) != tuple(scales.shape):
            raise ValueError(
                f"W4 storage layer {name}.w4_zeros shape mismatch: got {tuple(zeros.shape)}, "
                f"expected {tuple(scales.shape)}"
            )


def _infer_w4_storage_group_size(name: str, in_features: int, group_count: int) -> int:
    if group_count <= 0:
        raise ValueError(f"W4 storage layer {name}.w4_scales must have at least one group")
    if in_features % group_count == 0:
        return in_features // group_count
    default_groups = math.ceil(in_features / W4_STORAGE_GROUP_SIZE)
    if group_count == default_groups:
        return W4_STORAGE_GROUP_SIZE
    raise ValueError(
        f"W4 storage layer {name} has {group_count} groups for {in_features} input features; "
        "non-divisible custom group sizes are not supported without explicit metadata"
    )


def _owned_cpu_tensor(value: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    if dtype is not None and value.dtype != dtype:
        return value.to(dtype).contiguous()
    return value.contiguous().clone()


def _packed_tensor_dtype(field: str) -> torch.dtype | None:
    if field in {"wscales", "smooth", "svd_down", "svd_up", "bias_packed"}:
        return COMPUTE_DTYPE
    return None


def _w4_storage_tensor_dtype(field: str) -> torch.dtype | None:
    if field in {"w4_scales", "w4_zeros"}:
        return COMPUTE_DTYPE
    return torch.uint8 if field == "w4_qweight" else None


def _tensor_nbytes(value: torch.Tensor | None) -> int:
    if value is None:
        return 0
    return value.numel() * value.element_size()


def _mb(value: int | float) -> float:
    return float(value) / (1024 * 1024)


def _adapter_weights_nbytes(value) -> int:
    if isinstance(value, torch.Tensor):
        return _tensor_nbytes(value)
    if isinstance(value, tuple):
        return sum(_adapter_weights_nbytes(item) for item in value)
    if isinstance(value, list):
        return sum(_adapter_weights_nbytes(item) for item in value)
    return 0


def _adapter_weights_to_staging_source(value, dtype: torch.dtype):
    if isinstance(value, torch.Tensor):
        value = value.detach()
        if value.dtype in (torch.float16, torch.bfloat16, torch.float32):
            value = value.to(device="cpu", dtype=dtype)
        elif value.device.type != "cpu":
            value = value.to(device="cpu")
        return value.contiguous().clone()
    if isinstance(value, tuple):
        return tuple(_adapter_weights_to_staging_source(item, dtype) for item in value)
    if isinstance(value, list):
        return [_adapter_weights_to_staging_source(item, dtype) for item in value]
    return value


def _adapter_to_staging_source(adapter: comfy.weight_adapter.WeightAdapterBase, dtype: torch.dtype):
    if getattr(adapter, "_svdint4_staging_source_dtype", None) == dtype:
        return adapter
    staged = type(adapter)(adapter.loaded_keys, _adapter_weights_to_staging_source(adapter.weights, dtype))
    staged._svdint4_staging_source_dtype = dtype
    return staged


def _adapter_staging_nbytes(adapter: comfy.weight_adapter.WeightAdapterBase) -> int:
    counter = [0]
    comfy.lora.prefetch_prepared_value(adapter, counter, None, None, False)
    return int(counter[0])


@dataclasses.dataclass(frozen=True)
class SVDInt4AdapterOverlay:
    strength_patch: float
    patch_data: comfy.weight_adapter.WeightAdapterBase
    strength_model: float
    offset: object
    function: object
    staging_nbytes: int
    source_nbytes: int


class SVDInt4AdapterStagingRuntime:
    """Per-model GPU staging buffer for SVDInt4 adapter overlays."""

    def __init__(self):
        self.buffer: torch.Tensor | None = None
        self.buffer_size = 0

    def clear(self) -> None:
        self.buffer = None
        self.buffer_size = 0

    def _ensure_buffer(self, device: torch.device, required: int) -> torch.Tensor:
        if self.buffer is None or self.buffer.device != device or self.buffer_size < required:
            self.buffer = torch.empty((required,), dtype=torch.uint8, device=device)
            self.buffer_size = required
        return self.buffer

    def prepare(self, overlays: list[SVDInt4AdapterOverlay], device: torch.device) -> list[SVDInt4AdapterOverlay]:
        required = sum(overlay.staging_nbytes for overlay in overlays)
        if required <= 0:
            return overlays

        buffer = self._ensure_buffer(device, required)
        counter = [0]

        prepared: list[SVDInt4AdapterOverlay] = []
        for overlay in overlays:
            patch_data = comfy.lora.prefetch_prepared_value(
                overlay.patch_data,
                counter,
                buffer,
                None,
                True,
            )
            prepared.append(dataclasses.replace(overlay, patch_data=patch_data))

        if counter[0] > required:
            raise RuntimeError(
                f"SVDInt4 adapter staging buffer was too small: prepared {counter[0]} bytes into {required} bytes"
            )
        return prepared


@dataclasses.dataclass
class SVDInt4PackedParams:
    wscales: torch.Tensor
    svd_down: torch.Tensor
    svd_up: torch.Tensor
    smooth: torch.Tensor
    bias_packed: torch.Tensor
    orig_dtype: torch.dtype
    orig_shape: tuple[int, int]
    has_smooth: bool = False
    has_bias_packed: bool = False
    name: str = ""
    transposed: bool = False

    def _tensor_fields(self) -> tuple[str, ...]:
        return ("wscales", "svd_down", "svd_up", "smooth", "bias_packed")

    def to_device(self, device: torch.device) -> "SVDInt4PackedParams":
        kwargs = {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}
        for name in self._tensor_fields():
            kwargs[name] = kwargs[name].to(device=device)
        return type(self)(**kwargs)

    def clone(self) -> "SVDInt4PackedParams":
        kwargs = {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}
        for name in self._tensor_fields():
            kwargs[name] = kwargs[name].clone()
        return type(self)(**kwargs)

    def copy_from(self, src: "SVDInt4PackedParams", non_blocking: bool = False) -> None:
        for field in dataclasses.fields(self):
            value = getattr(src, field.name)
            if field.name in self._tensor_fields():
                getattr(self, field.name).copy_(value, non_blocking=non_blocking)
            else:
                object.__setattr__(self, field.name, value)


class SVDInt4PackedLayout(QuantizedLayout):
    Params = SVDInt4PackedParams
    MIN_SM_VERSION = (7, 5)

    @classmethod
    def quantize(cls, tensor: torch.Tensor, **kwargs):
        raise NotImplementedError("SVDInt4PackedLayout is loaded from prepacked tensors")

    @classmethod
    def dequantize(cls, qdata: torch.Tensor, params: SVDInt4PackedParams) -> torch.Tensor:
        raise RuntimeError("SVDInt4 packed weights cannot be dequantized through ComfyUI fallback ops")

    @classmethod
    def get_plain_tensors(cls, qtensor) -> tuple[torch.Tensor, ...]:
        params = qtensor._params
        return (
            qtensor._qdata,
            params.wscales,
            params.svd_down,
            params.svd_up,
            params.smooth,
            params.bias_packed,
        )

    @classmethod
    def state_dict_tensors(cls, qdata: torch.Tensor, params: SVDInt4PackedParams) -> dict[str, torch.Tensor]:
        tensors = {
            "": qdata,
            ".wscales": params.wscales,
            ".svd_down": params.svd_down,
            ".svd_up": params.svd_up,
        }
        if params.has_smooth:
            tensors[".smooth"] = params.smooth
        if params.has_bias_packed:
            tensors[".bias_packed"] = params.bias_packed
        return tensors


if _HAS_COMFY_QUANTIZED_TENSOR:
    register_layout_class(SVDINT4_LAYOUT_NAME, SVDInt4PackedLayout)


@dataclasses.dataclass(frozen=True)
class W4StorageParams:
    scales: torch.Tensor
    orig_dtype: torch.dtype
    orig_shape: tuple[int, int]
    zeros: torch.Tensor | None = None
    group_size: int = W4_STORAGE_GROUP_SIZE
    name: str = ""

    def _tensor_fields(self) -> tuple[str, ...]:
        return ("scales", "zeros")

    def to_device(self, device: torch.device) -> "W4StorageParams":
        zeros = None if self.zeros is None else self.zeros.to(device=device)
        return dataclasses.replace(self, scales=self.scales.to(device=device), zeros=zeros)

    def clone(self) -> "W4StorageParams":
        zeros = None if self.zeros is None else self.zeros.clone()
        return dataclasses.replace(self, scales=self.scales.clone(), zeros=zeros)

    def copy_from(self, src: "W4StorageParams", non_blocking: bool = False) -> None:
        self.scales.copy_(src.scales, non_blocking=non_blocking)
        if self.zeros is not None and src.zeros is not None:
            self.zeros.copy_(src.zeros, non_blocking=non_blocking)
        else:
            object.__setattr__(self, "zeros", src.zeros)
        object.__setattr__(self, "orig_dtype", src.orig_dtype)
        object.__setattr__(self, "orig_shape", src.orig_shape)
        object.__setattr__(self, "group_size", src.group_size)
        object.__setattr__(self, "name", src.name)


def _dequantize_w4_storage(qdata: torch.Tensor, params: W4StorageParams) -> torch.Tensor:
    out_features, in_features = params.orig_shape
    packed = qdata.to(torch.uint8)
    unpacked = torch.empty(
        (packed.shape[0], packed.shape[1] * 2),
        device=packed.device,
        dtype=torch.int16,
    )
    unpacked[:, 0::2] = (packed & 0x0F).to(torch.int16)
    unpacked[:, 1::2] = (packed >> 4).to(torch.int16)
    q = unpacked[:, :in_features].to(dtype=params.orig_dtype)
    scales = params.scales.to(device=q.device, dtype=params.orig_dtype)
    expanded_scales = scales.repeat_interleave(params.group_size, dim=1)[:, :in_features]
    if params.zeros is None:
        q.sub_(8.0)
        q.mul_(expanded_scales)
    else:
        zeros = params.zeros.to(device=q.device, dtype=params.orig_dtype)
        expanded_zeros = zeros.repeat_interleave(params.group_size, dim=1)[:, :in_features]
        q.mul_(expanded_scales).add_(expanded_zeros)
    return q.view(out_features, in_features)


class W4StorageLayout(QuantizedLayout):
    Params = W4StorageParams

    @classmethod
    def quantize(cls, tensor: torch.Tensor, **kwargs):
        raise NotImplementedError("W4StorageLayout is loaded from prepacked tensors")

    @classmethod
    def dequantize(cls, qdata: torch.Tensor, params: W4StorageParams) -> torch.Tensor:
        return _dequantize_w4_storage(qdata, params)

    @classmethod
    def get_plain_tensors(cls, qtensor) -> tuple[torch.Tensor, ...]:
        params = qtensor._params
        if params.zeros is None:
            return (qtensor._qdata, params.scales)
        return (qtensor._qdata, params.scales, params.zeros)

    @classmethod
    def state_dict_tensors(cls, qdata: torch.Tensor, params: W4StorageParams) -> dict[str, torch.Tensor]:
        tensors = {
            ".w4_qweight": qdata,
            ".w4_scales": params.scales,
        }
        if params.zeros is not None:
            tensors[".w4_zeros"] = params.zeros
        return tensors


if _HAS_COMFY_QUANTIZED_TENSOR:
    register_layout_class(W4_STORAGE_LAYOUT_NAME, W4StorageLayout)


@register_layout_op(torch.ops.aten.linear.default, W4StorageLayout)
def _handle_w4_storage_linear(qt, args, kwargs):
    input_tensor, weight = args[0], args[1]
    bias = args[2] if len(args) > 2 else kwargs.get("bias")
    if not isinstance(weight, QuantizedTensor):
        return torch.nn.functional.linear(input_tensor, weight, bias)
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()
    return torch.nn.functional.linear(input_tensor, weight.dequantize(), bias)


def _svdint4_forward(input_tensor: torch.Tensor, weight_qt, bias: torch.Tensor | None = None) -> torch.Tensor:
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()
    _validate_cuda_kernel_runtime(input_tensor)
    params = weight_qt._params
    out_features = params.orig_shape[1] if params.transposed else params.orig_shape[0]
    svd_int4_linear = _load_svdint4_linear()
    out = svd_int4_linear(
        input_tensor,
        weight_qt._qdata,
        params.wscales,
        params.svd_down,
        params.svd_up,
        smooth_packed=params.smooth if params.has_smooth else None,
        bias_packed=params.bias_packed if params.has_bias_packed else None,
        out_features=out_features,
    )
    if bias is not None:
        out = out + bias
    return out


@register_layout_op(torch.ops.aten.linear.default, SVDInt4PackedLayout)
def _handle_svdint4_linear(qt, args, kwargs):
    input_tensor, weight = args[0], args[1]
    bias = args[2] if len(args) > 2 else kwargs.get("bias")
    if not isinstance(weight, QuantizedTensor):
        return torch.nn.functional.linear(input_tensor, weight, bias)
    if weight._params.transposed:
        raise RuntimeError("SVDInt4 F.linear expects the stored, non-transposed packed weight")
    return _svdint4_forward(input_tensor, weight, bias)


@register_layout_op(torch.ops.aten.t.default, SVDInt4PackedLayout)
def _handle_svdint4_t(qt, args, kwargs):
    input_tensor = args[0]
    if not isinstance(input_tensor, QuantizedTensor):
        return torch.ops.aten.t.default(*args, **kwargs)
    old = input_tensor._params
    new_params = dataclasses.replace(
        old,
        orig_shape=(old.orig_shape[1], old.orig_shape[0]),
        transposed=not old.transposed,
    )
    return QuantizedTensor(input_tensor._qdata, SVDINT4_LAYOUT_NAME, new_params)


def _resolve_svdint4_rhs(rhs) -> None:
    if not isinstance(rhs, QuantizedTensor):
        raise TypeError("SVDInt4 RHS must be a QuantizedTensor")
    if not rhs._params.transposed:
        raise RuntimeError("SVDInt4 GEMM expects RHS to be W.T; use F.linear(x, W) or mm(x, W.t())")


@register_layout_op(torch.ops.aten.mm.default, SVDInt4PackedLayout)
def _handle_svdint4_mm(qt, args, kwargs):
    a, b = args[0], args[1]
    if not isinstance(b, QuantizedTensor):
        return torch.mm(a, b)
    _resolve_svdint4_rhs(b)
    return _svdint4_forward(a, b, bias=None)


@register_layout_op(torch.ops.aten.addmm.default, SVDInt4PackedLayout)
def _handle_svdint4_addmm(qt, args, kwargs):
    bias, a, b = args[0], args[1], args[2]
    if not isinstance(b, QuantizedTensor):
        return torch.addmm(bias, a, b)
    _resolve_svdint4_rhs(b)
    return _svdint4_forward(a, b, bias=bias)


class SVDInt4PackedTensor:
    """Packed SVDInt4 weight storage for one logical Linear weight."""

    is_svdint4_packed_tensor = True

    def __init__(
        self,
        name: str,
        in_features: int,
        out_features: int,
        tensors: dict[str, torch.Tensor],
        compute_dtype: torch.dtype,
    ):
        self.name = name
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.shape = torch.Size((self.out_features, self.in_features))
        self.dtype = compute_dtype
        self.requires_grad = False
        shape_fields = {field: (f"{name}.{field}", tuple(value.shape)) for field, value in tensors.items()}
        actual_in, actual_out = _validate_packed_layer_shapes(name, shape_fields)
        _validate_packed_layer_tensors(name, tensors, actual_in, actual_out)
        if actual_in != self.in_features or actual_out != self.out_features:
            raise ValueError(
                f"SVDInt4 layer {name} shape mismatch with ComfyUI model config: "
                f"packed=({actual_out}, {actual_in}), module=({self.out_features}, {self.in_features})"
            )
        self.has_bias_packed = "bias_packed" in tensors
        qweight = tensors["qweight"].contiguous()
        wscales = tensors["wscales"].contiguous()
        smooth = tensors.get("smooth")
        has_smooth = smooth is not None
        if smooth is None:
            smooth = torch.empty((0,), dtype=compute_dtype)
        svd_down = tensors["svd_down"].contiguous()
        svd_up = tensors["svd_up"].contiguous()
        bias_packed = tensors.get("bias_packed")
        if bias_packed is None:
            bias_packed = torch.empty((0,), dtype=compute_dtype)
        if _HAS_COMFY_QUANTIZED_TENSOR:
            params = SVDInt4PackedParams(
                wscales=wscales,
                smooth=smooth,
                svd_down=svd_down,
                svd_up=svd_up,
                bias_packed=bias_packed,
                orig_dtype=compute_dtype,
                orig_shape=(self.out_features, self.in_features),
                has_smooth=has_smooth,
                has_bias_packed=self.has_bias_packed,
                name=name,
            )
            self.tensor = QuantizedTensor(qweight, SVDINT4_LAYOUT_NAME, params)
        else:
            self.tensor = None
        self.packed_nbytes = sum(
            _tensor_nbytes(value)
            for value in (
                qweight,
                wscales,
                smooth if has_smooth else None,
                svd_down,
                svd_up,
                bias_packed if self.has_bias_packed else None,
            )
        )

    @property
    def nbytes(self) -> int:
        return self.packed_nbytes

    def numel(self) -> int:
        return self.out_features * self.in_features

    def element_size(self) -> int:
        return torch.tensor([], dtype=self.dtype).element_size()


class W4StorageTensor:
    """Weight-only INT4 storage that dequantizes to fp16 only when used."""

    is_svdint4_w4_storage_tensor = True

    def __init__(
        self,
        name: str,
        in_features: int,
        out_features: int,
        tensors: dict[str, torch.Tensor],
        compute_dtype: torch.dtype,
    ):
        self.name = name
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.shape = torch.Size((self.out_features, self.in_features))
        self.dtype = compute_dtype
        self.requires_grad = False
        _validate_w4_storage_layer_tensors(name, tensors, self.in_features, self.out_features)
        qweight = tensors["w4_qweight"].to(torch.uint8).contiguous()
        scales = tensors["w4_scales"].contiguous()
        zeros = tensors.get("w4_zeros")
        if zeros is not None:
            zeros = zeros.contiguous()
        group_size = _infer_w4_storage_group_size(name, self.in_features, int(scales.shape[1]))
        if _HAS_COMFY_QUANTIZED_TENSOR:
            params = W4StorageParams(
                scales=scales,
                orig_dtype=compute_dtype,
                orig_shape=(self.out_features, self.in_features),
                zeros=zeros,
                group_size=group_size,
                name=name,
            )
            self.tensor = QuantizedTensor(qweight, W4_STORAGE_LAYOUT_NAME, params)
        else:
            self.tensor = None
        self.packed_nbytes = _tensor_nbytes(qweight) + _tensor_nbytes(scales) + _tensor_nbytes(zeros)

    @property
    def nbytes(self) -> int:
        return self.packed_nbytes

    def numel(self) -> int:
        return self.out_features * self.in_features

    def element_size(self) -> int:
        return torch.tensor([], dtype=self.dtype).element_size()


class SVDInt4LinearOp(comfy.ops.manual_cast.Linear):
    packed_layer_names: frozenset[str] = frozenset()
    packed_layer_tensors: dict[str, dict[str, torch.Tensor]] = {}
    w4_layer_names: frozenset[str] = frozenset()
    w4_layer_tensors: dict[str, dict[str, torch.Tensor]] = {}

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        torch.nn.Module.__init__(self)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.comfy_need_lazy_init_bias = bool(bias)
        self.weight_comfy_model_dtype = dtype
        self.bias_comfy_model_dtype = dtype
        self.weight_function = []
        self.bias_function = []
        self.weight = None
        self.bias = None
        self.is_svdint4 = False
        self.is_w4_storage = False
        self.compute_dtype = COMPUTE_DTYPE
        self.packed_name = None
        self.packed_weight: SVDInt4PackedTensor | None = None
        self.w4_weight: W4StorageTensor | None = None
        self._svdint4_adapter_lora_overlays: list[SVDInt4AdapterOverlay] = []
        self._svdint4_adapter_staging_runtime: SVDInt4AdapterStagingRuntime | None = None
        self._svdint4_adapter_lora_warnings: set[str] = set()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        name = prefix[:-1] if prefix.endswith(".") else prefix
        if name in self.packed_layer_names:
            try:
                self._load_svdint4(name)
            except Exception as exc:
                LOG.warning("SVDInt4 layer %s could not be loaded: %s", name, exc)
                raise
            return
        if name in self.w4_layer_names:
            try:
                self._load_w4_storage(name, state_dict, prefix, local_metadata, missing_keys)
            except Exception as exc:
                LOG.warning("W4 storage layer %s could not be loaded: %s", name, exc)
                raise
            return
        self._load_dense(state_dict, prefix, local_metadata, missing_keys, unexpected_keys)

    def _load_dense(self, state_dict, prefix, local_metadata, missing_keys, unexpected_keys) -> None:
        assign_to_params_buffers = local_metadata.get("assign_to_params_buffers", False)
        found_weight = False
        found_bias = False
        for key, value in state_dict.items():
            if not key.startswith(prefix):
                continue
            suffix = key[len(prefix):]
            if "." in suffix:
                continue
            if suffix == "weight":
                self.weight = nn.Parameter(value if assign_to_params_buffers else value.clone(), requires_grad=False)
                found_weight = True
            elif suffix == "bias" and value is not None:
                self.bias = nn.Parameter(value if assign_to_params_buffers else value.clone(), requires_grad=False)
                found_bias = True
            else:
                unexpected_keys.append(key)

        if not found_weight:
            self.weight = nn.Parameter(torch.zeros((self.out_features, self.in_features)), requires_grad=False)
            missing_keys.append(prefix + "weight")
        if self.comfy_need_lazy_init_bias and not found_bias:
            self.bias = nn.Parameter(torch.zeros((self.out_features,)), requires_grad=False)
            missing_keys.append(prefix + "bias")

    def _load_w4_storage(self, name: str, state_dict, prefix, local_metadata, missing_keys) -> None:
        tensors = self.w4_layer_tensors.pop(name, None)
        if tensors is None:
            raise RuntimeError(f"W4 storage tensors for {name} were not preloaded")
        self.is_w4_storage = True
        self.compute_dtype = COMPUTE_DTYPE
        self.packed_name = name
        self.w4_weight = W4StorageTensor(
            name=name,
            in_features=self.in_features,
            out_features=self.out_features,
            tensors=tensors,
            compute_dtype=self.compute_dtype,
        )
        if self.w4_weight.tensor is None:
            raise RuntimeError("W4 storage weights require ComfyUI QuantizedTensor support")
        self.weight = nn.Parameter(self.w4_weight.tensor, requires_grad=False)

        assign_to_params_buffers = local_metadata.get("assign_to_params_buffers", False)
        bias = state_dict.get(prefix + "bias")
        if bias is not None:
            self.bias = nn.Parameter(bias if assign_to_params_buffers else bias.clone(), requires_grad=False)
        elif self.comfy_need_lazy_init_bias:
            self.bias = nn.Parameter(torch.zeros((self.out_features,)), requires_grad=False)
            missing_keys.append(prefix + "bias")
        else:
            self.bias = None

    def _load_svdint4(self, name: str) -> None:
        tensors = self.packed_layer_tensors.pop(name, None)
        if tensors is None:
            raise RuntimeError(f"SVDInt4 packed tensors for {name} were not preloaded")
        self.is_svdint4 = True
        self.compute_dtype = COMPUTE_DTYPE
        self.packed_name = name
        self.has_bias_packed = "bias_packed" in tensors
        self.bias = None
        self.packed_weight = SVDInt4PackedTensor(
            name=name,
            in_features=self.in_features,
            out_features=self.out_features,
            tensors=tensors,
            compute_dtype=self.compute_dtype,
        )
        if self.packed_weight.tensor is None:
            raise RuntimeError("SVDInt4 requires ComfyUI QuantizedTensor support")
        self.weight = nn.Parameter(self.packed_weight.tensor, requires_grad=False)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        if self.is_w4_storage:
            if self.weight is None or self.bias is None and self.comfy_need_lazy_init_bias:
                return
            if not isinstance(self.weight, QuantizedTensor) or getattr(self.weight, "_layout_cls", None) != W4_STORAGE_LAYOUT_NAME:
                raise RuntimeError("W4 storage Linear expected a ComfyUI-managed W4 QuantizedTensor weight")
            params = self.weight._params
            destination[prefix + "w4_qweight"] = self.weight._qdata if keep_vars else self.weight._qdata.detach()
            destination[prefix + "w4_scales"] = params.scales if keep_vars else params.scales.detach()
            if params.zeros is not None:
                destination[prefix + "w4_zeros"] = params.zeros if keep_vars else params.zeros.detach()
            if self.bias is not None:
                destination[prefix + "bias"] = self.bias if keep_vars else self.bias.detach()
            return
        if not self.is_svdint4:
            return nn.Module._save_to_state_dict(self, destination, prefix, keep_vars)
        if self.packed_weight is None or self.weight is None:
            return
        tensors = self._packed_tensors_from_weight(self.weight)
        for field, value in tensors.items():
            if value is None:
                continue
            destination[prefix + SVDINT4_STATE_PREFIX + field] = value if keep_vars else value.detach()

    def _packed_tensors_from_weight(self, weight) -> dict[str, torch.Tensor | None]:
        if not isinstance(weight, QuantizedTensor) or getattr(weight, "_layout_cls", None) != SVDINT4_LAYOUT_NAME:
            raise RuntimeError("SVDInt4 Linear expected a ComfyUI-managed SVDInt4 QuantizedTensor weight")
        params = weight._params
        return {
            "qweight": weight._qdata,
            "wscales": params.wscales,
            "smooth": params.smooth if params.has_smooth else None,
            "svd_down": params.svd_down,
            "svd_up": params.svd_up,
            "bias_packed": params.bias_packed if params.has_bias_packed else None,
        }

    def _warn_adapter_lora_once(self, code: str, message: str, *args) -> None:
        if code in self._svdint4_adapter_lora_warnings:
            return
        self._svdint4_adapter_lora_warnings.add(code)
        LOG.warning(message, *args)

    def _apply_lora_adapters(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        patches = self._svdint4_adapter_lora_overlays
        if not patches:
            return out
        runtime = self._svdint4_adapter_staging_runtime
        if runtime is not None:
            patches = runtime.prepare(patches, x.device)

        result = out
        for overlay in patches:
            patch_data = overlay.patch_data
            if overlay.offset is not None:
                self._warn_adapter_lora_once(
                    "lora_offset",
                    "SVDInt4 adapter LoRA overlay skipped %s: offset patches are not supported",
                    getattr(self, "packed_name", "<unknown>"),
                )
                continue
            if overlay.strength_model != 1.0:
                self._warn_adapter_lora_once(
                    "lora_strength_model",
                    "SVDInt4 adapter LoRA overlay skipped %s: strength_model != 1 is not supported",
                    getattr(self, "packed_name", "<unknown>"),
                )
                continue
            if not isinstance(patch_data, comfy.weight_adapter.WeightAdapterBase):
                self._warn_adapter_lora_once(
                    "lora_non_adapter",
                    "SVDInt4 adapter LoRA overlay skipped %s: only adapter LoRA patches are supported",
                    getattr(self, "packed_name", "<unknown>"),
                )
                continue

            old_attrs = {
                "multiplier": getattr(patch_data, "multiplier", None),
                "is_conv": getattr(patch_data, "is_conv", None),
                "conv_dim": getattr(patch_data, "conv_dim", None),
                "kw_dict": getattr(patch_data, "kw_dict", None),
                "kernel_size": getattr(patch_data, "kernel_size", None),
                "in_channels": getattr(patch_data, "in_channels", None),
                "out_channels": getattr(patch_data, "out_channels", None),
            }
            try:
                patch_data.multiplier = overlay.strength_patch
                patch_data.is_conv = False
                patch_data.conv_dim = 0
                patch_data.kw_dict = {}
                patch_data.kernel_size = (1,)
                patch_data.in_channels = self.in_features
                patch_data.out_channels = self.out_features
                delta = patch_data.h(x, result)
                if overlay.function is not None:
                    delta = overlay.function(delta)
                result = patch_data.g(result + delta.to(dtype=result.dtype))
            except Exception as exc:
                self._warn_adapter_lora_once(
                    "lora_failed",
                    "SVDInt4 adapter LoRA overlay failed for %s and was skipped: %s",
                    getattr(self, "packed_name", "<unknown>"),
                    exc,
                )
            finally:
                for attr, value in old_attrs.items():
                    setattr(patch_data, attr, value)
        return result

    def forward_comfy_cast_weights(self, input):
        if not self.is_svdint4:
            return super().forward_comfy_cast_weights(input)
        if input.device.type != "cuda":
            raise RuntimeError("SVDInt4 Linear requires CUDA input tensors")

        _validate_cuda_kernel_runtime(input)
        original_dtype = input.dtype
        x = input if input.dtype == self.compute_dtype else input.to(self.compute_dtype)
        if len(self.weight_function) == 0 and self.weight is not None and self.weight.device == x.device:
            out = torch.nn.functional.linear(x, self.weight, None)
        else:
            saved_weight_function = self.weight_function
            self.weight_function = []
            weight = None
            offload_stream = None
            try:
                weight, _, offload_stream = comfy.ops.cast_bias_weight(
                    self,
                    input=x,
                    dtype=self.compute_dtype,
                    offloadable=True,
                )
                out = torch.nn.functional.linear(x, weight, None)
            finally:
                self.weight_function = saved_weight_function
                if weight is not None:
                    comfy.ops.uncast_bias_weight(self, weight, None, offload_stream)
        out = self._apply_lora_adapters(x, out)
        if out.dtype != original_dtype:
            out = out.to(original_dtype)
        return out


class SVDInt4Ops(comfy.ops.manual_cast):
    def __init__(
        self,
        packed_layer_tensors: dict[str, dict[str, torch.Tensor]],
        w4_layer_tensors: dict[str, dict[str, torch.Tensor]] | None = None,
    ):
        if w4_layer_tensors is None:
            w4_layer_tensors = {}
        self.Linear = type(
            "Linear",
            (SVDInt4LinearOp,),
            {
                "packed_layer_names": frozenset(packed_layer_tensors),
                "packed_layer_tensors": packed_layer_tensors,
                "w4_layer_names": frozenset(w4_layer_tensors),
                "w4_layer_tensors": w4_layer_tensors,
            },
        )


def build_loader_state_dict(
    model_path: str | Path,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, str],
    dict[str, dict[str, torch.Tensor]],
    dict[str, dict[str, torch.Tensor]],
]:
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"SVDInt4 model file does not exist: {model_path}")
    state_dict: dict[str, torch.Tensor] = {}
    packed_layer_tensors: dict[str, dict[str, torch.Tensor]] = {}
    w4_layer_tensors: dict[str, dict[str, torch.Tensor]] = {}

    with safe_open(model_path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        _validate_metadata(metadata, model_path)
        packed_layers = _collect_packed_layers_from_handle(handle)
        if not packed_layers:
            raise ValueError(f"{model_path} does not contain any complete SVDInt4 Linear layers")
        packed_layer_shapes = {
            name: _validate_packed_layer_shapes(name, fields)
            for name, fields in packed_layers.items()
        }
        w4_layers = _collect_w4_storage_layers_from_handle(handle)

        for key in handle.keys():
            split = _split_packed_key(key)
            if split is not None:
                continue
            if _split_w4_storage_key(key) is not None:
                continue
            state_dict[key] = _owned_cpu_tensor(handle.get_tensor(key))

        for name, fields in packed_layers.items():
            tensors: dict[str, torch.Tensor] = {}
            for field, (key, _) in fields.items():
                tensors[field] = _owned_cpu_tensor(handle.get_tensor(key), _packed_tensor_dtype(field))
            in_features, out_features = packed_layer_shapes[name]
            _validate_packed_layer_tensors(name, tensors, in_features, out_features)
            packed_layer_tensors[name] = tensors

            state_dict[f"{name}.weight"] = torch.empty((out_features, in_features), device="meta", dtype=torch.float16)
            if "bias_packed" in fields:
                state_dict[f"{name}.bias"] = torch.empty((out_features,), device="meta", dtype=torch.float16)

        for name, fields in w4_layers.items():
            tensors = {}
            for field, (key, _) in fields.items():
                tensors[field] = _owned_cpu_tensor(handle.get_tensor(key), _w4_storage_tensor_dtype(field))
            w4_layer_tensors[name] = tensors
            qshape = fields["w4_qweight"][1]
            out_features = int(qshape[0])
            in_features_upper_bound = int(qshape[1]) * 2
            state_dict[f"{name}.weight"] = torch.empty(
                (out_features, in_features_upper_bound),
                device="meta",
                dtype=torch.float16,
            )

    LOG.info(
        "SVDInt4 loaded %d packed Linear layers and %d W4 storage Linear layers from one safetensors open: %s",
        len(packed_layer_tensors),
        len(w4_layer_tensors),
        model_path,
    )
    return state_dict, metadata, packed_layer_tensors, w4_layer_tensors


def _patch_key_name(key) -> str | None:
    if isinstance(key, str):
        return key
    if not isinstance(key, (tuple, list)) or len(key) == 0 or not isinstance(key[0], str):
        return None
    return key[0]


def _patch_key_offset_function(key):
    if isinstance(key, str):
        return None, None
    if not isinstance(key, (tuple, list)) or len(key) < 2:
        return None, None
    offset = key[1]
    function = key[2] if len(key) > 2 else None
    return offset, function


def _ensure_patcher_lora_overlay_state(model_patcher) -> dict:
    pending = getattr(model_patcher, "_svdint4_pending_adapter_lora_overlays", None)
    if pending is None:
        pending = {}
        model_patcher._svdint4_pending_adapter_lora_overlays = pending
        model_patcher._svdint4_adapter_lora_overlay_count = 0
        model_patcher._svdint4_unsupported_adapter_lora_patch_count = 0
    return pending


def _clone_svdint4_lora_overlay_state(src_model_patcher, dst_model_patcher) -> None:
    pending = getattr(src_model_patcher, "_svdint4_pending_adapter_lora_overlays", None)
    if pending is not None:
        dst_model_patcher._svdint4_pending_adapter_lora_overlays = {
            key: overlays[:]
            for key, overlays in pending.items()
        }
        dst_model_patcher._svdint4_adapter_lora_overlay_count = int(
            getattr(src_model_patcher, "_svdint4_adapter_lora_overlay_count", 0)
        )
        dst_model_patcher._svdint4_unsupported_adapter_lora_patch_count = int(
            getattr(src_model_patcher, "_svdint4_unsupported_adapter_lora_patch_count", 0)
        )


def _install_svdint4_patch_filter() -> None:
    add_patches = comfy.model_patcher.ModelPatcher.add_patches
    if getattr(add_patches, "_svdint4_filter_installed", False):
        return

    def svdint4_add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        packed_weight_keys = getattr(self.model, "_svdint4_weight_keys", None)
        if not packed_weight_keys:
            return add_patches(self, patches, strength_patch, strength_model)

        passthrough = {}
        handled = set()
        pending = _ensure_patcher_lora_overlay_state(self)

        overlaid = int(getattr(self, "_svdint4_adapter_lora_overlay_count", 0))
        skipped = int(getattr(self, "_svdint4_unsupported_adapter_lora_patch_count", 0))

        for key, patch_data in patches.items():
            weight_key = _patch_key_name(key)
            if weight_key is None or weight_key not in packed_weight_keys:
                passthrough[key] = patch_data
                continue

            handled.add(key)
            offset, function = _patch_key_offset_function(key)
            if isinstance(patch_data, comfy.weight_adapter.WeightAdapterBase):
                # Keep ComfyUI's adapter implementation, but route packed weights
                # away from dense patching because their .weight is not fp16.
                patch_data = _adapter_to_staging_source(patch_data, COMPUTE_DTYPE)
                pending.setdefault(weight_key, []).append(
                    (strength_patch, patch_data, strength_model, offset, function)
                )
                overlaid += 1
            else:
                skipped += 1

        patched = set(add_patches(self, passthrough, strength_patch, strength_model)) if passthrough else set()
        if handled:
            self._svdint4_adapter_lora_overlay_count = overlaid
            self._svdint4_unsupported_adapter_lora_patch_count = skipped
            self.patches_uuid = uuid.uuid4()
            patched.update(handled)
        return list(patched)

    svdint4_add_patches._svdint4_filter_installed = True
    comfy.model_patcher.ModelPatcher.add_patches = svdint4_add_patches


def _install_svdint4_lora_key_map() -> None:
    model_lora_keys_unet = comfy.lora.model_lora_keys_unet
    if getattr(model_lora_keys_unet, "_svdint4_key_map_installed", False):
        return

    def svdint4_model_lora_keys_unet(model, key_map=None):
        if key_map is None:
            key_map = {}
        key_map = model_lora_keys_unet(model, key_map)
        packed_weight_keys = getattr(model, "_svdint4_weight_keys", None)
        if not packed_weight_keys:
            return key_map

        for key in packed_weight_keys:
            if not key.endswith(".weight"):
                continue
            base = key[:-len(".weight")]
            key_map[base] = key
            if base.startswith("diffusion_model."):
                suffix = base[len("diffusion_model."):]
                key_map[suffix] = key
                key_map[f"lora_unet_{suffix.replace('.', '_')}"] = key
        return key_map

    svdint4_model_lora_keys_unet._svdint4_key_map_installed = True
    comfy.lora.model_lora_keys_unet = svdint4_model_lora_keys_unet


def _attach_lora_adapter_overlays(model_patcher) -> None:
    adapter_count = 0
    adapter_source_bytes = 0
    adapter_staging_bytes = 0
    max_layer_staging_bytes = 0
    pending_overlays = getattr(model_patcher, "_svdint4_pending_adapter_lora_overlays", None)
    if pending_overlays is None:
        pending_overlays = getattr(model_patcher.model, "_svdint4_pending_adapter_lora_overlays", {}) or {}
    runtime = getattr(model_patcher.model, "_svdint4_adapter_staging_runtime", None)
    if runtime is None:
        runtime = SVDInt4AdapterStagingRuntime()
        model_patcher.model._svdint4_adapter_staging_runtime = runtime

    for name, module in model_patcher.model.named_modules():
        if not getattr(module, "is_svdint4", False):
            continue

        overlays = pending_overlays.get(f"{name}.weight", [])
        if overlays:
            prepared_overlays = []
            layer_staging_bytes = 0
            for strength_patch, patch_data, strength_model, offset, function in overlays:
                if isinstance(patch_data, comfy.weight_adapter.WeightAdapterBase):
                    patch_data = _adapter_to_staging_source(patch_data, module.compute_dtype)
                    source_nbytes = _adapter_weights_nbytes(patch_data.weights)
                    staging_nbytes = _adapter_staging_nbytes(patch_data)
                    adapter_source_bytes += source_nbytes
                    adapter_staging_bytes += staging_nbytes
                    layer_staging_bytes += staging_nbytes
                    prepared_overlays.append(
                        SVDInt4AdapterOverlay(
                            strength_patch=strength_patch,
                            patch_data=patch_data,
                            strength_model=strength_model,
                            offset=offset,
                            function=function,
                            staging_nbytes=staging_nbytes,
                            source_nbytes=source_nbytes,
                        )
                    )
            module._svdint4_adapter_lora_overlays = prepared_overlays
            module._svdint4_adapter_staging_runtime = runtime
            max_layer_staging_bytes = max(max_layer_staging_bytes, layer_staging_bytes)
        else:
            module._svdint4_adapter_lora_overlays = []
            module._svdint4_adapter_staging_runtime = runtime
        adapter_count += len(overlays)

    if adapter_count:
        LOG.warning(
            "SVDInt4 attached %d adapter LoRA overlay patch(es), %.2f MB CPU source, "
            "%.2f MB total staged bytes, %.2f MB max per-layer GPU staging buffer. "
            "Adapter overlays are staged on demand as separate fp16 matmul paths and are not fused into the SVDInt4 kernel; "
            "repack the model if a LoRA should become part of the quantized base.",
            adapter_count,
            _mb(adapter_source_bytes),
            _mb(adapter_staging_bytes),
            _mb(max_layer_staging_bytes),
        )
    skipped_count = int(
        getattr(
            model_patcher,
            "_svdint4_unsupported_adapter_lora_patch_count",
            getattr(model_patcher.model, "_svdint4_unsupported_adapter_lora_patch_count", 0),
        )
    )
    if skipped_count:
        LOG.warning(
            "SVDInt4 ignored %d non-adapter patch(es) targeting packed Linear weights; "
            "dense diff/set patches are unsupported for packed weights",
            skipped_count,
        )


def _clear_adapter_staging_runtime(model_patcher, *_) -> None:
    runtime = getattr(model_patcher.model, "_svdint4_adapter_staging_runtime", None)
    if runtime is not None:
        runtime.clear()


def _after_model_load(model_patcher, *_) -> None:
    _attach_lora_adapter_overlays(model_patcher)
    root = model_patcher.model
    try:
        loaded_size = model_patcher.loaded_size()
    except Exception:
        loaded_size = getattr(root, "model_loaded_weight_memory", 0)
    try:
        is_dynamic = model_patcher.is_dynamic()
    except Exception:
        is_dynamic = False
    LOG.info(
        "SVDInt4 load state: reported %.2f MB, loaded %.2f MB, offload buffer %.2f MB, "
        "packed %.2f MB, W4 storage %.2f MB, visible %.2f MB, dynamic %s, lowvram %s",
        _mb(model_patcher.model_size()),
        _mb(loaded_size),
        _mb(getattr(root, "model_offload_buffer_memory", 0)),
        _mb(getattr(root, "_svdint4_packed_bytes", 0)),
        _mb(getattr(root, "_svdint4_w4_storage_bytes", 0)),
        _mb(getattr(root, "_svdint4_packed_state_bytes", 0)),
        is_dynamic,
        getattr(root, "model_lowvram", False),
    )


def load_svdint4_model(
    model_path: str | Path,
    disable_dynamic: bool = False,
):
    if not _HAS_COMFY_QUANTIZED_TENSOR:
        raise RuntimeError(
            "SVDInt4 requires a ComfyUI build with comfy.quant_ops.QuantizedTensor support. "
            "Update ComfyUI before loading SVDInt4 packed models."
        )
    _install_svdint4_patch_filter()
    _install_svdint4_lora_key_map()
    model_path = Path(model_path)
    state_dict, metadata, packed_layer_tensors, w4_layer_tensors = build_loader_state_dict(model_path)
    packed_bytes = sum(_tensor_nbytes(tensor) for fields in packed_layer_tensors.values() for tensor in fields.values())
    w4_storage_bytes = sum(_tensor_nbytes(tensor) for fields in w4_layer_tensors.values() for tensor in fields.values())
    packed_state_bytes = packed_bytes + w4_storage_bytes if _HAS_COMFY_QUANTIZED_TENSOR else 0
    unused_packed_layers: tuple[str, ...] = ()
    unused_w4_layers: tuple[str, ...] = ()
    try:
        model = comfy.sd.load_diffusion_model_state_dict(
            state_dict,
            model_options={"custom_operations": SVDInt4Ops(packed_layer_tensors, w4_layer_tensors)},
            metadata=metadata,
            disable_dynamic=disable_dynamic,
        )
    finally:
        if packed_layer_tensors:
            unused_packed_layers = tuple(sorted(packed_layer_tensors))
            packed_layer_tensors.clear()
        if w4_layer_tensors:
            unused_w4_layers = tuple(sorted(w4_layer_tensors))
            w4_layer_tensors.clear()
    if unused_packed_layers:
        sample = ", ".join(unused_packed_layers[:8])
        suffix = "" if len(unused_packed_layers) <= 8 else ", ..."
        raise RuntimeError(
            f"SVDInt4 model load left {len(unused_packed_layers)} packed Linear layer(s) unused "
            f"({sample}{suffix}). The file layout does not match the model architecture ComfyUI selected."
        )
    if unused_w4_layers:
        sample = ", ".join(unused_w4_layers[:8])
        suffix = "" if len(unused_w4_layers) <= 8 else ", ..."
        raise RuntimeError(
            f"SVDInt4 model load left {len(unused_w4_layers)} W4 storage Linear layer(s) unused "
            f"({sample}{suffix}). The file layout does not match the model architecture ComfyUI selected."
        )
    if model is None:
        raise RuntimeError(f"ComfyUI could not detect a supported model config from {model_path}")
    model.model._svdint4_packed_bytes = packed_bytes
    model.model._svdint4_w4_storage_bytes = w4_storage_bytes
    model.model._svdint4_packed_state_bytes = packed_state_bytes
    model.model._svdint4_pending_adapter_lora_overlays = {}
    model._svdint4_pending_adapter_lora_overlays = {}
    model._svdint4_adapter_lora_overlay_count = 0
    model._svdint4_unsupported_adapter_lora_patch_count = 0
    model.model._svdint4_adapter_lora_overlay_count = 0
    model.model._svdint4_unsupported_adapter_lora_patch_count = 0
    model.model._svdint4_adapter_staging_runtime = SVDInt4AdapterStagingRuntime()
    model.model._svdint4_metadata = metadata
    packed_weight_keys = set()
    for name, module in model.model.named_modules():
        if getattr(module, "is_svdint4", False):
            module.weight_function = []
            packed_weight_keys.add(f"{name}.weight")
    if not packed_weight_keys:
        raise RuntimeError(f"SVDInt4 model load did not install any packed Linear weights from {model_path}")
    model.model._svdint4_weight_keys = frozenset(packed_weight_keys)
    base_size = model.model_size()
    model.size = base_size
    LOG.info(
        "SVDInt4 model accounting: base %.2f MB, packed %.2f MB, reported %.2f MB",
        _mb(base_size),
        _mb(packed_bytes + w4_storage_bytes),
        _mb(model.size),
    )
    if w4_storage_bytes:
        LOG.info(
            "SVDInt4 W4 storage: %.2f MB is kept packed on disk/device and dequantized to fp16 on demand.",
            _mb(w4_storage_bytes),
        )
    LOG.info(
        "SVDInt4 adapter LoRA overlay: automatic with on-demand staging; "
        "ComfyUI WeightAdapter h/g paths are reused, and dense diff/set patches "
        "are unsupported for packed weights."
    )
    LOG.info(
        "SVDInt4 weight management: ComfyUI-managed QuantizedTensor weights; "
        "DynamicVRAM allowed: %s; patcher dynamic: %s.",
        not disable_dynamic,
        model.is_dynamic(),
    )
    model.add_callback(CallbacksMP.ON_LOAD, _after_model_load)
    model.add_callback(CallbacksMP.ON_CLONE, _clone_svdint4_lora_overlay_state)
    model.add_callback(CallbacksMP.ON_DETACH, _clear_adapter_staging_runtime)
    model.add_callback(CallbacksMP.ON_CLEANUP, _clear_adapter_staging_runtime)
    model.cached_patcher_init = (load_svdint4_model, (str(model_path),))
    return model
