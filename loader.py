from __future__ import annotations

import dataclasses
import logging
import os
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
SUPPORTED_FORMATS = {"svdint4-dit-single-v2"}
COMPUTE_DTYPE = torch.float16
SVDINT4_LAYOUT_NAME = "SVDInt4PackedLayout"
SVDINT4_STATE_PREFIX = "svdint4_"
SVDINT4_PROFILE = os.environ.get("SVDINT4_PROFILE", "").lower() in {"1", "true", "yes", "on"}
SVDINT4_PROFILE_INTERVAL = max(1, int(os.environ.get("SVDINT4_PROFILE_INTERVAL", "200")))
SVDINT4_ATTENTION_PROFILE_INTERVAL = max(1, int(os.environ.get("SVDINT4_ATTENTION_PROFILE_INTERVAL", "40")))
_SVDINT4_PROFILE_STATS: dict[str, dict[str, object]] = {}
_SVDINT4_PROFILE_CALLS = 0
_SVDINT4_PROFILE_PENDING: list[tuple[str, torch.cuda.Event, torch.cuda.Event, tuple[int, ...]]] = []
_SVDINT4_BASE_PROFILE_STATS: dict[str, dict[str, object]] = {}
_SVDINT4_BASE_PROFILE_CALLS = 0
_SVDINT4_BASE_PROFILE_PENDING: list[tuple[str, torch.cuda.Event, torch.cuda.Event, tuple[int, ...]]] = []
_SVDINT4_ADAPTER_PROFILE_STATS: dict[str, dict[str, object]] = {}
_SVDINT4_ADAPTER_PROFILE_CALLS = 0
_SVDINT4_ADAPTER_PROFILE_PENDING: list[tuple[str, torch.cuda.Event, torch.cuda.Event, tuple[int, ...]]] = []
_SVDINT4_ATTENTION_PROFILE_STATS: dict[str, dict[str, object]] = {}
_SVDINT4_ATTENTION_PROFILE_CALLS = 0
_SVDINT4_ATTENTION_PROFILE_PENDING: list[tuple[str, torch.cuda.Event, torch.cuda.Event, tuple[int, ...]]] = []


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


def _owned_cpu_tensor(value: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    if dtype is not None and value.dtype != dtype:
        return value.to(dtype).contiguous()
    return value.contiguous().clone()


def _packed_tensor_dtype(field: str) -> torch.dtype | None:
    if field in {"wscales", "smooth", "svd_down", "svd_up", "bias_packed"}:
        return COMPUTE_DTYPE
    return None


def _tensor_nbytes(value: torch.Tensor | None) -> int:
    if value is None:
        return 0
    return value.numel() * value.element_size()


def _mb(value: int | float) -> float:
    return float(value) / (1024 * 1024)


def _adapter_weights_to(value, device: torch.device, dtype: torch.dtype):
    if isinstance(value, torch.Tensor):
        if value.dtype in (torch.float16, torch.bfloat16, torch.float32):
            return value.to(device=device, dtype=dtype, non_blocking=True)
        return value.to(device=device, non_blocking=True)
    if isinstance(value, tuple):
        return tuple(_adapter_weights_to(item, device, dtype) for item in value)
    if isinstance(value, list):
        return [_adapter_weights_to(item, device, dtype) for item in value]
    return value


def _adapter_weights_nbytes(value) -> int:
    if isinstance(value, torch.Tensor):
        return _tensor_nbytes(value)
    if isinstance(value, tuple):
        return sum(_adapter_weights_nbytes(item) for item in value)
    if isinstance(value, list):
        return sum(_adapter_weights_nbytes(item) for item in value)
    return 0


def _adapter_to_device(adapter: comfy.weight_adapter.WeightAdapterBase, device: torch.device, dtype: torch.dtype):
    return type(adapter)(adapter.loaded_keys, _adapter_weights_to(adapter.weights, device, dtype))


def _flush_cuda_event_profile(
    pending: list[tuple[str, torch.cuda.Event, torch.cuda.Event, tuple[int, ...]]],
    stats_by_name: dict[str, dict[str, object]],
    call_count: int,
    label: str,
    call_unit: str,
) -> None:
    if not pending:
        return

    pending[-1][2].synchronize()
    for name, start, end, shape in pending:
        elapsed_ms = start.elapsed_time(end)
        stats = stats_by_name.setdefault(name, {"calls": 0, "ms": 0.0, "shape": shape})
        stats["calls"] = int(stats["calls"]) + 1
        stats["ms"] = float(stats["ms"]) + elapsed_ms
        stats["shape"] = shape
    pending.clear()

    total_ms = sum(float(item["ms"]) for item in stats_by_name.values())
    top = sorted(stats_by_name.items(), key=lambda item: float(item[1]["ms"]), reverse=True)[:8]
    summary = "; ".join(
        f"{name}: calls={item['calls']} avg={float(item['ms']) / int(item['calls']):.2f}ms total={float(item['ms']):.1f}ms shape={item['shape']}"
        for name, item in top
    )
    LOG.info(
        "%s after %d %s: total %.2fs; %s",
        label,
        call_count,
        call_unit,
        total_ms / 1000.0,
        summary,
    )


def _record_svdint4_profile(name: str, start: torch.cuda.Event, end: torch.cuda.Event, input_shape: tuple[int, ...]) -> None:
    global _SVDINT4_PROFILE_CALLS
    _SVDINT4_PROFILE_CALLS += 1
    _SVDINT4_PROFILE_PENDING.append((name, start, end, input_shape))
    if _SVDINT4_PROFILE_CALLS % SVDINT4_PROFILE_INTERVAL != 0:
        return

    _flush_svdint4_profile()


def _flush_svdint4_profile() -> None:
    _flush_cuda_event_profile(
        _SVDINT4_PROFILE_PENDING,
        _SVDINT4_PROFILE_STATS,
        _SVDINT4_PROFILE_CALLS,
        "SVDInt4 profile",
        "Linear calls",
    )


def _record_svdint4_base_profile(name: str, start: torch.cuda.Event, end: torch.cuda.Event, input_shape: tuple[int, ...]) -> None:
    global _SVDINT4_BASE_PROFILE_CALLS
    _SVDINT4_BASE_PROFILE_CALLS += 1
    _SVDINT4_BASE_PROFILE_PENDING.append((name, start, end, input_shape))
    if _SVDINT4_BASE_PROFILE_CALLS % SVDINT4_PROFILE_INTERVAL != 0:
        return

    _flush_svdint4_base_profile()


def _flush_svdint4_base_profile() -> None:
    _flush_cuda_event_profile(
        _SVDINT4_BASE_PROFILE_PENDING,
        _SVDINT4_BASE_PROFILE_STATS,
        _SVDINT4_BASE_PROFILE_CALLS,
        "SVDInt4 base Linear profile",
        "base Linear calls",
    )


def _record_svdint4_adapter_profile(name: str, start: torch.cuda.Event, end: torch.cuda.Event, input_shape: tuple[int, ...]) -> None:
    global _SVDINT4_ADAPTER_PROFILE_CALLS
    _SVDINT4_ADAPTER_PROFILE_CALLS += 1
    _SVDINT4_ADAPTER_PROFILE_PENDING.append((name, start, end, input_shape))
    if _SVDINT4_ADAPTER_PROFILE_CALLS % SVDINT4_PROFILE_INTERVAL != 0:
        return

    _flush_svdint4_adapter_profile()


def _flush_svdint4_adapter_profile() -> None:
    _flush_cuda_event_profile(
        _SVDINT4_ADAPTER_PROFILE_PENDING,
        _SVDINT4_ADAPTER_PROFILE_STATS,
        _SVDINT4_ADAPTER_PROFILE_CALLS,
        "SVDInt4 adapter LoRA profile",
        "adapter LoRA overlay calls",
    )


def _attention_profile_key(func, args, kwargs) -> tuple[str, tuple[int, ...]]:
    q = args[0]
    k = args[1]
    heads = kwargs.get("heads", args[3] if len(args) > 3 else None)
    skip_reshape = kwargs.get("skip_reshape", False)
    if skip_reshape:
        q_tokens = int(q.shape[-2])
        k_tokens = int(k.shape[-2])
        dim_head = int(q.shape[-1])
    else:
        q_tokens = int(q.shape[1])
        k_tokens = int(k.shape[1])
        dim_head = int(q.shape[-1] // heads) if heads else int(q.shape[-1])
    shape = (q_tokens, k_tokens, int(heads) if heads else -1, dim_head)
    func_name = getattr(func, "__name__", "attention")
    if q_tokens == k_tokens:
        attn_kind = "self"
    elif k_tokens <= 512:
        attn_kind = "cross_text_or_img"
    else:
        attn_kind = "cross"
    return f"{func_name}:{attn_kind}:q{q_tokens}:k{k_tokens}:h{heads}:d{dim_head}", shape


def _record_svdint4_attention_profile(name: str, start: torch.cuda.Event, end: torch.cuda.Event, shape: tuple[int, ...]) -> None:
    global _SVDINT4_ATTENTION_PROFILE_CALLS
    _SVDINT4_ATTENTION_PROFILE_CALLS += 1
    _SVDINT4_ATTENTION_PROFILE_PENDING.append((name, start, end, shape))
    if _SVDINT4_ATTENTION_PROFILE_CALLS % SVDINT4_ATTENTION_PROFILE_INTERVAL != 0:
        return

    _flush_svdint4_attention_profile()


def _flush_svdint4_attention_profile() -> None:
    _flush_cuda_event_profile(
        _SVDINT4_ATTENTION_PROFILE_PENDING,
        _SVDINT4_ATTENTION_PROFILE_STATS,
        _SVDINT4_ATTENTION_PROFILE_CALLS,
        "SVDInt4 attention profile",
        "attention calls",
    )


def _profile_attention_call(func, *args, **kwargs):
    name, shape = _attention_profile_key(func, args, kwargs)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = func(*args, **kwargs)
    end.record()
    _record_svdint4_attention_profile(name, start, end, shape)
    return out


def _install_attention_profiler(model_patcher) -> None:
    transformer_options = model_patcher.model_options.setdefault("transformer_options", {})
    existing_override = transformer_options.get("optimized_attention_override")
    if getattr(existing_override, "_svdint4_attention_profile_installed", False):
        return

    if existing_override is None:
        def profile_override(func, *args, **kwargs):
            return _profile_attention_call(func, *args, **kwargs)
    else:
        def profile_override(func, *args, **kwargs):
            return _profile_attention_call(lambda *a, **kw: existing_override(func, *a, **kw), *args, **kwargs)

    profile_override._svdint4_attention_profile_installed = True
    transformer_options["optimized_attention_override"] = profile_override


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
            for value in (qweight, wscales, smooth if has_smooth else None, svd_down, svd_up, bias_packed if self.has_bias_packed else None)
        )

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
        self.compute_dtype = COMPUTE_DTYPE
        self.packed_name = None
        self.packed_weight: SVDInt4PackedTensor | None = None
        self._svdint4_adapter_lora_overlays = []
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

        result = out
        for strength_patch, patch_data, strength_model, offset, function in patches:
            if offset is not None:
                self._warn_adapter_lora_once(
                    "lora_offset",
                    "SVDInt4 adapter LoRA overlay skipped %s: offset patches are not supported",
                    getattr(self, "packed_name", "<unknown>"),
                )
                continue
            if strength_model != 1.0:
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
                patch_data.multiplier = strength_patch
                patch_data.is_conv = False
                patch_data.conv_dim = 0
                patch_data.kw_dict = {}
                patch_data.kernel_size = (1,)
                patch_data.in_channels = self.in_features
                patch_data.out_channels = self.out_features
                delta = patch_data.h(x, result)
                if function is not None:
                    delta = function(delta)
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
        if SVDINT4_PROFILE:
            profile_name = getattr(self, "packed_name", "<unknown>")
            profile_shape = tuple(input.shape)
            total_start = torch.cuda.Event(enable_timing=True)
            total_end = torch.cuda.Event(enable_timing=True)
            base_start = torch.cuda.Event(enable_timing=True)
            base_end = torch.cuda.Event(enable_timing=True)
            total_start.record()
        original_dtype = input.dtype
        x = input if input.dtype == self.compute_dtype else input.to(self.compute_dtype)
        if SVDINT4_PROFILE:
            base_start.record()
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
        if SVDINT4_PROFILE:
            base_end.record()

        adapter_start = None
        adapter_end = None
        adapter_count = len(self._svdint4_adapter_lora_overlays)
        if SVDINT4_PROFILE and adapter_count:
            adapter_start = torch.cuda.Event(enable_timing=True)
            adapter_end = torch.cuda.Event(enable_timing=True)
            adapter_start.record()
        out = self._apply_lora_adapters(x, out)
        if SVDINT4_PROFILE and adapter_count:
            adapter_end.record()
        if out.dtype != original_dtype:
            out = out.to(original_dtype)
        if SVDINT4_PROFILE:
            total_end.record()
            _record_svdint4_base_profile(profile_name, base_start, base_end, profile_shape)
            if adapter_start is not None and adapter_end is not None:
                _record_svdint4_adapter_profile(profile_name, adapter_start, adapter_end, profile_shape)
            _record_svdint4_profile(profile_name, total_start, total_end, profile_shape)
        return out


class SVDInt4Ops(comfy.ops.manual_cast):
    def __init__(self, packed_layer_tensors: dict[str, dict[str, torch.Tensor]]):
        self.Linear = type(
            "Linear",
            (SVDInt4LinearOp,),
            {
                "packed_layer_names": frozenset(packed_layer_tensors),
                "packed_layer_tensors": packed_layer_tensors,
            },
        )


def build_loader_state_dict(
    model_path: str | Path,
) -> tuple[dict[str, torch.Tensor], dict[str, str], dict[str, dict[str, torch.Tensor]]]:
    model_path = Path(model_path)
    state_dict: dict[str, torch.Tensor] = {}
    packed_layer_tensors: dict[str, dict[str, torch.Tensor]] = {}

    with safe_open(model_path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        _validate_metadata(metadata, model_path)
        packed_layers = _collect_packed_layers_from_handle(handle)
        if not packed_layers:
            raise ValueError(f"{model_path} does not contain any complete SVDInt4 Linear layers")

        for key in handle.keys():
            split = _split_packed_key(key)
            if split is not None:
                continue
            state_dict[key] = _owned_cpu_tensor(handle.get_tensor(key))

        for name, fields in packed_layers.items():
            tensors: dict[str, torch.Tensor] = {}
            for field, (key, _) in fields.items():
                tensors[field] = _owned_cpu_tensor(handle.get_tensor(key), _packed_tensor_dtype(field))
            packed_layer_tensors[name] = tensors

            qshape = fields["qweight"][1]
            out_features = int(qshape[0])
            in_features = int(qshape[1] * 2)
            state_dict[f"{name}.weight"] = torch.empty((out_features, in_features), device="meta", dtype=torch.float16)
            if "bias_packed" in fields:
                state_dict[f"{name}.bias"] = torch.empty((out_features,), device="meta", dtype=torch.float16)

    LOG.info(
        "SVDInt4 loaded %d packed Linear layers from one safetensors open: %s",
        len(packed_layer_tensors),
        model_path,
    )
    return state_dict, metadata, packed_layer_tensors


def _patch_key_name(key) -> str:
    if isinstance(key, str):
        return key
    return key[0]


def _patch_key_offset_function(key):
    if isinstance(key, str):
        return None, None
    offset = key[1]
    function = key[2] if len(key) > 2 else None
    return offset, function


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
        enable_lora_adapters = bool(getattr(self.model, "_svdint4_enable_lora_adapters", False))
        pending = getattr(self.model, "_svdint4_pending_adapter_lora_overlays", None)
        if pending is None:
            pending = {}
            self.model._svdint4_pending_adapter_lora_overlays = pending

        ignored = int(getattr(self.model, "_svdint4_ignored_adapter_lora_patch_count", 0))
        overlaid = int(getattr(self.model, "_svdint4_adapter_lora_overlay_count", 0))
        skipped = int(getattr(self.model, "_svdint4_unsupported_adapter_lora_patch_count", 0))

        for key, patch_data in patches.items():
            weight_key = _patch_key_name(key)
            if weight_key not in packed_weight_keys:
                passthrough[key] = patch_data
                continue

            handled.add(key)
            offset, function = _patch_key_offset_function(key)
            if enable_lora_adapters and isinstance(patch_data, comfy.weight_adapter.WeightAdapterBase):
                pending.setdefault(weight_key, []).append((strength_patch, patch_data, strength_model, offset, function))
                overlaid += 1
            else:
                if enable_lora_adapters:
                    skipped += 1
                else:
                    ignored += 1

        patched = set(add_patches(self, passthrough, strength_patch, strength_model)) if passthrough else set()
        if handled:
            self.model._svdint4_ignored_adapter_lora_patch_count = ignored
            self.model._svdint4_adapter_lora_overlay_count = overlaid
            self.model._svdint4_unsupported_adapter_lora_patch_count = skipped
            self.patches_uuid = uuid.uuid4()
            patched.update(handled)
        return list(patched)

    svdint4_add_patches._svdint4_filter_installed = True
    comfy.model_patcher.ModelPatcher.add_patches = svdint4_add_patches


def _install_svdint4_lora_key_map() -> None:
    model_lora_keys_unet = comfy.lora.model_lora_keys_unet
    if getattr(model_lora_keys_unet, "_svdint4_key_map_installed", False):
        return

    def svdint4_model_lora_keys_unet(model, key_map={}):
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
    enable_lora_adapters = bool(getattr(model_patcher.model, "_svdint4_enable_lora_adapters", False))
    adapter_count = 0
    adapter_bytes = 0
    pending_overlays = getattr(model_patcher.model, "_svdint4_pending_adapter_lora_overlays", {}) or {}
    for name, module in model_patcher.model.named_modules():
        if not getattr(module, "is_svdint4", False):
            continue

        overlays = pending_overlays.get(f"{name}.weight", [])
        if enable_lora_adapters and overlays:
            prepared_overlays = []
            for strength_patch, patch_data, strength_model, offset, function in overlays:
                if isinstance(patch_data, comfy.weight_adapter.WeightAdapterBase):
                    patch_data = _adapter_to_device(patch_data, module.weight.device, module.compute_dtype)
                    adapter_bytes += _adapter_weights_nbytes(patch_data.weights)
                prepared_overlays.append((strength_patch, patch_data, strength_model, offset, function))
            module._svdint4_adapter_lora_overlays = prepared_overlays
        else:
            module._svdint4_adapter_lora_overlays = []
        adapter_count += len(overlays)

    if not enable_lora_adapters:
        ignored_count = int(getattr(model_patcher.model, "_svdint4_ignored_adapter_lora_patch_count", 0))
        if ignored_count:
            LOG.info(
                "SVDInt4 ignored %d adapter LoRA patch(es) targeting packed Linear weights; enable_lora_adapters to run them as fp16 adapter LoRA overlays",
                ignored_count,
            )
        return

    if adapter_count:
        LOG.warning(
            "SVDInt4 attached %d adapter LoRA overlay patch(es), %.2f MB resident. "
            "Adapter overlays are separate fp16 matmul paths and are not fused into the SVDInt4 kernel; "
            "disable enable_lora_adapters unless you intentionally need an external LoRA.",
            adapter_count,
            _mb(adapter_bytes),
        )
    skipped_count = int(getattr(model_patcher.model, "_svdint4_unsupported_adapter_lora_patch_count", 0))
    if skipped_count:
        LOG.warning(
            "SVDInt4 ignored %d non-adapter patch(es) targeting packed Linear weights; dense diff/set patches are unsupported for packed weights",
            skipped_count,
        )


def _after_model_load(model_patcher, *_) -> None:
    _attach_lora_adapter_overlays(model_patcher)
    root = model_patcher.model
    LOG.info(
        "SVDInt4 load state: reported %.2f MB, loaded %.2f MB, packed %.2f MB, visible %.2f MB",
        _mb(model_patcher.model_size()),
        _mb(getattr(root, "model_loaded_weight_memory", 0)),
        _mb(getattr(root, "_svdint4_packed_bytes", 0)),
        _mb(getattr(root, "_svdint4_packed_state_bytes", 0)),
    )


def _load_svdint4_model_cached(
    model_path: str | Path,
    enable_lora_adapters: bool = False,
    disable_dynamic: bool = True,
):
    return load_svdint4_model(
        model_path,
        disable_dynamic=disable_dynamic,
        enable_lora_adapters=enable_lora_adapters,
    )


def load_svdint4_model(
    model_path: str | Path,
    disable_dynamic: bool = True,
    enable_lora_adapters: bool = False,
):
    _install_svdint4_patch_filter()
    _install_svdint4_lora_key_map()
    model_path = Path(model_path)
    state_dict, metadata, packed_layer_tensors = build_loader_state_dict(model_path)
    packed_bytes = sum(_tensor_nbytes(tensor) for fields in packed_layer_tensors.values() for tensor in fields.values())
    packed_state_bytes = packed_bytes if _HAS_COMFY_QUANTIZED_TENSOR else 0
    try:
        model = comfy.sd.load_diffusion_model_state_dict(
            state_dict,
            model_options={"custom_operations": SVDInt4Ops(packed_layer_tensors)},
            metadata=metadata,
            disable_dynamic=disable_dynamic,
        )
    finally:
        if packed_layer_tensors:
            LOG.warning(
                "SVDInt4 model load left %d packed Linear layers unused; clearing them to release CPU memory",
                len(packed_layer_tensors),
            )
            packed_layer_tensors.clear()
    if model is None:
        raise RuntimeError(f"ComfyUI could not detect a supported model config from {model_path}")
    model.model._svdint4_packed_bytes = packed_bytes
    model.model._svdint4_packed_state_bytes = packed_state_bytes
    model.model._svdint4_enable_lora_adapters = bool(enable_lora_adapters)
    model.model._svdint4_pending_adapter_lora_overlays = {}
    model.model._svdint4_ignored_adapter_lora_patch_count = 0
    model.model._svdint4_adapter_lora_overlay_count = 0
    model.model._svdint4_unsupported_adapter_lora_patch_count = 0
    model.model._svdint4_metadata = metadata
    packed_weight_keys = set()
    for name, module in model.model.named_modules():
        if getattr(module, "is_svdint4", False):
            module.weight_function = []
            packed_weight_keys.add(f"{name}.weight")
    model.model._svdint4_weight_keys = frozenset(packed_weight_keys)
    base_size = model.model_size()
    model.size = base_size
    LOG.info(
        "SVDInt4 model accounting: base %.2f MB, packed %.2f MB, reported %.2f MB",
        _mb(base_size),
        _mb(packed_bytes),
        _mb(model.size),
    )
    LOG.info(
        "SVDInt4 adapter LoRA overlay: %s; dense diff/set patches are unsupported for packed weights.",
        "enabled" if model.model._svdint4_enable_lora_adapters else "disabled",
    )
    if disable_dynamic:
        LOG.info("SVDInt4 uses ComfyUI-managed resident QuantizedTensor weights; DynamicVRAM staging is disabled for this model.")
    if SVDINT4_PROFILE:
        _install_attention_profiler(model)
        LOG.warning(
            "SVDInt4 profiling is enabled; CUDA event timings will be logged every %d Linear calls "
            "(base/adapter/total) and every %d attention calls.",
            SVDINT4_PROFILE_INTERVAL,
            SVDINT4_ATTENTION_PROFILE_INTERVAL,
        )
    model.add_callback(CallbacksMP.ON_LOAD, _after_model_load)
    model.cached_patcher_init = (_load_svdint4_model_cached, (str(model_path), enable_lora_adapters))
    return model
