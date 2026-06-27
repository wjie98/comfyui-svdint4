from __future__ import annotations

import logging
import os
from pathlib import Path
import threading
import weakref

import torch
import torch.nn as nn
from safetensors import safe_open

import comfy.ops
import comfy.sd
import comfy.weight_adapter
from comfy.patcher_extension import CallbacksMP


LOG = logging.getLogger("comfyui-svdint4")

PACKED_FIELDS = {"qweight", "wscales", "smooth", "svd_down", "svd_up", "bias_packed"}
REQUIRED_FIELDS = {"qweight", "wscales", "svd_down", "svd_up"}
SUPPORTED_FORMATS = {"svdint4-dit-safetensors-v1"}
COMPUTE_DTYPE = torch.float16
ACCOUNTING_TOLERANCE_BYTES = 16 * 1024 * 1024
GIB = 1024 * 1024 * 1024
_TRUE_ENV = {"1", "true", "yes", "on"}
_CACHE_MODES = {"auto", "resident", "stream"}
_LORA_POLICIES = {"metadata", "packed_only", "external_bypass", "disabled"}
_ACTIVE_MODEL_ROOTS = weakref.WeakSet()
METADATA_CONTRACT_VERSION = "1"
METADATA_DEFAULTS = {
    "svdint4_contract_version": METADATA_CONTRACT_VERSION,
    "has_internal_svd_lora": "true",
    "lora_policy": "packed_only",
}


def _env_mode(name: str, default: str, valid: set[str]) -> str:
    value = os.environ.get(name, default).strip().lower()
    if value not in valid:
        LOG.warning("%s=%r is invalid; using %s", name, value, default)
        return default
    return value


def _cache_mode() -> str:
    if os.environ.get("SVDINT4_RESIDENT_GPU_CACHE", "").lower() in _TRUE_ENV:
        return "resident"
    return _env_mode("SVDINT4_CACHE_MODE", "auto", _CACHE_MODES)


def _legacy_lora_policy() -> str | None:
    value = os.environ.get("SVDINT4_LORA_BYPASS")
    if value is None:
        return None
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return "external_bypass"
    if value in {"0", "false", "no", "off"}:
        return "disabled"
    if value in _LORA_POLICIES:
        return value
    LOG.warning("SVDINT4_LORA_BYPASS=%r is invalid; ignoring it", value)
    return None


def _metadata_bool(metadata: dict[str, str], key: str, default: bool) -> bool:
    value = metadata.get(key)
    if value is None:
        return default
    return str(value).strip().lower() in _TRUE_ENV


def _metadata_lora_policy(metadata: dict[str, str]) -> str:
    policy = metadata.get("lora_policy")
    if policy in _LORA_POLICIES - {"metadata"}:
        return policy
    if policy is not None:
        LOG.warning("Unsupported SVDInt4 lora_policy=%r; using packed_only", policy)
    return "packed_only" if _metadata_bool(metadata, "has_internal_svd_lora", True) else "external_bypass"


def _resolve_lora_policy(metadata: dict[str, str], requested: str) -> str:
    legacy = _legacy_lora_policy()
    if legacy is not None:
        return _metadata_lora_policy(metadata) if legacy == "metadata" else legacy
    if requested == "metadata":
        return _metadata_lora_policy(metadata)
    if requested not in _LORA_POLICIES:
        LOG.warning("SVDInt4 lora_policy=%r is invalid; using metadata", requested)
        return _metadata_lora_policy(metadata)
    return requested


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


class _SVDInt4PackedMover:
    is_svdint4_packed_mover = True

    def __init__(self, module: "SVDInt4LinearOp"):
        self._module = weakref.ref(module)

    def move_to(self, device=None) -> int:
        if getattr(device, "type", None) == "cuda":
            module = self._module()
            if module is not None and module._use_resident_cache(device):
                return module._cache_tensors_on(device, account_loaded=False)
        if device is None or getattr(device, "type", None) == "cpu":
            module = self._module()
            if module is not None:
                return module._release_gpu_tensors(account_loaded=False)
        return 0


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
        self._svdint4_owner_model = None
        self._svdint4_cpu_tensors: dict[str, torch.Tensor | None] = {}
        self._svdint4_gpu_tensors: dict[torch.device, dict[str, torch.Tensor | None]] = {}
        self._svdint4_gpu_bytes: dict[torch.device, int] = {}
        self._svdint4_cache_lock = threading.RLock()
        self._svdint4_lora_patches = []
        self._svdint4_lora_warnings: set[str] = set()

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
        # Keep dense placeholders out of the live module. ComfyUI LoRA/dynamic
        # loading treats a visible Linear.weight as a full fp16 weight and will
        # allocate/stage dense patch buffers for it, while this module only uses
        # the packed SVDInt4 tensors below.
        self.weight = None
        self.bias = None
        self._svdint4_cpu_tensors = {
            "qweight": tensors["qweight"].contiguous(),
            "wscales": tensors["wscales"].contiguous(),
            "smooth": tensors.get("smooth"),
            "svd_down": tensors["svd_down"].contiguous(),
            "svd_up": tensors["svd_up"].contiguous(),
            "bias_packed": tensors.get("bias_packed"),
        }
        self._svdint4_gpu_tensors = {}
        self._svdint4_gpu_bytes = {}

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        if not self.is_svdint4:
            return nn.Module._save_to_state_dict(self, destination, prefix, keep_vars)
        destination[prefix + "weight"] = torch.empty((0,), device="meta", dtype=self.compute_dtype)
        if self.has_bias_packed:
            destination[prefix + "bias"] = torch.empty((0,), device="meta", dtype=self.compute_dtype)

    def _account_gpu_bytes(self, delta: int, account_loaded: bool) -> None:
        owner = self._svdint4_owner_model() if self._svdint4_owner_model is not None else None
        if owner is None:
            return
        owner._svdint4_cached_gpu_bytes = max(0, getattr(owner, "_svdint4_cached_gpu_bytes", 0) + delta)
        if account_loaded and hasattr(owner, "model_loaded_weight_memory"):
            owner.model_loaded_weight_memory = max(0, owner.model_loaded_weight_memory + delta)

    def _release_gpu_tensors(self, account_loaded: bool) -> int:
        with self._svdint4_cache_lock:
            released = sum(self._svdint4_gpu_bytes.values())
            self._svdint4_gpu_tensors.clear()
            self._svdint4_gpu_bytes.clear()
        if released:
            self._account_gpu_bytes(-released, account_loaded=account_loaded)
        return released

    def _release_gpu_tensors_accounted(self) -> int:
        return self._release_gpu_tensors(account_loaded=True)

    def _copy_tensors_to(self, device: torch.device) -> dict[str, torch.Tensor | None]:
        tensors: dict[str, torch.Tensor | None] = {}
        for name, value in self._svdint4_cpu_tensors.items():
            if value is None:
                tensors[name] = None
                continue
            dtype = self.compute_dtype if value.dtype in (torch.float16, torch.bfloat16, torch.float32) else value.dtype
            tensors[name] = value.to(device=device, dtype=dtype, non_blocking=True)
        return tensors

    def _owner_model(self):
        return self._svdint4_owner_model() if self._svdint4_owner_model is not None else None

    def _use_resident_cache(self, device: torch.device) -> bool:
        owner = self._owner_model()
        if owner is None:
            return False
        mode = _select_cache_policy(owner, device)
        return mode == "resident" and device.type == "cuda"

    def _cache_tensors_on(self, device: torch.device, account_loaded: bool) -> int:
        cached = self._svdint4_gpu_tensors.get(device)
        if cached is not None:
            return 0

        with self._svdint4_cache_lock:
            cached = self._svdint4_gpu_tensors.get(device)
            if cached is not None:
                return 0

            try:
                tensors = self._copy_tensors_to(device)
            except torch.OutOfMemoryError:
                released = 0
                owner = self._owner_model()
                if owner is not None:
                    owner._svdint4_effective_cache_mode = "stream"
                    released = _release_root_gpu_tensors(owner, account_loaded=account_loaded)
                LOG.warning(
                    "SVDInt4 resident cache ran out of VRAM while loading %s; falling back to stream mode",
                    getattr(self, "packed_name", "<unknown>"),
                )
                return -released if not account_loaded else 0

            moved = sum(_tensor_nbytes(value) for value in tensors.values())
            self._svdint4_gpu_tensors[device] = tensors
            self._svdint4_gpu_bytes[device] = moved

        self._account_gpu_bytes(moved, account_loaded=account_loaded)
        LOG.debug("SVDInt4 cached %.2f MB for %s on %s", _mb(moved), getattr(self, "packed_name", "<unknown>"), device)
        return moved

    def _packed_tensors_for(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        device = x.device
        if not self._use_resident_cache(device):
            return self._copy_tensors_to(device)

        self._cache_tensors_on(device, account_loaded=True)
        cached = self._svdint4_gpu_tensors.get(device)
        if cached is None:
            return self._copy_tensors_to(device)
        return cached

    def _warn_lora_once(self, code: str, message: str, *args) -> None:
        if code in self._svdint4_lora_warnings:
            return
        self._svdint4_lora_warnings.add(code)
        LOG.warning(message, *args)

    def _apply_lora_bypass(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        patches = self._svdint4_lora_patches
        if not patches:
            return out

        result = out
        for strength_patch, patch_data, strength_model, offset, function in patches:
            if offset is not None:
                self._warn_lora_once(
                    "lora_offset",
                    "SVDInt4 bypass LoRA skipped %s: offset patches are not supported",
                    getattr(self, "packed_name", "<unknown>"),
                )
                continue
            if strength_model != 1.0:
                self._warn_lora_once(
                    "lora_strength_model",
                    "SVDInt4 bypass LoRA skipped %s: strength_model != 1 is not supported",
                    getattr(self, "packed_name", "<unknown>"),
                )
                continue
            if not isinstance(patch_data, comfy.weight_adapter.WeightAdapterBase):
                self._warn_lora_once(
                    "lora_non_adapter",
                    "SVDInt4 bypass LoRA skipped %s: only adapter LoRA patches are supported",
                    getattr(self, "packed_name", "<unknown>"),
                )
                continue

            old_weights = patch_data.weights
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
                patch_data.weights = _adapter_weights_to(old_weights, x.device, x.dtype)
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
                self._warn_lora_once(
                    "lora_failed",
                    "SVDInt4 bypass LoRA failed for %s and was skipped: %s",
                    getattr(self, "packed_name", "<unknown>"),
                    exc,
                )
            finally:
                patch_data.weights = old_weights
                for attr, value in old_attrs.items():
                    setattr(patch_data, attr, value)
        return result

    def forward_comfy_cast_weights(self, input):
        if not self.is_svdint4:
            return super().forward_comfy_cast_weights(input)
        if input.device.type != "cuda":
            raise RuntimeError("SVDInt4 Linear requires CUDA input tensors")

        _validate_cuda_kernel_runtime(input)
        svd_int4_linear = _load_svdint4_linear()
        original_dtype = input.dtype
        x = input if input.dtype == self.compute_dtype else input.to(self.compute_dtype)
        tensors = self._packed_tensors_for(x)
        out = svd_int4_linear(
            x,
            tensors["qweight"],
            tensors["wscales"],
            tensors["svd_down"],
            tensors["svd_up"],
            smooth_packed=tensors["smooth"],
            bias_packed=tensors["bias_packed"],
            out_features=self.out_features,
        )
        out = self._apply_lora_bypass(x, out)
        return out if out.dtype == original_dtype else out.to(original_dtype)


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
        metadata = {**METADATA_DEFAULTS, **(handle.metadata() or {})}
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


def _release_root_gpu_tensors(root, account_loaded: bool) -> int:
    released = 0
    for module in root.modules():
        if getattr(module, "is_svdint4", False):
            released += module._release_gpu_tensors(account_loaded=account_loaded)
    return released


def _release_model_gpu_tensors(model_patcher, *_) -> None:
    released = _release_root_gpu_tensors(model_patcher.model, account_loaded=True)
    if released:
        LOG.info("SVDInt4 released %.2f MB of cached GPU packed tensors", _mb(released))


def _release_other_model_gpu_tensors(root) -> int:
    released = 0
    for other in list(_ACTIVE_MODEL_ROOTS):
        if other is root:
            continue
        released += _release_root_gpu_tensors(other, account_loaded=True)
    if released:
        LOG.info("SVDInt4 released %.2f MB of cached GPU packed tensors from inactive model(s)", _mb(released))
    return released


def _cache_headroom_bytes(total: int) -> int:
    return max(4 * GIB, total // 4)


def _select_cache_policy(root, device: torch.device) -> str:
    device_key = str(device)
    selected = getattr(root, "_svdint4_effective_cache_mode", None)
    if selected in {"resident", "stream"} and getattr(root, "_svdint4_cache_policy_device", None) == device_key:
        return selected

    requested = getattr(root, "_svdint4_requested_cache_mode", "auto")
    if requested in {"resident", "stream"}:
        root._svdint4_effective_cache_mode = requested
        root._svdint4_cache_policy_device = device_key
        if requested == "resident":
            _release_other_model_gpu_tensors(root)
        return requested

    packed_total = int(getattr(root, "_svdint4_packed_bytes", 0))
    if device is None or getattr(device, "type", None) != "cuda" or packed_total <= 0:
        root._svdint4_effective_cache_mode = "stream"
        root._svdint4_cache_policy_device = device_key
        return "stream"

    _release_other_model_gpu_tensors(root)
    try:
        free, total = torch.cuda.mem_get_info(device)
    except Exception as exc:
        LOG.warning("SVDInt4 could not query CUDA free memory for cache policy: %s", exc)
        root._svdint4_effective_cache_mode = "stream"
        root._svdint4_cache_policy_device = device_key
        return "stream"

    headroom = _cache_headroom_bytes(total)
    if free >= packed_total + headroom:
        root._svdint4_effective_cache_mode = "resident"
    else:
        root._svdint4_effective_cache_mode = "stream"
    root._svdint4_cache_policy_device = device_key
    LOG.info(
        "SVDInt4 cache policy: requested auto -> %s (free %.2f MB, packed %.2f MB, headroom %.2f MB)",
        root._svdint4_effective_cache_mode,
        _mb(free),
        _mb(packed_total),
        _mb(headroom),
    )
    return root._svdint4_effective_cache_mode


def _normalize_model_accounting(model_patcher) -> None:
    root = model_patcher.model
    packed_total = int(getattr(root, "_svdint4_packed_bytes", 0))
    cached = int(getattr(root, "_svdint4_cached_gpu_bytes", 0))
    unmaterialized = max(0, packed_total - cached)
    if packed_total <= 0 or unmaterialized <= 0:
        return

    loaded = int(getattr(root, "model_loaded_weight_memory", 0))
    reported = int(model_patcher.model_size())
    if loaded >= reported - ACCOUNTING_TOLERANCE_BYTES:
        root.model_loaded_weight_memory = max(0, loaded - unmaterialized)
        LOG.info(
            "SVDInt4 corrected full-load accounting: loaded %.2f MB -> %.2f MB "
            "(packed %.2f MB, cached %.2f MB)",
            _mb(loaded),
            _mb(root.model_loaded_weight_memory),
            _mb(packed_total),
            _mb(cached),
        )


def _attach_lora_bypass_patches(model_patcher) -> None:
    policy = getattr(model_patcher.model, "_svdint4_lora_policy", "packed_only")
    adapter_count = 0
    skipped_count = 0
    pending: list[tuple[SVDInt4LinearOp, list]] = []
    for name, module in model_patcher.model.named_modules():
        if not getattr(module, "is_svdint4", False):
            continue

        patches = model_patcher.patches.get(f"{name}.weight", [])
        bypass_patches = []
        for patch in patches:
            patch_data = patch[1]
            if isinstance(patch_data, comfy.weight_adapter.WeightAdapterBase):
                bypass_patches.append(patch)
            else:
                skipped_count += 1
        pending.append((module, bypass_patches))
        adapter_count += len(bypass_patches)

    if policy in {"packed_only", "disabled"}:
        for module, _ in pending:
            module._svdint4_lora_patches = []
        if adapter_count or skipped_count:
            LOG.info(
                "SVDInt4 LoRA policy %s ignored %d adapter patch(es) targeting packed Linear weights",
                policy,
                adapter_count,
            )
        return

    for module, bypass_patches in pending:
        module._svdint4_lora_patches = bypass_patches

    if adapter_count:
        LOG.info("SVDInt4 attached %d LoRA adapter patches as forward bypass paths", adapter_count)
    if skipped_count:
        LOG.warning(
            "SVDInt4 ignored %d non-adapter LoRA patches; dense diff/set patches are not supported for packed weights",
            skipped_count,
        )


def _after_model_load(model_patcher, *_) -> None:
    _ACTIVE_MODEL_ROOTS.add(model_patcher.model)
    _attach_lora_bypass_patches(model_patcher)
    _normalize_model_accounting(model_patcher)
    root = model_patcher.model
    LOG.info(
        "SVDInt4 load state: cache %s, reported %.2f MB, loaded %.2f MB, packed %.2f MB, cached %.2f MB",
        getattr(root, "_svdint4_effective_cache_mode", None) or "stream",
        _mb(model_patcher.model_size()),
        _mb(getattr(root, "model_loaded_weight_memory", 0)),
        _mb(getattr(root, "_svdint4_packed_bytes", 0)),
        _mb(getattr(root, "_svdint4_cached_gpu_bytes", 0)),
    )


def load_svdint4_model(
    model_path: str | Path,
    disable_dynamic: bool = False,
    cache_mode: str = "auto",
    lora_policy: str = "metadata",
):
    model_path = Path(model_path)
    state_dict, metadata, packed_layer_tensors = build_loader_state_dict(model_path)
    packed_bytes = sum(_tensor_nbytes(tensor) for fields in packed_layer_tensors.values() for tensor in fields.values())
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
    model.model._svdint4_cached_gpu_bytes = 0
    model.model._svdint4_requested_cache_mode = cache_mode if cache_mode in _CACHE_MODES else _cache_mode()
    model.model._svdint4_effective_cache_mode = None
    model.model._svdint4_cache_policy_device = None
    model.model._svdint4_lora_policy = _resolve_lora_policy(metadata, lora_policy)
    model.model._svdint4_metadata = metadata
    _ACTIVE_MODEL_ROOTS.add(model.model)
    for name, module in model.model.named_modules():
        if getattr(module, "is_svdint4", False):
            module._svdint4_owner_model = weakref.ref(model.model)
            model.add_weight_wrapper(f"{name}.weight", _SVDInt4PackedMover(module))
    base_size = model.model_size()
    model.size = base_size + packed_bytes
    LOG.info(
        "SVDInt4 model accounting: base %.2f MB, packed %.2f MB, reported %.2f MB",
        _mb(base_size),
        _mb(packed_bytes),
        _mb(model.size),
    )
    LOG.info("SVDInt4 cache policy requested: %s", model.model._svdint4_requested_cache_mode)
    LOG.info(
        "SVDInt4 LoRA policy: %s; standard adapter LoRAs can run as forward bypass paths only under external_bypass, "
        "dense diff/set LoRA patches are unsupported for packed weights.",
        model.model._svdint4_lora_policy,
    )
    model.add_callback(CallbacksMP.ON_LOAD, _after_model_load)
    model.add_callback(CallbacksMP.ON_DETACH, _release_model_gpu_tensors)
    model.add_callback(CallbacksMP.ON_CLEANUP, _release_model_gpu_tensors)
    model.cached_patcher_init = (load_svdint4_model, (str(model_path), disable_dynamic, cache_mode, lora_policy))
    return model
