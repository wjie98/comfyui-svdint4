from __future__ import annotations

import logging
from pathlib import Path
import weakref

import torch
import torch.nn as nn
from safetensors import safe_open

import comfy.ops
import comfy.sd
from comfy.patcher_extension import CallbacksMP


LOG = logging.getLogger("comfyui-svdint4")

PACKED_FIELDS = {"qweight", "wscales", "smooth", "svd_down", "svd_up", "bias_packed"}
REQUIRED_FIELDS = {"qweight", "wscales", "svd_down", "svd_up"}
SUPPORTED_FORMATS = {"svdint4-dit-safetensors-v1"}
COMPUTE_DTYPE = torch.float16


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


class _SVDInt4PackedMover:
    is_svdint4_packed_mover = True

    def __init__(self, module: "SVDInt4LinearOp"):
        self._module = weakref.ref(module)

    def move_to(self, device=None) -> int:
        if device is None or getattr(device, "type", None) == "cpu":
            module = self._module()
            if module is not None:
                return module._release_gpu_tensors()
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

    def _install_release_mover(self) -> None:
        if not any(getattr(item, "is_svdint4_packed_mover", False) for item in self.weight_function):
            self.weight_function.append(_SVDInt4PackedMover(self))

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
        self._install_release_mover()

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        if not self.is_svdint4:
            return nn.Module._save_to_state_dict(self, destination, prefix, keep_vars)
        nn.Module._save_to_state_dict(self, destination, prefix, keep_vars)

    def _account_gpu_bytes(self, delta: int) -> None:
        owner = self._svdint4_owner_model() if self._svdint4_owner_model is not None else None
        if owner is None or not hasattr(owner, "model_loaded_weight_memory"):
            return
        owner.model_loaded_weight_memory = max(0, owner.model_loaded_weight_memory + delta)

    def _release_gpu_tensors(self) -> int:
        released = sum(self._svdint4_gpu_bytes.values())
        self._svdint4_gpu_tensors.clear()
        self._svdint4_gpu_bytes.clear()
        return released

    def _release_gpu_tensors_accounted(self) -> int:
        released = self._release_gpu_tensors()
        if released:
            self._account_gpu_bytes(-released)
        return released

    def _packed_tensors_for(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        device = x.device
        cached = self._svdint4_gpu_tensors.get(device)
        if cached is not None:
            return cached

        tensors: dict[str, torch.Tensor | None] = {}
        for name, value in self._svdint4_cpu_tensors.items():
            if value is None:
                tensors[name] = None
                continue
            dtype = self.compute_dtype if value.dtype in (torch.float16, torch.bfloat16, torch.float32) else value.dtype
            tensors[name] = value.to(device=device, dtype=dtype, non_blocking=True)

        moved = sum(_tensor_nbytes(value) for value in tensors.values())
        self._svdint4_gpu_tensors[device] = tensors
        self._svdint4_gpu_bytes[device] = moved
        self._account_gpu_bytes(moved)
        return tensors

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


def _release_model_gpu_tensors(model_patcher, *_) -> None:
    released = 0
    for module in model_patcher.model.modules():
        if getattr(module, "is_svdint4", False):
            released += module._release_gpu_tensors_accounted()
    if released:
        LOG.info("SVDInt4 released %.2f MB of cached GPU packed tensors", released / (1024 * 1024))


def _install_model_release_movers(model_patcher, *_) -> None:
    for module in model_patcher.model.modules():
        if getattr(module, "is_svdint4", False):
            module._install_release_mover()


def load_svdint4_model(model_path: str | Path, disable_dynamic: bool = False):
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
    for module in model.model.modules():
        if getattr(module, "is_svdint4", False):
            module._svdint4_owner_model = weakref.ref(model.model)
    base_size = model.model_size()
    model.size = base_size + packed_bytes
    model.add_callback(CallbacksMP.ON_LOAD, _install_model_release_movers)
    model.add_callback(CallbacksMP.ON_DETACH, _release_model_gpu_tensors)
    model.add_callback(CallbacksMP.ON_CLEANUP, _release_model_gpu_tensors)
    model.cached_patcher_init = (load_svdint4_model, (str(model_path),))
    return model
