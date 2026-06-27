from __future__ import annotations

import logging
import os
import platform
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors import safe_open

import comfy.ops
import comfy.sd


LOG = logging.getLogger("comfyui-svdint4")

PACKED_FIELDS = {"qweight", "wscales", "smooth", "svd_down", "svd_up", "bias_packed"}
REQUIRED_FIELDS = {"qweight", "wscales", "svd_down", "svd_up"}
SUPPORTED_FORMATS = {"svdint4-dit-safetensors-v1"}
WINDOWS_KERNEL_OPT_IN = "SVDINT4_ENABLE_EXPERIMENTAL_WINDOWS_KERNEL"


def _kernel_install_hint() -> str:
    local_kernel = Path(__file__).resolve().parent / "kernel"
    if local_kernel.is_dir():
        return f"python -m pip install -v --no-build-isolation -e {local_kernel}"
    return "python -m pip install --no-build-isolation svdint4-kernel"


def _load_svdint4_linear():
    if platform.system() == "Windows" and os.environ.get(WINDOWS_KERNEL_OPT_IN) != "1":
        raise RuntimeError(
            "The SVDInt4 CUDA kernel is disabled by default on Windows because it has not "
            "completed production validation there and may trigger a driver-level crash. "
            f"Set {WINDOWS_KERNEL_OPT_IN}=1 only if you are intentionally testing the "
            "experimental Windows kernel path. Linux remains the supported runtime path."
        )
    try:
        from svdint4.ops import svd_int4_linear
    except Exception as exc:
        raise RuntimeError(
            "SVDInt4 kernel is not installed or failed to load in the ComfyUI environment. "
            "Install it with the ComfyUI environment's Torch, for example: "
            f"SVDINT4_ARCH_LIST='7.5;8.0;8.6;8.9' {_kernel_install_hint()}"
        ) from exc
    return svd_int4_linear


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


def _collect_packed_layers(model_path: Path) -> dict[str, dict[str, tuple[str, tuple[int, ...]]]]:
    layers: dict[str, dict[str, tuple[str, tuple[int, ...]]]] = {}
    with safe_open(model_path, framework="pt", device="cpu") as handle:
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


def _load_layer_tensors(model_path: Path, name: str) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(model_path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for field in PACKED_FIELDS:
            key = f"{name}.{field}"
            if key in keys:
                tensors[field] = handle.get_tensor(key)

    missing = REQUIRED_FIELDS - tensors.keys()
    if missing:
        raise KeyError(f"{name} missing required SVDInt4 tensors: {sorted(missing)}")
    return tensors


def _optional_float(value: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor | None:
    if value is None:
        return None
    if value.dtype in (torch.float16, torch.bfloat16, torch.float32):
        return value.to(dtype).contiguous()
    return value.contiguous()


def _infer_compute_dtype(tensors: dict[str, torch.Tensor]) -> torch.dtype:
    for key in ("wscales", "svd_down", "svd_up", "smooth", "bias_packed"):
        value = tensors.get(key)
        if value is not None and value.dtype in (torch.float16, torch.bfloat16):
            return value.dtype
    return torch.bfloat16


class SVDInt4LinearOp(comfy.ops.manual_cast.Linear):
    model_path: Path | None = None
    packed_layer_names: frozenset[str] = frozenset()

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
        self.compute_dtype = torch.bfloat16

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
        if self.model_path is None:
            raise RuntimeError("SVDInt4 Linear missing model_path")
        tensors = _load_layer_tensors(self.model_path, name)
        compute_dtype = _infer_compute_dtype(tensors)
        self.is_svdint4 = True
        self.compute_dtype = compute_dtype
        self.packed_name = name
        self.weight = torch.empty((self.out_features, self.in_features), device="meta", dtype=compute_dtype)
        self.bias = torch.empty((self.out_features,), device="meta", dtype=compute_dtype) if "bias_packed" in tensors else None
        self.register_buffer("qweight", tensors["qweight"].contiguous())
        self.register_buffer("wscales", tensors["wscales"].to(compute_dtype).contiguous())
        self.register_buffer("smooth", _optional_float(tensors.get("smooth"), compute_dtype))
        self.register_buffer("svd_down", tensors["svd_down"].to(compute_dtype).contiguous())
        self.register_buffer("svd_up", tensors["svd_up"].to(compute_dtype).contiguous())
        self.register_buffer("bias_packed", _optional_float(tensors.get("bias_packed"), compute_dtype))

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        if not self.is_svdint4:
            return nn.Module._save_to_state_dict(self, destination, prefix, keep_vars)
        nn.Module._save_to_state_dict(self, destination, prefix, keep_vars)
        destination[prefix + "weight"] = torch.empty((0,), device="meta", dtype=self.compute_dtype)
        if self.bias is not None:
            destination[prefix + "bias"] = torch.empty((0,), device="meta", dtype=self.compute_dtype)

    def _move_buffers_for(self, x: torch.Tensor) -> None:
        for name, value in self._buffers.items():
            if value is None:
                continue
            dtype = self.compute_dtype if value.dtype in (torch.float16, torch.bfloat16, torch.float32) else value.dtype
            if value.device != x.device or value.dtype != dtype:
                self._buffers[name] = value.to(device=x.device, dtype=dtype, non_blocking=True)

    def forward_comfy_cast_weights(self, input):
        if not self.is_svdint4:
            return super().forward_comfy_cast_weights(input)
        if input.device.type != "cuda":
            raise RuntimeError("SVDInt4 Linear requires CUDA input tensors")

        svd_int4_linear = _load_svdint4_linear()
        original_dtype = input.dtype
        x = input if input.dtype == self.compute_dtype else input.to(self.compute_dtype)
        self._move_buffers_for(x)
        out = svd_int4_linear(
            x,
            self.qweight,
            self.wscales,
            self.svd_down,
            self.svd_up,
            smooth_packed=self.smooth,
            bias_packed=self.bias_packed,
            out_features=self.out_features,
        )
        return out if out.dtype == original_dtype else out.to(original_dtype)


class SVDInt4Ops(comfy.ops.manual_cast):
    def __init__(self, model_path: str | Path, packed_layer_names: set[str]):
        self.Linear = type(
            "Linear",
            (SVDInt4LinearOp,),
            {"model_path": Path(model_path), "packed_layer_names": frozenset(packed_layer_names)},
        )


def build_loader_state_dict(model_path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, str], set[str]]:
    model_path = Path(model_path)
    metadata = _model_metadata(model_path)
    _validate_metadata(metadata, model_path)
    packed_layers = _collect_packed_layers(model_path)
    if not packed_layers:
        raise ValueError(f"{model_path} does not contain any complete SVDInt4 Linear layers")
    state_dict: dict[str, torch.Tensor] = {}

    with safe_open(model_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            split = _split_packed_key(key)
            if split is not None:
                continue
            state_dict[key] = handle.get_tensor(key)

    for name, fields in packed_layers.items():
        qshape = fields["qweight"][1]
        out_features = int(qshape[0])
        in_features = int(qshape[1] * 2)
        state_dict[f"{name}.weight"] = torch.empty((out_features, in_features), device="meta", dtype=torch.float16)
        if "bias_packed" in fields:
            state_dict[f"{name}.bias"] = torch.empty((out_features,), device="meta", dtype=torch.float16)

    LOG.info("SVDInt4 loaded %d packed Linear layers from %s", len(packed_layers), model_path)
    return state_dict, metadata, set(packed_layers)


def load_svdint4_model(model_path: str | Path, disable_dynamic: bool = False):
    model_path = Path(model_path)
    state_dict, metadata, packed_layer_names = build_loader_state_dict(model_path)
    model = comfy.sd.load_diffusion_model_state_dict(
        state_dict,
        model_options={"custom_operations": SVDInt4Ops(model_path, packed_layer_names)},
        metadata=metadata,
        disable_dynamic=disable_dynamic,
    )
    if model is None:
        raise RuntimeError(f"ComfyUI could not detect a supported model config from {model_path}")
    model.cached_patcher_init = (load_svdint4_model, (str(model_path),))
    return model
