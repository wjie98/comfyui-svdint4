from __future__ import annotations

import logging
import os
from pathlib import Path

import comfy.context_windows
import comfy.patcher_extension
import folder_paths
import torch
from safetensors import safe_open


LOG = logging.getLogger("comfyui-svdint4")
FOLDER_NAME = "diffusion_models"
MODEL_EXTENSIONS = {".safetensors", ".sft"}
ENV_PATHS = ("SVDINT4_DIT_PATHS",)
SUPPORTED_FORMATS = {"svdint4-dit-single-v2"}
_BERNINI_ROPE_WRAPPER_KEY = "svdint4_bernini_context_rope"


def _model_dirs() -> list[str]:
    return folder_paths.get_folder_paths(FOLDER_NAME)


def _register_extra_model_dirs() -> None:
    changed = False
    for env_name in ENV_PATHS:
        for item in os.environ.get(env_name, "").split(os.pathsep):
            if not item:
                continue
            path = Path(item).expanduser()
            if not path.is_dir():
                LOG.warning("Ignoring %s entry because it is not a directory: %s", env_name, item)
                continue
            before = _model_dirs()
            folder_paths.add_model_folder_path(FOLDER_NAME, str(path))
            changed = changed or before != _model_dirs()
    if changed:
        folder_paths.filename_list_cache.pop(FOLDER_NAME, None)


def _model_names() -> list[str]:
    _register_extra_model_dirs()
    names: list[str] = []
    for name in folder_paths.get_filename_list(FOLDER_NAME):
        if Path(name).suffix.lower() not in MODEL_EXTENSIONS:
            continue
        path = folder_paths.get_full_path(FOLDER_NAME, name)
        if path is not None and _is_svdint4_file(path):
            names.append(name)
    return names


def _is_svdint4_file(model_path: str | Path) -> bool:
    try:
        with safe_open(model_path, framework="pt", device="cpu") as handle:
            return (handle.metadata() or {}).get("format") in SUPPORTED_FORMATS
    except Exception:
        return False


def _resolve_model_path(model_name: str) -> str:
    _register_extra_model_dirs()
    return folder_paths.get_full_path_or_raise(FOLDER_NAME, model_name)


def _is_wan_frame_count(frame_count: int) -> bool:
    return frame_count >= 1 and (frame_count - 1) % 4 == 0


def _ceil_wan_frame_count(frame_count: int) -> int:
    if frame_count < 1:
        raise ValueError("Video must contain at least one frame.")
    return ((frame_count - 1 + 3) // 4) * 4 + 1


class SVDInt4DiffusionModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (
                    _model_names(),
                    {
                        "tooltip": (
                            "SVDInt4 DiT file from ComfyUI/models/diffusion_models. "
                            "Only supported SVDInt4 single-file safetensors assets are shown."
                        )
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_diffusion_model"
    CATEGORY = "SVDInt4/loaders"
    TITLE = "Load SVDInt4 DiT"

    def load_diffusion_model(
        self,
        unet_name: str,
    ):
        from .loader import load_svdint4_model

        return (load_svdint4_model(_resolve_model_path(unet_name)),)


class BerniniPadVideoLength:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE",),
                "target_length": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 16385,
                        "step": 1,
                        "tooltip": (
                            "Target real-frame count. Use 0 to round the input length up "
                            "to the next 4*n+1 frame count."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("video", "length", "input_length")
    FUNCTION = "pad"
    CATEGORY = "SVDInt4/video"
    TITLE = "Bernini Pad Video Length"

    def pad(self, video, target_length: int):
        frame_count = int(video.shape[0])
        if frame_count < 1:
            raise ValueError("Bernini Pad Video Length requires at least one input frame.")

        if target_length == 0:
            output_length = _ceil_wan_frame_count(frame_count)
        else:
            output_length = int(target_length)
            if not _is_wan_frame_count(output_length):
                raise ValueError(
                    f"target_length must be 0 or 4*n+1; got {target_length}."
                )
            if output_length < frame_count:
                raise ValueError(
                    f"target_length={output_length} is shorter than the input video "
                    f"({frame_count} frames). This node only pads; trim upstream if needed."
                )

        pad_count = output_length - frame_count
        if pad_count <= 0:
            return (video, output_length, frame_count)

        tail = video[-1:].repeat(pad_count, *([1] * (video.ndim - 1)))
        return (torch.cat((video, tail), dim=0), output_length, frame_count)


def _window_start_and_stride(window, use_causal_anchor: bool) -> tuple[float, float]:
    indices = list(getattr(window, "index_list", []) or [])
    if not indices:
        return 0.0, 1.0

    start = indices[0]
    anchor_idx = getattr(window, "causal_anchor_index", None)
    if use_causal_anchor and anchor_idx is not None and anchor_idx >= 0:
        start = anchor_idx

    stride = 1
    if len(indices) > 1:
        deltas = [b - a for a, b in zip(indices, indices[1:])]
        if deltas and all(delta == deltas[0] for delta in deltas) and deltas[0] > 0:
            stride = deltas[0]

    return float(start), float(stride)


def _with_transformer_options(args, kwargs, transformer_options):
    if len(args) >= 6 and isinstance(args[5], dict):
        new_args = list(args)
        new_args[5] = transformer_options
        return tuple(new_args), kwargs
    new_kwargs = dict(kwargs)
    new_kwargs["transformer_options"] = transformer_options
    return args, new_kwargs


def _bernini_context_rope_wrapper(executor, *args, **kwargs):
    transformer_options = None
    if len(args) >= 6 and isinstance(args[5], dict):
        transformer_options = args[5]
    elif isinstance(kwargs.get("transformer_options"), dict):
        transformer_options = kwargs["transformer_options"]

    if transformer_options is None:
        return executor(*args, **kwargs)

    window = transformer_options.get("context_window")
    if window is None or not getattr(window, "index_list", None):
        return executor(*args, **kwargs)

    start, stride = _window_start_and_stride(
        window,
        bool(transformer_options.get("svdint4_bernini_context_use_causal_anchor", False)),
    )
    if start == 0.0 and stride == 1.0:
        return executor(*args, **kwargs)

    rope_options = dict(transformer_options.get("rope_options") or {})
    rope_options["shift_t"] = float(rope_options.get("shift_t", 0.0)) + start
    if stride != 1.0:
        rope_options["scale_t"] = float(rope_options.get("scale_t", 1.0)) * stride

    new_transformer_options = dict(transformer_options)
    new_transformer_options["rope_options"] = rope_options
    args, kwargs = _with_transformer_options(args, kwargs, new_transformer_options)
    return executor(*args, **kwargs)


class BerniniContextWindowsCore:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "context_length": (
                    "INT",
                    {
                        "default": 81,
                        "min": 1,
                        "max": 16385,
                        "step": 4,
                        "tooltip": "Context window length in real video frames. Wan latent length is derived as (frames - 1) // 4 + 1.",
                    },
                ),
                "context_overlap": (
                    "INT",
                    {
                        "default": 16,
                        "min": 0,
                        "max": 16384,
                        "step": 4,
                        "tooltip": "Window overlap in real video frames. The latent overlap is context_overlap // 4.",
                    },
                ),
                "context_schedule": (
                    [
                        comfy.context_windows.ContextSchedules.UNIFORM_STANDARD,
                        comfy.context_windows.ContextSchedules.STATIC_STANDARD,
                        comfy.context_windows.ContextSchedules.BATCHED,
                    ],
                    {"default": comfy.context_windows.ContextSchedules.UNIFORM_STANDARD},
                ),
                "fuse_method": (
                    comfy.context_windows.ContextFuseMethods.LIST_STATIC,
                    {"default": comfy.context_windows.ContextFuseMethods.PYRAMID},
                ),
                "freenoise": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "context_stride": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 32,
                        "tooltip": "Advanced: context stride for uniform schedules. RoPE time scale is adjusted for uniform strided windows.",
                    },
                ),
                "split_conds_to_windows": ("BOOLEAN", {"default": False}),
                "causal_window_fix": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Advanced: prepend the previous latent frame to non-zero windows. Disabled by default for Bernini reference alignment.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "SVDInt4/patches"
    TITLE = "Bernini Context Windows (Core MODEL)"

    def apply(
        self,
        model,
        context_length: int,
        context_overlap: int,
        context_schedule: str,
        fuse_method: str,
        freenoise: bool,
        context_stride: int = 1,
        split_conds_to_windows: bool = False,
        causal_window_fix: bool = False,
    ):
        latent_context_length = max(((context_length - 1) // 4) + 1, 1)
        latent_context_overlap = max(context_overlap // 4, 0)
        if latent_context_length <= 1:
            latent_context_overlap = 0
        else:
            latent_context_overlap = min(latent_context_overlap, latent_context_length - 1)

        patched = model.clone()
        patched.model_options["context_handler"] = comfy.context_windows.IndexListContextHandler(
            context_schedule=comfy.context_windows.get_matching_context_schedule(context_schedule),
            fuse_method=comfy.context_windows.get_matching_fuse_method(fuse_method),
            context_length=latent_context_length,
            context_overlap=latent_context_overlap,
            context_stride=max(int(context_stride), 1),
            closed_loop=False,
            dim=2,
            freenoise=freenoise,
            cond_retain_index_list=[],
            split_conds_to_windows=split_conds_to_windows,
            latent_retain_index_list=[],
            causal_window_fix=causal_window_fix,
        )
        patched.model_options.setdefault("transformer_options", {})[
            "svdint4_bernini_context_use_causal_anchor"
        ] = bool(causal_window_fix)

        patched.remove_wrappers_with_key(
            comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING,
            "ContextWindows_prepare_sampling",
        )
        patched.remove_wrappers_with_key(
            comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE,
            "ContextWindows_sampler_sample",
        )
        patched.remove_wrappers_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            _BERNINI_ROPE_WRAPPER_KEY,
        )

        comfy.context_windows.create_prepare_sampling_wrapper(patched)
        if freenoise:
            comfy.context_windows.create_sampler_sample_wrapper(patched)
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            _BERNINI_ROPE_WRAPPER_KEY,
            _bernini_context_rope_wrapper,
        )

        LOG.info(
            "Bernini core context windows enabled: %s real frames -> %s latent frames, "
            "%s real overlap -> %s latent overlap, schedule=%s, fuse=%s",
            context_length,
            latent_context_length,
            context_overlap,
            latent_context_overlap,
            context_schedule,
            fuse_method,
        )
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "SVDInt4DiffusionModelLoader": SVDInt4DiffusionModelLoader,
    "SVDInt4BerniniPadVideoLength": BerniniPadVideoLength,
    "SVDInt4BerniniContextWindowsCore": BerniniContextWindowsCore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SVDInt4DiffusionModelLoader": "Load SVDInt4 DiT",
    "SVDInt4BerniniPadVideoLength": "Bernini Pad Video Length",
    "SVDInt4BerniniContextWindowsCore": "Bernini Context Windows (Core MODEL)",
}
