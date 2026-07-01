from __future__ import annotations

import logging
import os
import inspect
import math
from pathlib import Path

import comfy.context_windows
import comfy.patcher_extension
import folder_paths
import torch
import torch.nn.functional as F
from safetensors import safe_open


LOG = logging.getLogger("comfyui-svdint4")
FOLDER_NAME = "diffusion_models"
MODEL_EXTENSIONS = {".safetensors", ".sft"}
ENV_PATHS = ("SVDINT4_DIT_PATHS",)
SUPPORTED_FORMATS = {"svdint4-dit-single-v2"}
_BERNINI_ROPE_WRAPPER_KEY = "svdint4_bernini_context_rope"
_ANCHOR_SCHEDULE = "anchor_sparse"
_ABSOLUTE_INDEX_KEY = "svdint4_bernini_absolute_latent_indices"


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


def _validate_context_window_frames(context_length: int, context_overlap: int) -> tuple[int, int]:
    context_length = int(context_length)
    context_overlap = int(context_overlap)
    if context_length < 5 or not _is_wan_frame_count(context_length):
        raise ValueError(f"context_length must be 4*n+1 real frames with n>=1; got {context_length}.")
    if context_overlap <= 0 or context_overlap % 4 != 0:
        raise ValueError(f"context_overlap must be a positive 4*n real-frame count; got {context_overlap}.")
    if context_overlap >= context_length:
        raise ValueError(
            f"context_overlap must be shorter than context_length; got overlap={context_overlap}, "
            f"length={context_length}."
        )
    return ((context_length - 1) // 4) + 1, context_overlap // 4


def _validate_anchor_length_frames(anchor_length: int) -> int:
    anchor_length = int(anchor_length)
    if anchor_length < 0 or anchor_length % 4 != 0:
        raise ValueError(f"anchor_length must be 0 or a positive 4*n real-frame count; got {anchor_length}.")
    return anchor_length // 4


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
        target_frame_count = (
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
        )
        return {
            "required": {
                "image": ("IMAGE",),
                "target_frame_count": target_frame_count,
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height", "length", "target_length")
    FUNCTION = "pad"
    CATEGORY = "SVDInt4/video"
    TITLE = "Bernini Pad Video Length"

    def pad(self, image, target_frame_count: int):
        if image.ndim < 3:
            raise ValueError(f"Expected IMAGE tensor shaped [frames, height, width, channels], got {tuple(image.shape)}.")
        frame_count = int(image.shape[0])
        if frame_count < 1:
            raise ValueError("Bernini Pad Video Length requires at least one input frame.")
        height = int(image.shape[1])
        width = int(image.shape[2])

        if target_frame_count == 0:
            output_length = _ceil_wan_frame_count(frame_count)
        else:
            output_length = int(target_frame_count)
            if not _is_wan_frame_count(output_length):
                raise ValueError(
                    f"target_frame_count must be 0 or 4*n+1; got {target_frame_count}."
                )
            if output_length < frame_count:
                raise ValueError(
                    f"target_frame_count={output_length} is shorter than the input video "
                    f"({frame_count} frames). This node only pads; trim upstream if needed."
                )

        pad_count = output_length - frame_count
        if pad_count <= 0:
            return (image, width, height, frame_count, output_length)

        tail = image[-1:].repeat(pad_count, *([1] * (image.ndim - 1)))
        return (torch.cat((image, tail), dim=0), width, height, frame_count, output_length)


def _window_start_and_stride(window) -> tuple[float, float]:
    indices = list(getattr(window, "index_list", []) or [])
    if not indices:
        return 0.0, 1.0

    start = indices[0]
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


def _install_bernini_absolute_rope_patch() -> None:
    from comfy.ldm.wan import model as wan_model

    if hasattr(wan_model.WanModel, "_svdint4_original_forward"):
        return

    wan_model.WanModel._svdint4_original_forward = wan_model.WanModel._forward
    wan_model.WanModel._forward = _wan_forward_with_optional_absolute_indices


def _wan_forward_with_optional_absolute_indices(
    self,
    x,
    timestep,
    context,
    clip_fea=None,
    time_dim_concat=None,
    transformer_options={},
    **kwargs,
):
    import comfy.ldm.common_dit

    target_indices = transformer_options.get(_ABSOLUTE_INDEX_KEY, None)
    if target_indices is None:
        return self._svdint4_original_forward(
            x,
            timestep,
            context,
            clip_fea=clip_fea,
            time_dim_concat=time_dim_concat,
            transformer_options=transformer_options,
            **kwargs,
        )

    if time_dim_concat is not None or (self.ref_conv is not None and "reference_latent" in kwargs):
        LOG.warning(
            "Bernini absolute context RoPE indices were ignored for a Wan path with "
            "time_dim_concat/reference_latent; falling back to ComfyUI RoPE."
        )
        return self._svdint4_original_forward(
            x,
            timestep,
            context,
            clip_fea=clip_fea,
            time_dim_concat=time_dim_concat,
            transformer_options=transformer_options,
            **kwargs,
        )

    bs, c, t, h, w = x.shape
    del bs, c
    x = comfy.ldm.common_dit.pad_to_patch_size(x, self.patch_size)
    freqs = _rope_encode_with_absolute_indices(
        self,
        target_indices,
        t,
        h,
        w,
        device=x.device,
        dtype=x.dtype,
        transformer_options=transformer_options,
        source_id=0,
    )

    context_latents = kwargs.get("context_latents", None)
    if context_latents is not None:
        context_latents = [comfy.ldm.common_dit.pad_to_patch_size(lat, self.patch_size) for lat in context_latents]
        for i, lat in enumerate(context_latents):
            context_indices = target_indices if lat.shape[-3] == len(target_indices) else None
            freqs = torch.cat(
                [
                    freqs,
                    _rope_encode_with_absolute_indices(
                        self,
                        context_indices,
                        lat.shape[-3],
                        lat.shape[-2],
                        lat.shape[-1],
                        device=x.device,
                        dtype=x.dtype,
                        transformer_options=transformer_options,
                        source_id=i + 1,
                    ),
                ],
                dim=1,
            )
        kwargs = {**kwargs, "context_latents": context_latents}

    return self.forward_orig(
        x,
        timestep,
        context,
        clip_fea=clip_fea,
        freqs=freqs,
        transformer_options=transformer_options,
        **kwargs,
    )[:, :, :t, :h, :w]


def _rope_encode_with_absolute_indices(
    model,
    indices,
    t,
    h,
    w,
    *,
    device,
    dtype,
    transformer_options,
    source_id: int,
):
    if indices is None:
        return model.rope_encode(
            t,
            h,
            w,
            device=device,
            dtype=dtype,
            transformer_options=transformer_options,
            source_id=source_id,
        )

    patch_size = model.patch_size
    steps_t = ((t + (patch_size[0] // 2)) // patch_size[0])
    steps_h = ((h + (patch_size[1] // 2)) // patch_size[1])
    steps_w = ((w + (patch_size[2] // 2)) // patch_size[2])
    temporal = _normalize_temporal_indices(indices, steps_t).to(device=device, dtype=dtype)

    h_len = steps_h
    w_len = steps_w
    h_start = 0.0
    w_start = 0.0
    rope_options = transformer_options.get("rope_options", None)
    if rope_options is not None:
        temporal = temporal * float(rope_options.get("scale_t", 1.0)) + float(rope_options.get("shift_t", 0.0))
        h_len = (h_len - 1.0) * float(rope_options.get("scale_y", 1.0)) + 1.0
        w_len = (w_len - 1.0) * float(rope_options.get("scale_x", 1.0)) + 1.0
        h_start += float(rope_options.get("shift_y", 0.0))
        w_start += float(rope_options.get("shift_x", 0.0))

    img_ids = torch.zeros((steps_t, steps_h, steps_w, 3), device=device, dtype=dtype)
    img_ids[:, :, :, 0] = temporal.reshape(-1, 1, 1)
    img_ids[:, :, :, 1] = torch.linspace(h_start, h_len - 1 + h_start, steps_h, device=device, dtype=dtype).reshape(1, -1, 1)
    img_ids[:, :, :, 2] = torch.linspace(w_start, w_len - 1 + w_start, steps_w, device=device, dtype=dtype).reshape(1, 1, -1)
    img_ids = img_ids.reshape(steps_t * steps_h * steps_w, 3)
    return model.rope_embedder(img_ids).reshape(1, -1, model.dim // model.num_heads // 2, 2)


def _normalize_temporal_indices(indices, steps_t: int) -> torch.Tensor:
    if isinstance(indices, torch.Tensor):
        values = indices.detach().flatten().to(dtype=torch.float32, device="cpu")
    else:
        values = torch.tensor([float(v) for v in indices], dtype=torch.float32)
    if values.numel() == 0:
        values = torch.arange(steps_t, dtype=torch.float32)
    if values.numel() < steps_t:
        last = values[-1]
        pad = torch.arange(1, steps_t - values.numel() + 1, dtype=torch.float32) + last
        values = torch.cat([values, pad], dim=0)
    return values[:steps_t]


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

    if getattr(window, "svdint4_use_absolute_indices", False):
        indices = list(window.index_list)
        new_transformer_options = dict(transformer_options)
        new_transformer_options[_ABSOLUTE_INDEX_KEY] = tuple(int(index) for index in indices)
        args, kwargs = _with_transformer_options(args, kwargs, new_transformer_options)
        return executor(*args, **kwargs)

    start, stride = _window_start_and_stride(window)
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


def _filter_supported_kwargs(callable_obj, kwargs: dict) -> dict:
    try:
        params = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in params}


def _filter_context_handler_kwargs(kwargs: dict) -> dict:
    filtered = _filter_supported_kwargs(comfy.context_windows.IndexListContextHandler, kwargs)
    if len(filtered) == len(kwargs):
        return filtered

    params = set(filtered)
    for key, value in kwargs.items():
        if key in params:
            continue
        if value is None or value is False:
            continue
        if isinstance(value, (str, list, tuple, dict, set)) and not value:
            continue
        LOG.warning(
            "Current ComfyUI IndexListContextHandler does not support %s; ignoring value %r.",
            key,
            value,
        )
    return filtered


def _make_index_list_context_window(index_list: tuple[int, ...], *, dim: int, total_frames: int, context_overlap: int):
    if not index_list:
        raise ValueError("IndexListContextWindow requires at least one latent index.")
    dim = int(dim)
    total_frames = int(total_frames)
    context_overlap = int(context_overlap)
    kwargs = {
        "dim": dim,
        "total_frames": total_frames,
        "context_overlap": context_overlap,
    }
    window_cls = comfy.context_windows.IndexListContextWindow
    supported_kwargs = _filter_supported_kwargs(window_cls, kwargs)
    try:
        window = window_cls(list(index_list), **supported_kwargs)
    except TypeError as exc:
        if not supported_kwargs or "unexpected keyword" not in str(exc):
            raise
        window = window_cls(list(index_list))
    window.dim = dim
    window.total_frames = total_frames
    window.context_length = len(index_list)
    window.context_overlap = context_overlap
    if total_frames > 0:
        window.center_ratio = (min(index_list) + max(index_list)) / (2 * total_frames)
    else:
        window.center_ratio = 0.0
    return window


class BerniniContextHandlerBase(comfy.context_windows.IndexListContextHandler):
    def __init__(self, *, first_frame_sink: bool, **kwargs):
        kwargs["causal_window_fix"] = False
        super().__init__(**_filter_context_handler_kwargs(kwargs))
        self.first_frame_sink = bool(first_frame_sink)

    def _build_context_window(
        self,
        *,
        target_indices: tuple[int, ...],
        context_indices: tuple[int, ...],
        full_length: int,
        context_overlap: int,
    ):
        target_indices = tuple(int(index) for index in target_indices)
        if not target_indices:
            raise ValueError("Bernini context windows require at least one write-back latent index.")

        model_indices = set(int(index) for index in context_indices)
        model_indices.update(target_indices)
        if self.first_frame_sink and full_length > 0 and 0 not in target_indices:
            model_indices.add(0)
        model_indices = tuple(sorted(model_indices))

        window = _make_index_list_context_window(
            model_indices,
            dim=self.dim,
            total_frames=full_length,
            context_overlap=context_overlap,
        )
        window.svdint4_use_absolute_indices = True
        window.svdint4_write_latent_indices = target_indices
        window.svdint4_write_model_positions = tuple(model_indices.index(index) for index in target_indices)
        window.svdint4_write_context_overlap = int(context_overlap)
        return window

    def combine_context_window_results(
        self,
        x_in: torch.Tensor,
        sub_conds_out,
        sub_conds,
        window,
        window_idx: int,
        total_windows: int,
        timestep: torch.Tensor,
        conds_final: list[torch.Tensor],
        counts_final: list[torch.Tensor],
        biases_final: list[torch.Tensor],
    ):
        write_indices = getattr(window, "svdint4_write_latent_indices", None)
        write_positions = getattr(window, "svdint4_write_model_positions", None)
        if write_indices is None or write_positions is None:
            return super().combine_context_window_results(
                x_in,
                sub_conds_out,
                sub_conds,
                window,
                window_idx,
                total_windows,
                timestep,
                conds_final,
                counts_final,
                biases_final,
            )

        if self.fuse_method.name == comfy.context_windows.ContextFuseMethods.RELATIVE:
            first = write_indices[0]
            last = write_indices[-1]
            center = (first + last) / 2
            width = (last - first + 1e-2) / 2
            for pos, index in zip(write_positions, write_indices):
                bias = 1 - abs(index - center) / width
                bias = max(1e-2, bias)
                for i in range(len(sub_conds_out)):
                    bias_total = biases_final[i][index]
                    prev_weight = bias_total / (bias_total + bias)
                    new_weight = bias / (bias_total + bias)
                    dst = tuple([slice(None)] * self.dim + [index])
                    src = tuple([slice(None)] * self.dim + [pos])
                    conds_final[i][dst] = conds_final[i][dst] * prev_weight + sub_conds_out[i][src] * new_weight
                    biases_final[i][index] = bias_total + bias
            return

        weights = comfy.context_windows.get_context_weights(
            len(write_indices),
            x_in.shape[self.dim],
            list(write_indices),
            self,
            sigma=timestep,
            context_overlap=getattr(window, "svdint4_write_context_overlap", self.context_overlap),
        )
        weights_tensor = comfy.context_windows.match_weights_to_dim(weights, x_in, self.dim, device=x_in.device)
        for output, final, count in zip(sub_conds_out, conds_final, counts_final):
            for weight_pos, (pos, index) in enumerate(zip(write_positions, write_indices)):
                dst = tuple([slice(None)] * self.dim + [index])
                src = tuple([slice(None)] * self.dim + [pos])
                weight_src = tuple([slice(None)] * self.dim + [weight_pos])
                final[dst] += output[src] * weights_tensor[weight_src]
                count[dst] += weights_tensor[weight_src]


class BerniniScheduledContextHandler(BerniniContextHandlerBase):
    def get_context_windows(self, model, x_in: torch.Tensor, model_options: dict[str]):
        full_length = x_in.size(self.dim)
        windows = []
        for window in super().get_context_windows(model, x_in, model_options):
            indices = tuple(int(index) for index in window.index_list)
            context_overlap = getattr(window, "context_overlap", self.context_overlap)
            windows.append(
                self._build_context_window(
                    target_indices=indices,
                    context_indices=indices,
                    full_length=full_length,
                    context_overlap=context_overlap,
                )
            )
        return windows


class BerniniAnchorContextHandler(BerniniContextHandlerBase):
    def __init__(
        self,
        *,
        center_latents: int,
        halo_latents: int,
        anchor_latents: int,
        first_frame_sink: bool,
        first_frame_anchor_included: bool,
        **kwargs,
    ):
        if center_latents <= 0:
            raise ValueError("anchor_sparse mode requires context_length/center_latents > 0.")
        if halo_latents < 0:
            raise ValueError("anchor_sparse mode requires context_overlap/halo_latents >= 0.")
        if anchor_latents < 0:
            raise ValueError("anchor_sparse mode requires anchor_latents >= 0.")
        super().__init__(
            context_length=center_latents,
            context_overlap=halo_latents,
            first_frame_sink=first_frame_sink,
            **kwargs,
        )
        self.center_latents = int(center_latents)
        self.halo_latents = int(halo_latents)
        self.anchor_latents = int(anchor_latents)
        self.first_frame_anchor_included = bool(first_frame_anchor_included)
        self._anchor_indices: tuple[int, ...] = ()

    def execute(self, calc_cond_batch, model, conds, x_in, timestep, model_options):
        self._anchor_indices = _select_anchor_indices_from_conds(
            conds,
            x_in,
            self.dim,
            self.anchor_latents,
            include_first_frame=self.first_frame_anchor_included,
        )
        try:
            return super().execute(calc_cond_batch, model, conds, x_in, timestep, model_options)
        finally:
            self._anchor_indices = ()

    def get_context_windows(self, model, x_in: torch.Tensor, model_options: dict[str]):
        full_length = x_in.size(self.dim)
        windows = []
        for center_start, center_end in _center_ranges(full_length, self.center_latents):
            local_start = max(0, center_start - self.halo_latents)
            local_end = min(full_length, center_end + self.halo_latents)
            local = tuple(range(local_start, local_end))
            local_set = set(local)
            budget = self.anchor_latents
            candidates = tuple(index for index in self._anchor_indices if index not in local_set)
            chosen = _choose_anchor_subset(candidates, center_start, center_end, budget, full_length)
            windows.append(
                self._build_context_window(
                    target_indices=tuple(range(center_start, center_end)),
                    context_indices=tuple(sorted(local + chosen)),
                    full_length=full_length,
                    context_overlap=self.context_overlap,
                )
            )
        return windows


def _center_ranges(frame_count: int, center_size: int) -> list[tuple[int, int]]:
    ranges = []
    start = 0
    while start < frame_count:
        end = min(start + center_size, frame_count)
        ranges.append((start, end))
        start = end
    return ranges or [(0, frame_count)]


def _select_anchor_indices_from_conds(
    conds,
    x_in: torch.Tensor,
    dim: int,
    anchor_latents: int,
    *,
    include_first_frame: bool,
) -> tuple[int, ...]:
    context_latents = _extract_context_latents_from_conds(conds)
    source = None
    reference = None
    for latent in context_latents:
        if not isinstance(latent, torch.Tensor) or latent.ndim <= dim:
            continue
        if latent.shape[dim] != x_in.shape[dim]:
            continue
        if source is None:
            source = latent
        elif reference is None:
            reference = latent
            break
    if source is None:
        raise ValueError(
            "anchor_sparse mode requires BerniniConditioning context_latents matching the target latent length. "
            "Connect Bernini Conditioning to the sampler so the model conds include context_latents."
        )
    return _select_frame_anchors(source, reference, anchor_latents, include_first_frame=include_first_frame)


def _extract_context_latents_from_conds(conds) -> list[torch.Tensor]:
    for cond_group in conds:
        if cond_group is None:
            continue
        for cond in cond_group:
            model_conds = cond.get("model_conds", {}) if isinstance(cond, dict) else {}
            context_cond = model_conds.get("context_latents")
            values = getattr(context_cond, "cond", None)
            if isinstance(values, list) and values:
                return values
    return []


def _select_frame_anchors(
    source_latents: torch.Tensor,
    reference_latents: torch.Tensor | None,
    anchor_latents: int,
    *,
    include_first_frame: bool,
) -> tuple[int, ...]:
    source = _as_tchw(source_latents)
    frame_count = int(source.shape[0])
    if frame_count == 0 or anchor_latents <= 0:
        return ()
    target_count = min(int(anchor_latents), frame_count)
    if target_count <= 0:
        return ()
    reference = _as_tchw(reference_latents) if reference_latents is not None else None
    scores = _score_anchor_frames(source, reference)
    initial = (0,) if include_first_frame else ()
    excluded = () if include_first_frame else (0,)
    if frame_count > 1 and target_count > 1:
        initial = initial + (frame_count - 1,)
    return tuple(_select_with_gap(scores, target_count, min_gap=8, initial_indices=initial, excluded_indices=excluded))


def _as_tchw(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim == 4:
        return latents.detach().float().cpu()
    if latents.ndim == 5 and latents.shape[0] == 1:
        return latents[0].permute(1, 0, 2, 3).detach().float().cpu()
    raise ValueError("anchor_sparse mode expects latents shaped [T,C,H,W] or [1,C,T,H,W].")


def _score_anchor_frames(source: torch.Tensor, reference: torch.Tensor | None) -> torch.Tensor:
    temporal = _temporal_scores(source)
    spatial = _spatial_scores(source)
    score = _robust_z(temporal) + 0.25 * _robust_z(spatial)
    if reference is not None:
        score = score + 0.5 * _robust_z(_reference_scores(source, reference))
    return torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)


def _temporal_scores(source: torch.Tensor) -> torch.Tensor:
    if source.shape[0] == 1:
        return torch.zeros(1, dtype=torch.float32)
    flattened = source.flatten(1)
    normalized = F.normalize(flattened, dim=1, eps=1e-6)
    distance = 1.0 - (normalized[1:] * normalized[:-1]).sum(dim=1)
    return torch.cat([distance[:1], distance], dim=0)


def _spatial_scores(source: torch.Tensor) -> torch.Tensor:
    tokens = source.permute(0, 2, 3, 1).flatten(1, 2)
    mean = tokens.mean(dim=1, keepdim=True)
    token_distance = 1.0 - (F.normalize(tokens, dim=2, eps=1e-6) * F.normalize(mean, dim=2, eps=1e-6)).sum(dim=2)
    return torch.quantile(token_distance, 0.95, dim=1)


def _reference_scores(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    source_flat = F.normalize(source.flatten(1), dim=1, eps=1e-6)
    reference_flat = F.normalize(reference.flatten(1), dim=1, eps=1e-6)
    return (source_flat @ reference_flat.T).max(dim=1).values


def _robust_z(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values
    median = values.median()
    mad = (values - median).abs().median()
    if float(mad.item()) < 1e-6:
        std = values.std(unbiased=False)
        if float(std.item()) < 1e-6:
            return torch.zeros_like(values)
        return (values - values.mean()) / (std + 1e-6)
    return (values - median) / (mad + 1e-6)


def _select_with_gap(
    scores: torch.Tensor,
    target_count: int,
    min_gap: int,
    initial_indices: tuple[int, ...] = (),
    excluded_indices: tuple[int, ...] = (),
) -> list[int]:
    frame_count = int(scores.numel())
    excluded = set(int(index) for index in excluded_indices)
    selected: list[int] = []
    for index in initial_indices:
        if 0 <= index < frame_count and index not in excluded and index not in selected:
            selected.append(int(index))
        if len(selected) >= target_count:
            break
    for index in torch.argsort(scores, descending=True).tolist():
        if len(selected) >= target_count:
            break
        if index in excluded or index in selected:
            continue
        if all(abs(index - other) >= min_gap for other in selected):
            selected.append(int(index))
    for index in torch.argsort(scores, descending=True).tolist():
        if len(selected) >= target_count:
            break
        if index not in excluded and index not in selected:
            selected.append(int(index))
    return sorted(selected)


def _choose_anchor_subset(
    candidates: tuple[int, ...],
    center_start: int,
    center_end: int,
    budget: int,
    frame_count: int,
) -> tuple[int, ...]:
    if budget <= 0 or not candidates:
        return ()
    center = (center_start + center_end - 1) / 2.0
    ranked = []
    for index in candidates:
        is_edge = index == 0 or index == frame_count - 1
        ranked.append((0 if is_edge else 1, abs(index - center), index))
    return tuple(sorted(item[2] for item in sorted(ranked)[:budget]))


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
                        "min": 5,
                        "max": 16385,
                        "step": 4,
                        "tooltip": (
                            "Context window length in real video frames. Must be 4*n+1; "
                            "81 frames maps to 21 Wan latent frames."
                        ),
                    },
                ),
                "context_overlap": (
                    "INT",
                    {
                        "default": 16,
                        "min": 4,
                        "max": 16384,
                        "step": 4,
                        "tooltip": (
                            "Context overlap or local halo in real video frames. Must be positive 4*n; "
                            "16 frames maps to 4 Wan latent frames."
                        ),
                    },
                ),
                "context_schedule": (
                    [
                        _ANCHOR_SCHEDULE,
                        comfy.context_windows.ContextSchedules.UNIFORM_STANDARD,
                        comfy.context_windows.ContextSchedules.STATIC_STANDARD,
                        comfy.context_windows.ContextSchedules.BATCHED,
                    ],
                    {
                        "default": _ANCHOR_SCHEDULE,
                        "tooltip": (
                            "anchor_sparse uses adaptive global anchor latents with center-only scatter; "
                            "other schedules use ComfyUI's standard context windows."
                        ),
                    },
                ),
                "fuse_method": (
                    comfy.context_windows.ContextFuseMethods.LIST_STATIC,
                    {"default": comfy.context_windows.ContextFuseMethods.PYRAMID},
                ),
                "freenoise": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "anchor_length": (
                    "INT",
                    {
                        "default": 16,
                        "min": 0,
                        "max": 4096,
                        "step": 4,
                        "tooltip": (
                            "anchor_sparse only: adaptive global anchor budget in real video frames. "
                            "Must be 0 or 4*n; 16 frames maps to 4 anchor latents. Excludes first_frame_sink."
                        ),
                    },
                ),
                "context_stride": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 32,
                        "tooltip": "Standard schedules only: context stride for uniform schedules. RoPE time scale is adjusted for uniform strided windows.",
                    },
                ),
                "first_frame_sink": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Use latent frame 0 as a context-only sink for windows that do not already include it. "
                            "The first window still writes frame 0 normally."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "SVDInt4/patches"
    TITLE = "Bernini Context Windows"

    def apply(
        self,
        model,
        context_length: int,
        context_overlap: int,
        context_schedule: str,
        fuse_method: str,
        freenoise: bool,
        anchor_length: int = 16,
        context_stride: int = 1,
        first_frame_sink: bool = True,
    ):
        latent_context_length, latent_context_overlap = _validate_context_window_frames(
            context_length,
            context_overlap,
        )
        anchor_latents = _validate_anchor_length_frames(anchor_length)
        if context_schedule == _ANCHOR_SCHEDULE:
            context_handler = BerniniAnchorContextHandler(
                context_schedule=comfy.context_windows.get_matching_context_schedule(comfy.context_windows.ContextSchedules.STATIC_STANDARD),
                fuse_method=comfy.context_windows.get_matching_fuse_method(comfy.context_windows.ContextFuseMethods.FLAT),
                center_latents=latent_context_length,
                halo_latents=latent_context_overlap,
                anchor_latents=anchor_latents,
                first_frame_sink=first_frame_sink,
                first_frame_anchor_included=not first_frame_sink,
                context_stride=1,
                closed_loop=False,
                dim=2,
                freenoise=freenoise,
                cond_retain_index_list=[],
                latent_retain_index_list=[],
            )
        else:
            context_handler = BerniniScheduledContextHandler(
                context_schedule=comfy.context_windows.get_matching_context_schedule(context_schedule),
                fuse_method=comfy.context_windows.get_matching_fuse_method(fuse_method),
                context_length=latent_context_length,
                context_overlap=latent_context_overlap,
                context_stride=max(int(context_stride), 1),
                closed_loop=False,
                dim=2,
                freenoise=freenoise,
                cond_retain_index_list=[],
                latent_retain_index_list=[],
                first_frame_sink=first_frame_sink,
            )

        patched = model.clone()
        patched.model_options["context_handler"] = context_handler
        patched.model_options.setdefault("transformer_options", {})
        effective_fuse_method = (
            comfy.context_windows.ContextFuseMethods.FLAT
            if context_schedule == _ANCHOR_SCHEDULE
            else fuse_method
        )

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
        _install_bernini_absolute_rope_patch()
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            _BERNINI_ROPE_WRAPPER_KEY,
            _bernini_context_rope_wrapper,
        )

        LOG.info(
            "Bernini context windows enabled: schedule=%s, length=%s -> %s latent frames, "
            "overlap/halo=%s -> %s latent frames, anchor_length=%s -> %s anchor latents, "
            "first_frame_sink=%s, fuse=%s",
            context_schedule,
            context_length,
            latent_context_length,
            context_overlap,
            latent_context_overlap,
            anchor_length if context_schedule == _ANCHOR_SCHEDULE else 0,
            anchor_latents if context_schedule == _ANCHOR_SCHEDULE else 0,
            first_frame_sink,
            effective_fuse_method,
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
    "SVDInt4BerniniContextWindowsCore": "Bernini Context Windows",
}
