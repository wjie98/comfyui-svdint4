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
_ANCHOR_MODE = "anchor_sparse"
_STANDARD_MODE = "standard"
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
        anchor_idx = getattr(window, "causal_anchor_index", None)
        if (
            bool(transformer_options.get("svdint4_bernini_context_use_causal_anchor", False))
            and anchor_idx is not None
            and anchor_idx >= 0
        ):
            indices = [anchor_idx] + indices
        new_transformer_options = dict(transformer_options)
        new_transformer_options[_ABSOLUTE_INDEX_KEY] = tuple(int(index) for index in indices)
        args, kwargs = _with_transformer_options(args, kwargs, new_transformer_options)
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


def _make_context_handler(**kwargs):
    params = inspect.signature(comfy.context_windows.IndexListContextHandler).parameters
    if kwargs.get("causal_window_fix", False) and "causal_window_fix" not in params:
        LOG.warning("Current ComfyUI does not support context window causal_window_fix; ignoring it.")
    if kwargs.get("split_conds_to_windows", False) and "split_conds_to_windows" not in params:
        LOG.warning("Current ComfyUI does not support split_conds_to_windows; ignoring it.")
    supported_kwargs = {k: v for k, v in kwargs.items() if k in params}
    return comfy.context_windows.IndexListContextHandler(**supported_kwargs)


class BerniniAnchorContextHandler(comfy.context_windows.IndexListContextHandler):
    def __init__(self, *, center_latents: int, halo_latents: int, anchor_count: int, **kwargs):
        if center_latents <= 0:
            raise ValueError("anchor_sparse mode requires context_length/center_latents > 0.")
        if halo_latents < 0:
            raise ValueError("anchor_sparse mode requires context_overlap/halo_latents >= 0.")
        if anchor_count < 0:
            raise ValueError("anchor_sparse mode requires anchor_count >= 0.")
        super().__init__(
            context_length=center_latents,
            context_overlap=halo_latents,
            **kwargs,
        )
        self.center_latents = int(center_latents)
        self.halo_latents = int(halo_latents)
        self.anchor_count = int(anchor_count)
        self._anchor_indices: tuple[int, ...] = ()

    def execute(self, calc_cond_batch, model, conds, x_in, timestep, model_options):
        self._anchor_indices = _select_anchor_indices_from_conds(conds, x_in, self.dim, self.anchor_count)
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
            budget = self.anchor_count
            candidates = tuple(index for index in self._anchor_indices if index not in local_set)
            chosen = _choose_anchor_subset(candidates, center_start, center_end, budget, full_length)
            model_indices = tuple(sorted(local + chosen))
            window = comfy.context_windows.IndexListContextWindow(
                list(model_indices),
                dim=self.dim,
                total_frames=full_length,
                context_overlap=self.context_overlap,
            )
            window.svdint4_use_absolute_indices = True
            window.svdint4_center_latent_indices = tuple(range(center_start, center_end))
            window.svdint4_center_model_positions = tuple(model_indices.index(index) for index in window.svdint4_center_latent_indices)
            windows.append(window)
        return windows

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
        center_indices = getattr(window, "svdint4_center_latent_indices", None)
        center_positions = getattr(window, "svdint4_center_model_positions", None)
        if center_indices is None or center_positions is None:
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

        for output, final, count in zip(sub_conds_out, conds_final, counts_final):
            for pos, index in zip(center_positions, center_indices):
                dst = tuple([slice(None)] * self.dim + [index])
                src = tuple([slice(None)] * self.dim + [pos])
                final[dst] += output[src]
                count[dst] += 1.0


def _center_ranges(frame_count: int, center_size: int) -> list[tuple[int, int]]:
    ranges = []
    start = 0
    while start < frame_count:
        end = min(start + center_size, frame_count)
        ranges.append((start, end))
        start = end
    return ranges or [(0, frame_count)]


def _select_anchor_indices_from_conds(conds, x_in: torch.Tensor, dim: int, anchor_count: int) -> tuple[int, ...]:
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
            "Connect Bernini Conditioning to this node's optional condition input and to the sampler."
        )
    return _select_frame_anchors(source, reference, anchor_count)


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


def _select_frame_anchors(source_latents: torch.Tensor, reference_latents: torch.Tensor | None, anchor_count: int) -> tuple[int, ...]:
    source = _as_tchw(source_latents)
    frame_count = int(source.shape[0])
    if frame_count == 0 or anchor_count <= 0:
        return ()
    target_count = min(int(anchor_count), frame_count)
    if target_count <= 0:
        return ()
    reference = _as_tchw(reference_latents) if reference_latents is not None else None
    scores = _score_anchor_frames(source, reference)
    return tuple(_select_with_gap(scores, target_count, min_gap=8, force_edges=True))


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


def _select_with_gap(scores: torch.Tensor, target_count: int, min_gap: int, force_edges: bool) -> list[int]:
    frame_count = int(scores.numel())
    selected: list[int] = []
    if force_edges and frame_count > 0:
        selected.append(0)
        if frame_count > 1 and len(selected) < target_count:
            selected.append(frame_count - 1)
    for index in torch.argsort(scores, descending=True).tolist():
        if len(selected) >= target_count:
            break
        if index in selected:
            continue
        if all(abs(index - other) >= min_gap for other in selected):
            selected.append(int(index))
    for index in torch.argsort(scores, descending=True).tolist():
        if len(selected) >= target_count:
            break
        if index not in selected:
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


def _validate_anchor_condition(condition) -> None:
    if condition is None:
        raise ValueError("anchor_sparse mode requires the optional condition input from Bernini Conditioning.")
    for entry in condition:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2 or not isinstance(entry[1], dict):
            continue
        context = entry[1].get("context_latents")
        if isinstance(context, list) and context:
            return
    raise ValueError("anchor_sparse mode requires condition input containing Bernini context_latents.")


class BerniniContextWindowsCore:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "context_length": (
                    "INT",
                    {
                        "default": 17,
                        "min": 1,
                        "max": 16385,
                        "step": 1,
                        "tooltip": (
                            "standard mode: context window length in real video frames. "
                            "anchor_sparse mode: center latent frames to write back."
                        ),
                    },
                ),
                "context_overlap": (
                    "INT",
                    {
                        "default": 2,
                        "min": 0,
                        "max": 16384,
                        "step": 1,
                        "tooltip": (
                            "standard mode: overlap in real video frames. "
                            "anchor_sparse mode: local halo latent frames on each side."
                        ),
                    },
                ),
                "sampling_mode": (
                    [_STANDARD_MODE, _ANCHOR_MODE],
                    {
                        "default": _ANCHOR_MODE,
                        "tooltip": "standard uses ComfyUI context schedules; anchor_sparse uses adaptive global anchors with absolute RoPE and center-only scatter.",
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
                "anchor_count": (
                    "INT",
                    {
                        "default": 6,
                        "min": 0,
                        "max": 128,
                        "tooltip": "anchor_sparse only: maximum global anchor latent frames added to each window.",
                    },
                ),
                "context_stride": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 32,
                        "tooltip": "Advanced: context stride for uniform schedules. RoPE time scale is adjusted for uniform strided windows.",
                    },
                ),
                "condition": ("CONDITIONING",),
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
    TITLE = "Bernini Context Windows"

    def apply(
        self,
        model,
        context_length: int,
        context_overlap: int,
        sampling_mode: str,
        context_schedule: str,
        fuse_method: str,
        freenoise: bool,
        anchor_count: int = 6,
        context_stride: int = 1,
        condition=None,
        causal_window_fix: bool = False,
    ):
        if sampling_mode == _ANCHOR_MODE:
            _validate_anchor_condition(condition)
            latent_context_length = max(int(context_length), 1)
            latent_context_overlap = max(int(context_overlap), 0)
            context_handler = BerniniAnchorContextHandler(
                context_schedule=comfy.context_windows.get_matching_context_schedule(comfy.context_windows.ContextSchedules.STATIC_STANDARD),
                fuse_method=comfy.context_windows.get_matching_fuse_method(comfy.context_windows.ContextFuseMethods.FLAT),
                center_latents=latent_context_length,
                halo_latents=latent_context_overlap,
                anchor_count=max(int(anchor_count), 0),
                context_stride=1,
                closed_loop=False,
                dim=2,
                freenoise=freenoise,
                cond_retain_index_list=[],
                latent_retain_index_list=[],
                causal_window_fix=causal_window_fix,
            )
        else:
            latent_context_length = max(((context_length - 1) // 4) + 1, 1)
            latent_context_overlap = max(context_overlap // 4, 0)
            if latent_context_length <= 1:
                latent_context_overlap = 0
            else:
                latent_context_overlap = min(latent_context_overlap, latent_context_length - 1)
            context_handler = _make_context_handler(
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
                causal_window_fix=causal_window_fix,
            )

        patched = model.clone()
        patched.model_options["context_handler"] = context_handler
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
        _install_bernini_absolute_rope_patch()
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            _BERNINI_ROPE_WRAPPER_KEY,
            _bernini_context_rope_wrapper,
        )

        LOG.info(
            "Bernini context windows enabled: mode=%s, length=%s -> %s latent frames, "
            "overlap/halo=%s -> %s latent frames, anchors=%s, schedule=%s, fuse=%s",
            sampling_mode,
            context_length,
            latent_context_length,
            context_overlap,
            latent_context_overlap,
            anchor_count if sampling_mode == _ANCHOR_MODE else 0,
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
    "SVDInt4BerniniContextWindowsCore": "Bernini Context Windows",
}
