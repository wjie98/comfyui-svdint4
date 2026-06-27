from __future__ import annotations

import os
from pathlib import Path

import folder_paths
from safetensors import safe_open


FOLDER_NAME = "diffusion_models"
MODEL_EXTENSIONS = {".safetensors", ".sft"}
ENV_PATHS = ("SVDINT4_DIT_PATHS",)
SUPPORTED_FORMATS = {"svdint4-dit-single-v2"}


def _model_dirs() -> list[str]:
    return folder_paths.get_folder_paths(FOLDER_NAME)


def _register_extra_model_dirs() -> None:
    changed = False
    for env_name in ENV_PATHS:
        for item in os.environ.get(env_name, "").split(os.pathsep):
            if not item:
                continue
            before = _model_dirs()
            folder_paths.add_model_folder_path(FOLDER_NAME, item)
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
                "external_lora_bypass": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Run standard adapter LoRAs as fp16 forward bypass paths instead of ignoring packed Linear LoRA patches.",
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
        external_lora_bypass: bool = False,
    ):
        from .loader import load_svdint4_model

        return (load_svdint4_model(_resolve_model_path(unet_name), external_lora_bypass=external_lora_bypass),)


NODE_CLASS_MAPPINGS = {
    "SVDInt4DiffusionModelLoader": SVDInt4DiffusionModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SVDInt4DiffusionModelLoader": "Load SVDInt4 DiT",
}
