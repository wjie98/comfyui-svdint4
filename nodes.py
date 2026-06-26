from __future__ import annotations

import os
from pathlib import Path

import folder_paths

from .loader import is_svdint4_file, load_svdint4_model


FOLDER_NAME = "svdint4"
MODEL_EXTENSIONS = {".safetensors", ".sft"}


def _register_model_dirs() -> None:
    roots = [str(Path(folder_paths.models_dir) / "svdint4")]
    env_roots = os.environ.get("SVDINT4_MODEL_PATHS", "")
    roots.extend([item for item in env_roots.split(os.pathsep) if item])
    folder_paths.folder_names_and_paths[FOLDER_NAME] = (roots, MODEL_EXTENSIONS)


def _model_names() -> list[str]:
    _register_model_dirs()
    names: list[str] = []
    for name in folder_paths.get_filename_list(FOLDER_NAME):
        path = folder_paths.get_full_path(FOLDER_NAME, name)
        if path is not None and is_svdint4_file(path):
            names.append(name)
    return names or ["manual"]


def _resolve_model_path(model_name: str) -> str:
    _register_model_dirs()
    if model_name == "manual":
        raise ValueError(f"Put SVDInt4 .safetensors files in {Path(folder_paths.models_dir) / 'svdint4'}")
    return folder_paths.get_full_path_or_raise(FOLDER_NAME, model_name)


class SVDInt4ModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model_name": (_model_names(),)}}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "svdint4"
    TITLE = "SVDInt4 Model Loader"

    def load_model(self, model_name: str):
        return (load_svdint4_model(_resolve_model_path(model_name)),)


NODE_CLASS_MAPPINGS = {
    "SVDInt4ModelLoader": SVDInt4ModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SVDInt4ModelLoader": "SVDInt4 Model Loader",
}
