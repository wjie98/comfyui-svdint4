from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from urllib import error, request


def _default_comfy_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_json(url: str) -> dict:
    with request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _patch_steps(prompt: dict, steps: int) -> int:
    changed = 0
    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if isinstance(inputs, dict) and "steps" in inputs:
            inputs["steps"] = steps
            changed += 1
    return changed


def run_workflow(server: str, workflow: Path, steps: int | None, timeout_s: float, poll_s: float) -> None:
    prompt = json.loads(workflow.read_text(encoding="utf-8"))
    if steps is not None:
        changed = _patch_steps(prompt, steps)
        print(f"workflow steps patched: {changed}")

    client_id = str(uuid.uuid4())
    queued = _post_json(f"{server.rstrip('/')}/prompt", {"prompt": prompt, "client_id": client_id})
    prompt_id = queued["prompt_id"]
    print(f"queued prompt: {prompt_id}")

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        history = _get_json(f"{server.rstrip('/')}/history/{prompt_id}")
        entry = history.get(prompt_id)
        if entry is None:
            time.sleep(poll_s)
            continue
        status = entry.get("status", {})
        if status.get("completed"):
            print("workflow completed")
            return
        messages = status.get("messages") or []
        for _, payload in messages:
            if isinstance(payload, dict) and payload.get("exception_message"):
                raise RuntimeError(payload["exception_message"])
        time.sleep(poll_s)

    raise TimeoutError(f"workflow did not finish within {timeout_s:.0f}s")


def run_local_load_forward(
    comfy_root: Path,
    model_path: Path,
    skip_forward: bool,
    cache_mode: str,
    lora_policy: str,
) -> None:
    sys.path.insert(0, str(comfy_root))
    sys.path.insert(0, str(comfy_root / "custom_nodes" / "comfyui-svdint4"))

    import torch
    import comfy.model_management as model_management
    import loader

    print(f"model: {model_path}")
    if torch.cuda.is_available():
        device = model_management.get_torch_device()
        major, minor = torch.cuda.get_device_capability(device)
        print(f"cuda device: {torch.cuda.get_device_name(device)} sm{major}{minor}")
        if major * 10 + minor < 75:
            raise RuntimeError("SVDInt4 requires Turing/sm75 or newer")
    else:
        print("cuda unavailable")

    t0 = time.perf_counter()
    patcher = loader.load_svdint4_model(model_path, cache_mode=cache_mode, lora_policy=lora_policy)
    t1 = time.perf_counter()
    modules = [(name, module) for name, module in patcher.model.named_modules() if getattr(module, "is_svdint4", False)]
    print(f"loaded patcher in {t1 - t0:.3f}s; svdint4 layers: {len(modules)}; reported size: {patcher.model_size() / 1024**2:.2f} MB")

    if skip_forward:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("single-layer forward smoke requires CUDA")
    if not modules:
        raise RuntimeError("no SVDInt4 layers found")

    device = patcher.load_device
    torch.cuda.reset_peak_memory_stats(device)
    patcher.load(device_to=device, lowvram_model_memory=model_management.maximum_vram_for_weights(device), full_load=False)
    torch.cuda.synchronize(device)
    print(f"after model.load: loaded={patcher.loaded_size() / 1024**2:.2f} MB peak={torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")

    name, module = modules[0]
    x = torch.randn(2, 4, module.in_features, device=device, dtype=torch.float16)
    torch.cuda.synchronize(device)
    t2 = time.perf_counter()
    y = module(x)
    torch.cuda.synchronize(device)
    t3 = time.perf_counter()
    print(
        f"forward ok: {name} input={tuple(x.shape)} output={tuple(y.shape)} "
        f"dtype={y.dtype} time_ms={(t3 - t2) * 1000:.3f} "
        f"loaded={patcher.loaded_size() / 1024**2:.2f} MB "
        f"svd_cache={getattr(patcher.model, '_svdint4_cached_gpu_bytes', 0) / 1024**2:.2f} MB"
    )

    freed = patcher.partially_unload(patcher.offload_device, memory_to_free=1) or 0
    torch.cuda.synchronize(device)
    print(f"partial unload freed={freed / 1024**2:.2f} MB loaded={patcher.loaded_size() / 1024**2:.2f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test ComfyUI SVDInt4 loading and runtime paths.")
    parser.add_argument("--comfy-root", type=Path, default=_default_comfy_root())
    parser.add_argument("--model", type=Path, help="SVDInt4 .safetensors file for local load/forward smoke.")
    parser.add_argument("--skip-forward", action="store_true", help="Only load the model locally; do not run a CUDA layer forward.")
    parser.add_argument("--cache-mode", choices=("auto", "resident", "stream"), default="auto")
    parser.add_argument("--lora-policy", choices=("metadata", "packed_only", "external_bypass", "disabled"), default="metadata")
    parser.add_argument("--workflow", type=Path, help="ComfyUI API-format workflow JSON for a real denoise smoke.")
    parser.add_argument("--server", default="http://127.0.0.1:8188", help="Running ComfyUI server URL for --workflow.")
    parser.add_argument("--steps", type=int, help="Patch every workflow input named 'steps' to this value.")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--poll", type=float, default=1.0)
    args = parser.parse_args()

    if args.model is None and args.workflow is None:
        parser.error("provide --model, --workflow, or both")

    if args.model is not None:
        run_local_load_forward(args.comfy_root, args.model, args.skip_forward, args.cache_mode, args.lora_policy)

    if args.workflow is not None:
        try:
            run_workflow(args.server, args.workflow, args.steps, args.timeout, args.poll)
        except error.URLError as exc:
            raise RuntimeError(f"could not reach ComfyUI server at {args.server}: {exc}") from exc


if __name__ == "__main__":
    main()
