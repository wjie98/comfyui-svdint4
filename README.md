# ComfyUI SVDInt4

ComfyUI custom node for loading SVDInt4-quantized Wan/Bernini DiT models.

The plugin keeps the base model in packed INT4 form and runs the built-in
SVDQuant low-rank branch inside the CUDA kernel. Regular user LoRAs remain
normal ComfyUI LoRAs and should be applied after the SVDInt4 loader node.

## Requirements

- NVIDIA GPU with CUDA support
- Turing, Ampere, Ada, or newer GPU architecture
- Python 3.10 or newer
- PyTorch with CUDA
- CUDA toolkit with `nvcc` if installing from source
- ComfyUI

The default source build targets `sm_75`, `sm_80`, `sm_86`, and `sm_89`.

## Installation

Clone the plugin into ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/wjie98/comfyui-svdint4.git
```

The ComfyUI plugin itself has no heavy Python dependency. This lets ComfyUI
Manager install or update the custom node without compiling CUDA code.

Install the CUDA kernel in the same Python environment that runs ComfyUI:

```bash
python -m pip install -v --no-build-isolation --no-cache-dir --no-deps --upgrade --force-reinstall \
  "git+https://github.com/wjie98/comfyui-svdint4.git@main#subdirectory=kernel"
```

For local development from an already cloned checkout:

```bash
cd ComfyUI/custom_nodes/comfyui-svdint4
python -m pip install -v --no-build-isolation -e ./kernel
```

To limit the architectures built for your machine:

```bash
SVDINT4_ARCH_LIST="8.0;8.6" \
python -m pip install -v --no-build-isolation --no-cache-dir --no-deps --upgrade --force-reinstall \
  "git+https://github.com/wjie98/comfyui-svdint4.git@main#subdirectory=kernel"
```

Use `--no-build-isolation` so pip builds against the PyTorch already installed
in your ComfyUI environment. Use `--no-deps` to avoid reinstalling or
downloading PyTorch when rebuilding the kernel.

If you must use the SSH URL, initialize GitHub's SSH host key in the same
Windows account first:

```powershell
New-Item -ItemType Directory -Force $env:USERPROFILE\.ssh | Out-Null
ssh-keyscan github.com | Out-File -Append -Encoding ascii $env:USERPROFILE\.ssh\known_hosts
ssh -T git@github.com
```

Verify the printed fingerprint against GitHub's published SSH key
fingerprints before trusting the host key.

## Verify The Kernel

```bash
python - <<'PY'
import torch
import svdint4
from svdint4.ops import svd_int4_linear

print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("svdint4:", svdint4.__file__)
print("kernel api:", callable(svd_int4_linear))
PY
```

The installed pip distribution is named `svdint4-kernel`; the Python package is
imported as `svdint4`.

## Model Files

Place SVDInt4 DiT model files under ComfyUI's normal diffusion model folder:

```text
ComfyUI/models/diffusion_models/<model-name>.safetensors
```

Each `.safetensors` file contains one DiT branch. Wan2.2/Bernini workflows that
use separate high-noise and low-noise DiTs should use two loader nodes, one for
each file.

The file must use the SVDInt4 single-file layout:

```text
metadata:
  format = svdint4-dit-safetensors-v1
  svdint4_contract_version = 1
  has_internal_svd_lora = true
  lora_policy = packed_only

tensors:
  blocks.N.self_attn.q.qweight
  blocks.N.self_attn.q.wscales
  blocks.N.self_attn.q.svd_down
  blocks.N.self_attn.q.svd_up
  blocks.N.self_attn.q.smooth
  blocks.N.self_attn.q.bias_packed
  ...
  non-quantized model tensors use their normal ComfyUI/Diffusers keys
```

The node scans the existing `diffusion_models` paths, including legacy
`models/unet`, and only shows files whose metadata has
`format=svdint4-dit-safetensors-v1`. For custom model locations, set
`SVDINT4_DIT_PATHS` before starting ComfyUI. Separate multiple paths with `:` on
Linux/macOS or `;` on Windows. `SVDINT4_MODEL_PATHS` is still accepted for
backward compatibility.

## Converting Shard Packs

Older SVDInt4 development packs may be stored as directories with one branch per
folder:

```text
packed-model/
  high/
    manifest.json
    kept_fp16.safetensors
    block_00.safetensors
    ...
  low/
    manifest.json
    kept_fp16.safetensors
    block_00.safetensors
    ...
```

Convert each branch into one `.safetensors` file before using it in ComfyUI:

1. Start with all tensors from `kept_fp16.safetensors`.
2. Read every `block_XX.safetensors` listed by `manifest.json`.
3. Copy packed tensors into the same output file.
4. Rename old internal low-rank keys:
   - `.lora_down` -> `.svd_down`
   - `.lora_up` -> `.svd_up`
5. Save with metadata from `svdint4.packing.build_svdint4_metadata(...)`,
   including `format=svdint4-dit-safetensors-v1`. For packed SVDQuant models,
   keep `has_internal_svd_lora=true` and `lora_policy=packed_only`.

High-noise and low-noise branches should become two separate files, for example:

```text
bernini-high.safetensors
bernini-low.safetensors
```

## Nodes

- `Load SVDInt4 DiT`
  Selects one SVDInt4 DiT `.safetensors` file from `diffusion_models` and
  returns a ComfyUI `MODEL`. The node exposes:
  - `cache_mode`: `auto`, `resident`, or `stream`
  - `lora_policy`: `metadata`, `packed_only`, `external_bypass`, or `disabled`

The loader always runs the SVDInt4 kernel in FP16. This keeps the runtime path
compatible with Turing GPUs and avoids accidental BF16 dispatch on cards that do
not support it.

Packed SVDInt4 tensors are kept in CPU memory at model-load time. By default
`cache_mode=auto` keeps the active high/low branch resident on the GPU when
there is enough free VRAM, and falls back to streaming packed tensors per Linear
when there is not enough headroom. Resident packed tensors are loaded through
ComfyUI's weight-function path, so they are included in ComfyUI's loaded model
memory accounting. The loader reports the full packed model size to ComfyUI so
VRAM planning can unload other models before denoising starts.

For diagnostics, the loader logs the base fp16 bytes, packed SVDInt4 bytes,
reported model size, resident GPU cache state, and cache release bytes.

The `SVDINT4_CACHE_MODE` environment variable is kept as a legacy fallback for
direct Python loading. The ComfyUI node option should be preferred.

The node category is:

```text
SVDInt4/loaders
```

## LoRA

The packed SVDQuant low-rank tensors inside the model are part of the base
quantized model and are not user LoRAs. New files should declare this with
`has_internal_svd_lora=true` and `lora_policy=packed_only`. Old files without
these keys are treated as `packed_only`.

With `lora_policy=packed_only`, regular ComfyUI LoRA nodes may still load keys
without warnings, but patches targeting packed SVDInt4 Linear weights are not
applied. Set `lora_policy=external_bypass` only when you intentionally want a
separate user LoRA adapter path on top of the packed model. Dense `diff`/`set`
weight patches are intentionally not supported for packed SVDInt4 weights.
Repack the model only when the LoRA is meant to become part of the quantized
base.

## Smoke Tests

Local load and single-layer CUDA forward:

```bash
python custom_nodes/comfyui-svdint4/scripts/smoke_test.py \
  --model ComfyUI/models/diffusion_models/your-model.safetensors \
  --cache-mode auto \
  --lora-policy metadata
```

Real denoise smoke on a running ComfyUI server with an API-format workflow:

```bash
python custom_nodes/comfyui-svdint4/scripts/smoke_test.py \
  --workflow smoke-workflow-api.json \
  --server http://127.0.0.1:8188 \
  --steps 3
```

On Windows/Turing, run both tests after installing the kernel in the same
environment that starts ComfyUI. The first test catches loader/kernel import
and single-kernel issues; the workflow test catches DynamicVRAM, high/low DiT,
VAE/text encoder, and scheduler integration issues.

## Troubleshooting

`ModuleNotFoundError: No module named 'svdint4'`

Install the kernel in the same environment that launches ComfyUI:

```bash
python -m pip install -v --no-build-isolation --no-cache-dir --no-deps --upgrade --force-reinstall \
  "git+https://github.com/wjie98/comfyui-svdint4.git@main#subdirectory=kernel"
```

`CUDA version mismatches the version that was used to compile PyTorch`

Make sure the CUDA toolkit used by `nvcc` matches your PyTorch CUDA version.
For source builds, also make sure `--no-build-isolation` is present.

Windows runtime

The loader uses FP16 by default on every supported GPU, including Ampere and
newer cards. SVDInt4 requires Turing/sm75 or newer.

`fatal error C1083: ... cusparse.h: No such file or directory`

Update to the latest `comfyui-svdint4` commit. Older builds included heavy
PyTorch extension headers from the binding file, which could pull in PyTorch
CUDA sparse headers on Windows. The current binding uses lighter ATen/pybind
headers and does not require `cusparse.h` directly.

`Error checking compiler version for cl`

This warning can appear when MSVC prints localized diagnostics and PyTorch
cannot decode them with the active Windows code page. The build script sets
`VSLANG=1033` automatically on Windows. If you still see this warning, set it
before running pip:

```powershell
$env:VSLANG = "1033"
python -m pip install -v --no-build-isolation --no-cache-dir --no-deps --upgrade --force-reinstall `
  "git+https://github.com/wjie98/comfyui-svdint4.git@main#subdirectory=kernel"
```

The model dropdown is empty

No valid SVDInt4 DiT files were found. Put the single-file assets in
`ComfyUI/models/diffusion_models` and make sure their metadata contains
`format=svdint4-dit-safetensors-v1`.

ComfyUI starts, but generation fails when sampling

The plugin can be loaded without the CUDA extension, but inference requires the
kernel to be installed and importable from the ComfyUI environment.

## License

This project is distributed under Apache-2.0. See `kernel/LICENSE`,
`kernel/NOTICE`, and `kernel/LICENSES/` for kernel license details.
