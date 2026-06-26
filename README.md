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
python -m pip install -v --no-build-isolation \
  "git+https://github.com/wjie98/comfyui-svdint4.git#subdirectory=kernel"
```

For local development from an already cloned checkout:

```bash
cd ComfyUI/custom_nodes/comfyui-svdint4
python -m pip install -v --no-build-isolation -e ./kernel
```

To limit the architectures built for your machine:

```bash
SVDINT4_ARCH_LIST="8.0;8.6" \
python -m pip install -v --no-build-isolation \
  "git+https://github.com/wjie98/comfyui-svdint4.git#subdirectory=kernel"
```

Use `--no-build-isolation` for local CUDA builds so pip uses the PyTorch already
installed in your ComfyUI environment.

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

Place SVDInt4 model files under:

```text
ComfyUI/models/svdint4/<model-name>.safetensors
```

Each `.safetensors` file contains one DiT branch. Wan2.2/Bernini workflows that
use separate high-noise and low-noise DiTs should use two loader nodes, one for
each file.

The file must use the SVDInt4 single-file layout:

```text
metadata:
  format = svdint4-dit-safetensors-v1

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

For custom model locations, set `SVDINT4_MODEL_PATHS` before starting ComfyUI.
Separate multiple paths with `:` on Linux/macOS or `;` on Windows.

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
5. Save with metadata `format=svdint4-dit-safetensors-v1`.

High-noise and low-noise branches should become two separate files, for example:

```text
bernini-high.safetensors
bernini-low.safetensors
```

## Nodes

- `SVDInt4 Model Loader`
  Selects one `.safetensors` model file and returns a ComfyUI `MODEL`.

The loader infers the kernel compute dtype from the packed tensors. There is no
separate model dtype setting.

## LoRA

Apply normal LoRA nodes after the SVDInt4 loader node. Extra LoRA is handled as
a ComfyUI-side adapter/bypass path and does not require repacking the INT4
model.

The packed SVDQuant low-rank tensors inside the model are part of the base
quantized model and are not user LoRAs.

## Troubleshooting

`ModuleNotFoundError: No module named 'svdint4'`

Install the kernel in the same environment that launches ComfyUI:

```bash
python -m pip install -v --no-build-isolation \
  "git+https://github.com/wjie98/comfyui-svdint4.git#subdirectory=kernel"
```

`CUDA version mismatches the version that was used to compile PyTorch`

Make sure the CUDA toolkit used by `nvcc` matches your PyTorch CUDA version.
For source builds, also make sure `--no-build-isolation` is present.

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
python -m pip install -v --no-build-isolation `
  "git+https://github.com/wjie98/comfyui-svdint4.git#subdirectory=kernel"
```

`Put SVDInt4 .safetensors files in ...`

No model files were found in `ComfyUI/models/svdint4`.

ComfyUI starts, but generation fails when sampling

The plugin can be loaded without the CUDA extension, but inference requires the
kernel to be installed and importable from the ComfyUI environment.

## License

This project is distributed under Apache-2.0. See `kernel/LICENSE`,
`kernel/NOTICE`, and `kernel/LICENSES/` for kernel license details.
