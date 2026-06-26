# svdint4-kernel

CUDA/PyTorch extension used by ComfyUI SVDInt4.

Most users should install it from the parent `comfyui-svdint4` directory:

```bash
python -m pip install -v --no-build-isolation -e ./kernel
```

The pip distribution is named `svdint4-kernel`; the Python package is imported
as `svdint4`.

## Requirements

- Python 3.10 or newer
- PyTorch with CUDA
- CUDA toolkit with `nvcc`
- CUDA sparse development headers (`cusparse.h`)
- C++20-capable compiler
- NVIDIA GPU, `sm_75` or newer

## Build

From this `kernel/` directory:

```bash
python -m pip install -v --no-build-isolation -e .
```

By default the extension builds for `sm_75`, `sm_80`, `sm_86`, and `sm_89`.
Override this with:

```bash
SVDINT4_ARCH_LIST="8.0;8.6" \
python -m pip install -v --no-build-isolation -e .
```

If `nvcc` selects the wrong host compiler:

```bash
CXX=/path/to/g++ SVDINT4_CUDAHOSTCXX=/path/to/g++ \
python -m pip install -v --no-build-isolation -e .
```

On Windows, run from an x64 Visual Studio Developer shell.

If a Windows conda build fails with `cusparse.h: No such file or directory`,
install the CUDA sparse development headers into the same environment:

```powershell
conda install -n comfyui -c nvidia libcusparse-dev
```

If you use pip CUDA component packages instead:

```powershell
python -m pip install nvidia-cusparse-cu12
```

For custom CUDA Toolkit locations, set `SVDINT4_CUDA_INCLUDE_DIRS` to the
directory containing `cusparse.h`.

## Check

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

## License

Apache-2.0. See `LICENSE`, `NOTICE`, and `LICENSES/`.
