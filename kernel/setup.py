import os
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IS_WINDOWS = platform.system() == "Windows"

if IS_WINDOWS:
    # Keep MSVC diagnostics in English so PyTorch's compiler probe can decode
    # cl.exe output reliably on non-English Windows installations.
    os.environ.setdefault("VSLANG", "1033")

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _arch_list() -> str:
    value = os.environ.get("SVDINT4_ARCH_LIST", "7.5;8.0;8.6;8.9")
    arches = []
    for raw in value.replace(",", ";").split(";"):
        arch = raw.strip()
        if not arch:
            continue
        if arch in {"75", "80", "86", "89"}:
            arch = f"{arch[0]}.{arch[1]}"
        arches.append(arch)
    return ";".join(arches)


os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _arch_list())

COMMON_DEFINES = [
    "-DENABLE_BF16=1",
    "-DSVDINT4_MINIMAL=1",
]

HOST_CXX_STANDARD = os.environ.get(
    "SVDINT4_HOST_CXX_STANDARD",
    os.environ.get("SVDINT4_CXX_STANDARD", "c++20" if IS_WINDOWS else "c++2a"),
)
NVCC_CXX_STANDARD = os.environ.get("SVDINT4_NVCC_CXX_STANDARD", "c++20")

NVCC_FLAGS = [
    *COMMON_DEFINES,
    f"-std={NVCC_CXX_STANDARD}",
    "-O3",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--ptxas-options=--allow-expensive-optimizations=true",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_HALF2_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
]

if IS_WINDOWS:
    NVCC_FLAGS.extend(
        [
            "--use-local-env",
            "-Xcompiler",
            "/MD",
            "-Xcompiler",
            "/O2",
            "-Xcompiler",
            "/EHsc",
            "-Xcompiler",
            "/bigobj",
            "-Xcompiler",
            "/FS",
            "-Xcompiler",
            "/DNOMINMAX",
            "-Xcompiler",
            "/DWIN32_LEAN_AND_MEAN",
        ]
    )

CUDAHOSTCXX = os.environ.get("SVDINT4_CUDAHOSTCXX") or os.environ.get("CUDAHOSTCXX")
if CUDAHOSTCXX:
    NVCC_FLAGS.extend(["-ccbin", CUDAHOSTCXX])

if IS_WINDOWS:
    std_flag = HOST_CXX_STANDARD if HOST_CXX_STANDARD.startswith("/std:") else f"/std:{HOST_CXX_STANDARD}"
    CXX_FLAGS = [
        "/DENABLE_BF16=1",
        "/DSVDINT4_MINIMAL=1",
        std_flag,
        "/O2",
        "/EHsc",
        "/MD",
        "/bigobj",
        "/FS",
        "/DNOMINMAX",
        "/DWIN32_LEAN_AND_MEAN",
        "/wd4251",
        "/wd4275",
        "/wd4819",
    ]
else:
    CXX_FLAGS = [
        *COMMON_DEFINES,
        f"-std={HOST_CXX_STANDARD}",
        "-O3",
        "-fvisibility=hidden",
    ]


ext = CUDAExtension(
    name="svdint4._C",
    sources=[
        "csrc/bindings.cpp",
        "csrc/dispatcher.cu",
        "csrc/gemm_instantiations.cu",
    ],
    include_dirs=[
        str((ROOT / "csrc").resolve()),
    ],
    extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
)


setup(
    name="svdint4-kernel",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
)
