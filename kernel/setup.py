import os
import platform
import site
import sys
import sysconfig
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


ROOT = Path(__file__).resolve().parent
IS_WINDOWS = platform.system() == "Windows"


def _cuda_include_dirs() -> list[str]:
    include_dirs: list[str] = []
    seen: set[str] = set()

    def add(path: str | os.PathLike | None) -> None:
        if path is None:
            return
        candidate = Path(path)
        if candidate.is_dir():
            resolved = str(candidate.resolve())
            if resolved not in seen:
                seen.add(resolved)
                include_dirs.append(resolved)

    if CUDA_HOME:
        add(Path(CUDA_HOME) / "include")

    add(Path(sys.prefix) / "Library" / "include")
    add(Path(sys.prefix) / "include")

    site_roots: list[str | None] = []
    try:
        site_roots.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        site_roots.append(site.getusersitepackages())
    except Exception:
        pass
    site_roots.extend(
        [
            sysconfig.get_path("purelib"),
            sysconfig.get_path("platlib"),
            str(Path(sys.prefix) / "Lib" / "site-packages"),
        ]
    )
    for root in site_roots:
        if not root:
            continue
        nvidia_root = Path(root) / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for package_dir in nvidia_root.iterdir():
            add(package_dir / "include")

    for path in os.environ.get("SVDINT4_CUDA_INCLUDE_DIRS", "").split(os.pathsep):
        add(path.strip())

    return include_dirs


def _has_header(include_dirs: list[str], header: str) -> bool:
    return any((Path(path) / header).is_file() for path in include_dirs)


CUDA_INCLUDE_DIRS = _cuda_include_dirs()
if not _has_header(CUDA_INCLUDE_DIRS, "cusparse.h"):
    raise RuntimeError(
        "Could not find cusparse.h, which is required by PyTorch CUDA headers. "
        "Install CUDA sparse development headers in the same environment, then retry. "
        "For Windows conda environments, try: "
        "conda install -n comfyui -c nvidia libcusparse-dev. "
        "For pip CUDA component packages, try: "
        "python -m pip install nvidia-cusparse-cu12. "
        "If the header is already installed elsewhere, set "
        "SVDINT4_CUDA_INCLUDE_DIRS to the directory containing cusparse.h."
    )


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
        *CUDA_INCLUDE_DIRS,
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
