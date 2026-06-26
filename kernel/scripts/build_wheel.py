from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCH_LIST = "7.5;8.0;8.6;8.9"


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a svdint4-kernel wheel.")
    parser.add_argument("--arch-list", default=os.environ.get("SVDINT4_ARCH_LIST", DEFAULT_ARCH_LIST))
    parser.add_argument("--with-isolation", action="store_true", help="Use PEP 517 build isolation.")
    parser.add_argument("--skip-build-deps", action="store_true", help="Do not install pip/build/wheel/ninja.")
    args = parser.parse_args()

    env = os.environ.copy()
    env["SVDINT4_ARCH_LIST"] = args.arch_list

    if platform.system() == "Windows":
        if shutil.which("cl") is None:
            print("warning: cl.exe was not found; run from an x64 Visual Studio Developer shell.", file=sys.stderr)
        env.setdefault("DISTUTILS_USE_SDK", "1")
        env.setdefault("MSSdk", "1")

    if shutil.which("nvcc") is None:
        print("warning: nvcc was not found on PATH; CUDA_HOME must point at a CUDA toolkit.", file=sys.stderr)

    if not args.skip_build_deps:
        run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "build", "wheel", "ninja"], env)

    shutil.rmtree(ROOT / "build", ignore_errors=True)
    shutil.rmtree(ROOT / "svdint4.egg-info", ignore_errors=True)
    shutil.rmtree(ROOT / "svdint4_kernel.egg-info", ignore_errors=True)

    cmd = [sys.executable, "-m", "build", "--wheel"]
    if not args.with_isolation:
        cmd.extend(["--no-isolation", "--skip-dependency-check"])
    run(cmd, env)


if __name__ == "__main__":
    main()
