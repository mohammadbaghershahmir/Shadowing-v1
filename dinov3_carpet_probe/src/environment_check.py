"""Gigazaki / GPU environment inspection for the carpet probe."""

from __future__ import annotations

import sys
from pathlib import Path

from dinov3_carpet_probe.src.io_utils import DEFAULT_OUTPUT_ROOT, ensure_dir, write_json
from dinov3_carpet_probe.src.reproducibility import cuda_gpu_metadata, package_versions


def collect_environment() -> dict:
    import platform

    gpu = cuda_gpu_metadata()
    info = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": package_versions(),
        "cuda_gpu": gpu,
        "expected_venv_hint": r"C:\Users\Allah\Gigazaki\Scripts\python.exe",
        "gigazaki_match": "Gigazaki" in sys.executable.replace("/", "\\"),
    }
    return info


def run_environment_check(output_root: Path | None = None, require_cuda: bool = True) -> dict:
    info = collect_environment()
    out_root = Path(output_root) if output_root else DEFAULT_OUTPUT_ROOT
    env_dir = ensure_dir(out_root / "_env")
    write_json(env_dir / "environment.json", info)

    print("=== DINOv3 carpet probe environment ===")
    print(f"Python executable: {info['python_executable']}")
    print(f"Gigazaki path match: {info['gigazaki_match']}")
    print(f"Packages: {info['packages']}")
    print(f"CUDA/GPU: {info['cuda_gpu']}")
    print(f"Wrote: {env_dir / 'environment.json'}")

    if require_cuda and not info["cuda_gpu"].get("cuda_available"):
        raise RuntimeError("CUDA is not available. Carpet probe requires a CUDA GPU in Gigazaki.")
    if not info["gigazaki_match"]:
        print(
            "WARNING: Python executable does not look like Gigazaki. "
            "Activate with: C:\\Users\\Allah\\Gigazaki\\Scripts\\Activate.ps1"
        )
    return info


def main() -> None:
    run_environment_check()


if __name__ == "__main__":
    main()
