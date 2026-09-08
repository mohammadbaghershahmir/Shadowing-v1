"""Deterministic seeds and run metadata capture."""

from __future__ import annotations

import os
import platform
import random
from datetime import datetime, timezone
from typing import Any

import numpy as np


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in [
        "torch",
        "torchvision",
        "transformers",
        "timm",
        "numpy",
        "sklearn",
        "matplotlib",
        "PIL",
        "gradio",
        "huggingface_hub",
        "yaml",
    ]:
        try:
            if name == "PIL":
                import PIL

                versions["Pillow"] = getattr(PIL, "__version__", "unknown")
            elif name == "sklearn":
                import sklearn

                versions["scikit-learn"] = sklearn.__version__
            elif name == "yaml":
                import yaml

                versions["PyYAML"] = getattr(yaml, "__version__", "unknown")
            else:
                mod = __import__(name)
                versions[name] = getattr(mod, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001
            versions[name] = f"unavailable: {exc}"
    return versions


def cuda_gpu_metadata() -> dict[str, Any]:
    meta: dict[str, Any] = {
        "cuda_available": False,
        "torch_version": None,
        "torch_cuda": None,
        "device_name": None,
        "total_vram_mb": None,
        "bf16_supported": False,
        "fp16_supported": False,
        "driver": None,
    }
    try:
        import torch

        meta["torch_version"] = torch.__version__
        meta["torch_cuda"] = torch.version.cuda
        meta["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            meta["device_name"] = props.name
            meta["total_vram_mb"] = round(props.total_memory / (1024**2), 1)
            meta["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
            meta["fp16_supported"] = True
            try:
                import subprocess

                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                meta["driver"] = out.strip().splitlines()[0]
            except Exception:  # noqa: BLE001
                meta["driver"] = None
    except ImportError:
        pass
    return meta


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def base_run_metadata(seed: int, experiment_cfg: dict[str, Any]) -> dict[str, Any]:
    import sys

    return {
        "run_id": make_run_id(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": package_versions(),
        "cuda_gpu": cuda_gpu_metadata(),
        "experiment": experiment_cfg,
        "warning": (
            "Upsampled feature maps are not true pixel-level predictions. "
            "PCA colors are artificial and are not carpet colors. "
            "K-means clusters are structural feature clusters, not color palettes."
        ),
    }
