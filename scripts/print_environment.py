from __future__ import annotations

import importlib.metadata
import platform
import shutil
import sys


def version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


print(f"python={sys.version}")
print(f"platform={platform.platform()}")
for package in ("torch", "torchvision", "torchmil", "huggingface_hub", "numpy", "pandas", "scikit-learn", "torchmetrics", "matplotlib", "seaborn", "PyYAML", "tqdm", "pytest"):
    print(f"{package}={version(package)}")
try:
    import torch

    print(f"torch_cuda={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_device_count={torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        print(f"cuda_device_{index}={torch.cuda.get_device_name(index)}")
except Exception as exc:
    print(f"torch_probe_error={exc!r}")
total, used, free = shutil.disk_usage(".")
print(f"disk_bytes_total={total}")
print(f"disk_bytes_free={free}")

