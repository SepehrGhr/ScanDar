"""Device selection and mixed precision.

Training happens on two very different machines — a 6 GB RTX 3060 laptop GPU and
whatever Colab hands out that day — so nothing in the codebase hard-codes ``cuda``
or a batch size that only fits one of them. Configs carry ``batch_size`` and
``grad_accum`` separately so the *effective* batch stays identical across both.
"""

from __future__ import annotations

from dataclasses import dataclass


def get_device(prefer: str = "auto"):
    """Return a torch device. ``prefer`` is one of ``auto``, ``cuda``, ``cpu``."""
    import torch

    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("cuda requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class DeviceInfo:
    name: str
    kind: str
    total_memory_gib: float | None
    torch_version: str
    cuda_version: str | None

    def __str__(self) -> str:
        if self.kind == "cuda":
            return (
                f"{self.name} ({self.total_memory_gib:.1f} GiB) "
                f"| torch {self.torch_version} + cu{self.cuda_version}"
            )
        return f"CPU | torch {self.torch_version}"


def describe_device(device=None) -> DeviceInfo:
    """A one-line summary of what we are training on, for run logs and reports."""
    import torch

    device = device if device is not None else get_device()
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        return DeviceInfo(
            name=props.name,
            kind="cuda",
            total_memory_gib=props.total_memory / 2**30,
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
        )
    return DeviceInfo(
        name="cpu",
        kind="cpu",
        total_memory_gib=None,
        torch_version=torch.__version__,
        cuda_version=None,
    )


def amp_enabled(device=None) -> bool:
    """Mixed precision is worth it on CUDA and pointless (or harmful) on CPU."""
    import torch

    device = device if device is not None else get_device()
    return device.type == "cuda" and torch.cuda.is_available()


def recommended_workers(requested: int | str = "auto") -> int:
    """DataLoader workers. The synthetic generator is CPU-bound, so this matters.

    ``auto`` leaves a couple of cores for the main process and the GPU feed.
    """
    import os

    if requested != "auto":
        return int(requested)
    cpus = os.cpu_count() or 2
    return max(1, min(8, cpus - 2))
