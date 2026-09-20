"""What this machine can actually run.

The assistant must degrade cleanly: a laptop with no GPU should be told which
models are realistic rather than being allowed to start a 24 GB model and hang.
Every probe here is defensive, because the answer "we could not tell" is always
acceptable and a crash during a hardware check never is.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GpuInfo:
    """A single visible accelerator."""

    index: int
    name: str
    total_memory_gb: float


@dataclass(frozen=True)
class HardwareReport:
    """A summary used to recommend models and pick a precision."""

    cpu_count: int
    total_ram_gb: float
    gpus: tuple[GpuInfo, ...]
    torch_available: bool
    cuda_available: bool
    mps_available: bool

    @property
    def has_gpu(self) -> bool:
        return bool(self.gpus) or self.mps_available

    @property
    def max_vram_gb(self) -> float:
        return max((gpu.total_memory_gb for gpu in self.gpus), default=0.0)

    def summary(self) -> str:
        """One line for the UI."""
        parts = [f"{self.cpu_count} CPU cores"]
        if self.total_ram_gb:
            parts.append(f"{self.total_ram_gb:.0f} GB RAM")
        if self.gpus:
            names = ", ".join(f"{gpu.name} ({gpu.total_memory_gb:.0f} GB)" for gpu in self.gpus)
            parts.append(names)
        elif self.mps_available:
            parts.append("Apple Metal GPU")
        else:
            parts.append("no GPU detected")
        return " | ".join(parts)


def _total_ram_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().total / 1e9
    except Exception:
        pass
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return (pages * page_size) / 1e9
    except (AttributeError, ValueError, OSError):
        return 0.0


def detect_hardware() -> HardwareReport:
    """Probe CPU, memory and accelerators without importing heavy modules twice."""
    cpu_count = os.cpu_count() or 1
    total_ram_gb = _total_ram_gb()

    gpus: list[GpuInfo] = []
    torch_available = False
    cuda_available = False
    mps_available = False

    try:
        import torch

        torch_available = True
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                gpus.append(
                    GpuInfo(
                        index=index,
                        name=str(properties.name),
                        total_memory_gb=properties.total_memory / 1e9,
                    )
                )
        backends = getattr(torch, "backends", None)
        mps = getattr(backends, "mps", None) if backends is not None else None
        mps_available = bool(mps is not None and mps.is_available())
    except Exception:
        # torch is an optional dependency and a broken CUDA driver can raise
        # from the import itself; either way the answer is "no usable GPU".
        pass

    return HardwareReport(
        cpu_count=cpu_count,
        total_ram_gb=total_ram_gb,
        gpus=tuple(gpus),
        torch_available=torch_available,
        cuda_available=cuda_available,
        mps_available=mps_available,
    )


def free_disk_gb(path: str | Path) -> float:
    """Free space on the filesystem holding *path*, walking up if needed."""
    candidate = Path(path).expanduser()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    try:
        return shutil.disk_usage(candidate).free / 1e9
    except OSError:
        return 0.0


def recommended_precision(report: HardwareReport, model_size_gb: float) -> str:
    """Pick a load precision for a model of *model_size_gb* on this machine.

    Returns one of ``float32``, ``bfloat16``, ``int8`` or ``int4``.  Quantized
    options are only offered on CUDA because ``bitsandbytes`` has no usable CPU
    path; a CPU host gets float32, which is slow but always correct.
    """
    if not report.cuda_available:
        return "float32"
    vram = report.max_vram_gb
    if vram <= 0 or model_size_gb <= 0:
        return "bfloat16"
    if vram >= model_size_gb * 1.25:
        return "bfloat16"
    if vram >= model_size_gb * 0.65:
        return "int8"
    return "int4"


def can_run(report: HardwareReport, *, min_vram_gb: float, cpu_capable: bool) -> tuple[bool, str]:
    """Return whether a model is runnable here, with a reason when it is not."""
    if report.max_vram_gb >= min_vram_gb > 0:
        return True, "Fits in GPU memory."
    if report.has_gpu and min_vram_gb > 0:
        if report.max_vram_gb:
            return cpu_capable, (
                f"Needs about {min_vram_gb:.0f} GB of GPU memory and this GPU has "
                f"{report.max_vram_gb:.0f} GB. It can run quantized or on CPU, more slowly."
            )
        return cpu_capable, "GPU memory could not be measured; it may need to run on CPU."
    if cpu_capable:
        return True, "No GPU detected, so this model will run on CPU. Expect slow replies."
    return False, "This model needs a GPU and none was detected."


__all__ = [
    "GpuInfo",
    "HardwareReport",
    "can_run",
    "detect_hardware",
    "free_disk_gb",
    "recommended_precision",
]
