"""Hardware probe, device resolution, and ONNX execution provider orchestration.

Provides hardware-safe resolution across CPU and NVIDIA CUDA environments,
enabling automatic GPU acceleration when drivers are present while guaranteeing
graceful degradation to CPU in headless/isolated test containers.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import onnxruntime as ort
import torch

logger = logging.getLogger(__name__)


def is_cuda_available() -> bool:
    """Safe check for CUDA availability without raising unhandled exceptions.

    Returns:
        True if torch detects functional CUDA devices and drivers, False otherwise.
    """
    try:
        return bool(torch.cuda.is_available())
    except Exception as exc:
        logger.debug("CUDA probe raised exception: %s", exc)
        return False


def resolve_torch_device(device_target: str = "auto") -> torch.device:
    """Resolve a target device string to a valid torch.device with fallback.

    Args:
        device_target: Target device ('auto', 'cuda', 'cpu', 'cuda:<id>').

    Returns:
        torch.device instance ('cuda:0' or 'cpu').
    """
    cleaned = (device_target or "auto").strip().lower()

    if cleaned == "cpu":
        return torch.device("cpu")

    cuda_ready = is_cuda_available()

    if cleaned == "auto":
        if cuda_ready:
            return torch.device("cuda:0")
        return torch.device("cpu")

    if cleaned == "cuda" or re.match(r"^cuda:\d+$", cleaned):
        if cuda_ready:
            return torch.device(cleaned if ":" in cleaned else "cuda:0")
        logger.warning(
            "CUDA device '%s' requested but CUDA is not available. Falling back to CPU.",
            cleaned,
        )
        return torch.device("cpu")

    raise ValueError(
        f"Invalid device target '{device_target}'. Expected 'auto', 'cuda', 'cpu', or 'cuda:<id>'."
    )


def resolve_ort_providers(
    device_target: str = "auto",
    preferred_provider: str | None = None,
) -> list[str]:
    """Resolve prioritized ONNX Runtime execution providers with automatic fallback.

    Args:
        device_target: Device string ('auto', 'cuda', 'cpu').
        preferred_provider: Optional explicit provider name override.

    Returns:
        List of execution providers in priority order (e.g. ['CUDAExecutionProvider', 'CPUExecutionProvider']).
    """
    available = ort.get_available_providers()
    target = (device_target or "auto").strip().lower()

    if preferred_provider:
        pref = preferred_provider.strip()
        if pref in available:
            if pref == "CUDAExecutionProvider":
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            return [pref]
        logger.warning(
            "Preferred provider '%s' not in available providers %s. Resolving automatically.",
            pref,
            available,
        )

    if target == "cpu":
        return ["CPUExecutionProvider"]

    if target in {"cuda", "auto"}:
        if "CUDAExecutionProvider" in available and is_cuda_available():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if target == "cuda":
            logger.warning(
                "CUDA requested for ONNX Runtime but CUDAExecutionProvider is unavailable. "
                "Falling back to CPUExecutionProvider."
            )
        return ["CPUExecutionProvider"]

    return ["CPUExecutionProvider"]


def get_device_telemetry() -> dict[str, Any]:
    """Retrieve runtime GPU telemetry and memory allocation statistics.

    Returns:
        Dictionary with GPU name, memory statistics, and CUDA availability status.
    """
    cuda_ready = is_cuda_available()
    telemetry: dict[str, Any] = {
        "cuda_available": cuda_ready,
        "device_count": torch.cuda.device_count() if cuda_ready else 0,
        "gpu_name": None,
        "vram_allocated_mb": None,
        "vram_reserved_mb": None,
        "vram_total_mb": None,
        "compute_capability": None,
    }

    if cuda_ready:
        try:
            telemetry["gpu_name"] = torch.cuda.get_device_name(0)
            telemetry["compute_capability"] = torch.cuda.get_device_capability(0)
            allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
            reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)
            telemetry["vram_allocated_mb"] = round(allocated, 2)
            telemetry["vram_reserved_mb"] = round(reserved, 2)
            props = torch.cuda.get_device_properties(0)
            telemetry["vram_total_mb"] = round(props.total_memory / (1024 * 1024), 2)
        except Exception as exc:
            logger.debug("Failed to read CUDA telemetry: %s", exc)

    return telemetry
