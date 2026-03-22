"""
Device abstraction layer for cross-platform PyTorch support.

Provides unified device selection across CUDA, MPS (Apple Silicon), and CPU,
so the Point2CAD pipeline works on Linux (NVIDIA GPU), macOS (Apple Silicon
or Intel), and any CPU-only machine.
"""

import torch
import warnings


def select_device(preferred=None):
    """Select the best available compute device.

    Priority: cuda > mps > cpu. If ``preferred`` is given, it is tried first.

    Args:
        preferred: Optional device string ("cuda", "mps", "cpu") or
            ``torch.device``. If the requested backend is unavailable a
            warning is emitted and the next-best device is returned.

    Returns:
        torch.device
    """
    if preferred is not None:
        preferred = str(preferred)
        if preferred == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if preferred == "mps" and _mps_available():
            return torch.device("mps")
        if preferred == "cpu":
            return torch.device("cpu")
        if preferred not in ("cuda", "mps", "cpu"):
            warnings.warn(f"Unknown device '{preferred}', falling back to auto-detect")
        else:
            warnings.warn(f"Requested device '{preferred}' is not available, falling back")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def empty_cache(device):
    """Free cached memory on the given device (no-op for CPU)."""
    device = torch.device(device) if isinstance(device, str) else device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def wrap_model_for_device(model, device):
    """Wrap model in DataParallel when running on CUDA, pass-through otherwise.

    ``torch.nn.DataParallel`` only supports CUDA. On MPS and CPU the model
    is returned unwrapped.
    """
    if device.type == "cuda" and torch.cuda.device_count() >= 1:
        return torch.nn.DataParallel(model, device_ids=[0])
    return model


def load_checkpoint(path, device):
    """Load a PyTorch checkpoint with the correct ``map_location``.

    Also handles the ``module.`` key prefix that ``DataParallel`` adds:
    if the checkpoint keys start with ``module.`` but the current setup
    does not use DataParallel, the prefix is stripped automatically.

    Args:
        path: Path to the ``.pth`` checkpoint file.
        device: Target device (used as ``map_location``).

    Returns:
        The loaded state dict.
    """
    state_dict = torch.load(path, map_location=device)

    # Strip DataParallel 'module.' prefix if present
    if all(k.startswith("module.") for k in state_dict.keys()):
        if device.type != "cuda":
            state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

    return state_dict


def _mps_available():
    """Check if MPS (Metal Performance Shaders) backend is available."""
    return (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )
