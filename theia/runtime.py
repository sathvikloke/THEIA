"""Device and mixed-precision resolution.

One place decides where tensors live and whether autocast is real, so no module
has to re-derive it (and get it wrong). The previous code hard-coded
`cuda if available else cpu`, which silently ignored Apple Silicon, and asked for
`torch.cuda.amp` on machines with no CUDA, where it disables itself with a warning
nobody reads.
"""
from __future__ import annotations

import torch


def resolve_device(pref: str = "auto") -> torch.device:
    """'auto' picks cuda > mps > cpu. Anything else is honoured verbatim."""
    if pref and pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_settings(device: torch.device, want_amp: bool) -> tuple[bool, torch.dtype]:
    """Return (enabled, dtype) for autocast on this device.

    Only CUDA gets fp16. MPS autocast is bf16 and still patchy across ops, and CPU
    autocast buys nothing here, so both are reported honestly as disabled rather
    than silently degrading.
    """
    if not want_amp:
        return False, torch.float32
    if device.type == "cuda":
        return True, torch.float16
    return False, torch.float32


def make_grad_scaler(device: torch.device, enabled: bool):
    """GradScaler is a CUDA-only concept; elsewhere return a disabled one."""
    if device.type == "cuda":
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.amp.GradScaler("cpu", enabled=False)


def autocast(device: torch.device, enabled: bool, dtype: torch.dtype):
    return torch.amp.autocast(device_type=device.type, enabled=enabled, dtype=dtype)


def describe(device: torch.device, amp_on: bool) -> str:
    bits = [f"device={device.type}"]
    if device.type == "cuda":
        bits.append(torch.cuda.get_device_name(0))
    bits.append(f"amp={'on' if amp_on else 'off'}")
    return " | ".join(bits)
