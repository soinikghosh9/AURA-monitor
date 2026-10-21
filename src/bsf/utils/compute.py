"""Compute tiers + device resolution, so one pipeline runs on CI, a laptop GPU, and a workstation.

Tiers (``configs/tiers/*.yaml``) scale the expensive knobs (simulation sizes, training epochs/batch,
number of seeds) without touching code:
  * ``ci``          — tiny, CPU-only; the smoke tier for CI and quick checks.
  * ``laptop``      — modest (e.g. GTX 1650 4 GB / Ryzen); the Phase-1 defaults.
  * ``workstation`` — full (e.g. RTX 5070 8 GB / 32 GB).

Blackwell note: RTX 50-series (sm_120) needs PyTorch CUDA-12.8 wheels
(``pip install torch --index-url https://download.pytorch.org/whl/cu128``); ``describe_compute``
detects an unsupported arch and prints that guidance instead of crashing at the first CUDA op.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bsf.utils.io import load_yaml


@dataclass
class Tier:
    name: str
    device: str  # 'auto' | 'cpu' | 'cuda'
    scaling: dict
    csd: dict
    epileptor: dict
    cycle: dict
    detector: dict


def _tier_path(name: str) -> Path:
    candidates = [
        Path("configs/tiers") / f"{name}.yaml",
        Path(__file__).resolve().parents[3] / "configs" / "tiers" / f"{name}.yaml",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"tier {name!r} not found (looked in {[str(c) for c in candidates]}); "
        "available tiers live in configs/tiers/"
    )


def load_tier(name: str = "laptop") -> Tier:
    cfg = load_yaml(_tier_path(name))
    return Tier(
        name=cfg.get("name", name),
        device=cfg.get("device", "auto"),
        scaling=cfg.get("scaling", {}),
        csd=cfg.get("csd", {}),
        epileptor=cfg.get("epileptor", {}),
        cycle=cfg.get("cycle", {}),
        detector=cfg.get("detector", {}),
    )


def resolve_device(pref: str = "auto") -> str:
    """Return 'cuda' or 'cpu'. ``pref='cpu'`` forces CPU; 'auto'/'cuda' use CUDA when available."""
    if pref == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def describe_compute(pref: str = "auto") -> dict:
    """Structured report of the available compute, including a Blackwell/cu128 warning if needed."""
    info: dict = {"requested": pref, "device": "cpu", "torch": None}
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["device"] = resolve_device(pref)
            info["cuda"] = torch.version.cuda
            info["gpu"] = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            info["capability"] = f"sm_{cap[0]}{cap[1]}"
            try:
                archs = list(torch.cuda.get_arch_list())
            except Exception:
                archs = []
            info["torch_arch_list"] = archs
            if archs and info["capability"] not in archs:
                info["warning"] = (
                    f"GPU {info['gpu']} ({info['capability']}) is not in this torch build's arch "
                    f"list {archs}. For Blackwell RTX 50-series install CUDA-12.8 wheels:\n"
                    f"    pip install torch --index-url https://download.pytorch.org/whl/cu128"
                )
        else:
            info["note"] = "CUDA not available; using CPU"
    except ImportError:
        info["note"] = "torch not installed (pip install -e '.[deep]')"
    return info


def format_compute(pref: str = "auto") -> str:
    info = describe_compute(pref)
    lines = [f"device     : {info['device']} (requested {info['requested']})",
             f"torch      : {info.get('torch')}"]
    if info.get("gpu"):
        lines.append(f"gpu        : {info['gpu']} ({info.get('capability')}, CUDA {info.get('cuda')})")
    if info.get("note"):
        lines.append(f"note       : {info['note']}")
    if info.get("warning"):
        lines.append(f"WARNING    : {info['warning']}")
    return "\n".join(lines)
