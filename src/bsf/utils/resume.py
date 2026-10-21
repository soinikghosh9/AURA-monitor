"""Lightweight resumability for long, detached experiment runs.

A run's results dir gets a fresh UTC stamp each launch (provenance), which would defeat resume. This
module gives each experiment a STABLE checkpoint dir keyed by NAME (not stamp), an atomic JSON state
keyed by sub-unit (per dog / per patient / per horizon), and torch checkpoint helpers. On restart an
experiment loads completed units and recomputes only the missing ones, then writes the merged
results.json to a normal stamped results dir. Combined with the queue journal (skip done scripts), this
makes the whole queue interruption- and offline-safe.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
CKPT_ROOT = REPO / "results" / "_adv_ckpt"


def ckpt_dir(name: str) -> Path:
    d = CKPT_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_state(name: str) -> dict:
    p = ckpt_dir(name) / "state.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def save_state(name: str, state: dict) -> None:
    _atomic_write(ckpt_dir(name) / "state.json", json.dumps(state, indent=2, default=str))


def done_units(name: str) -> dict:
    """Map of completed sub-unit -> its recorded result."""
    return dict(load_state(name).get("units", {}))


def update_unit(name: str, unit: str, value: Any) -> dict:
    """Record one completed sub-unit's result and persist immediately (atomic). Returns full state."""
    st = load_state(name)
    st.setdefault("units", {})[unit] = value
    save_state(name, st)
    return st


def save_torch_ckpt(name: str, tag: str, payload: dict) -> Path:
    import torch
    p = ckpt_dir(name) / f"{tag}.pt"
    tmp = p.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, p)
    return p


def load_torch_ckpt(name: str, tag: str):
    import torch
    p = ckpt_dir(name) / f"{tag}.pt"
    if p.exists():
        try:
            return torch.load(p, map_location="cpu", weights_only=False)
        except Exception:  # noqa: BLE001
            return None
    return None


def reset(name: str) -> None:
    d = CKPT_ROOT / name
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
