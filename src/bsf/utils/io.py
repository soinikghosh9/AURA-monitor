"""Small, dependency-light IO helpers used across the pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=indent, default=str), encoding="utf-8")
    return path


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_yaml(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")
    return path


def load_yaml(path: str | Path) -> Any:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def save_npz(path: str | Path, **arrays: np.ndarray) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    np.savez_compressed(path, **arrays)
    return path if path.suffix else path.with_suffix(".npz")


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def latest_result_dirs(pattern: str, root: str | Path = "results",
                       include_smoke: bool = False) -> list[Path]:
    """Result directories matching ``pattern``, newest LAST, smoke runs excluded by default.

    ``sorted(glob(f"{name}_*"))[-1]`` is the obvious way to find the newest run and is wrong here:
    directories are ``<name>_<UTCstamp>``, so a sibling experiment whose name extends the prefix sorts
    after every timestamp of the base run. Concretely ``sorted(glob("raw_ssl_*"))[-1]`` returns
    ``raw_ssl_smoke_...`` rather than the real ``raw_ssl_...``, because 's' > '2'. Any aggregation
    built on that idiom silently reports smoke-test numbers.

    This sorts by the trailing UTC stamp and drops ``*_smoke_*`` runs unless asked for them.

    Patterns of the form ``"<name>_*"`` are additionally matched STRICTLY: the wildcard must expand to
    a bare UTC stamp, so ``"deep_fair_*"`` returns ``deep_fair_<stamp>`` and never
    ``deep_fair_null_<stamp>``. Excluding smoke runs by name is not enough -- ANY experiment whose name
    extends another's prefix shadows it, and that is easy to reintroduce simply by adding a follow-up
    experiment (which is exactly how ``deep_fair_null`` shadowed ``deep_fair``).
    """
    import re
    stamp_re = re.compile(r"^(\d{8}T\d{6}Z)$")
    strict_prefix = pattern[:-1] if pattern.endswith("_*") else None
    out = []
    for d in Path(root).glob(pattern):
        if not d.is_dir():
            continue
        if not include_smoke and "_smoke_" in d.name:
            continue
        if strict_prefix is not None:
            tail = d.name[len(strict_prefix):]
            if not stamp_re.match(tail):
                continue                      # a sibling experiment, not a run of this one
            out.append((tail, d))
            continue
        m = re.search(r"_(\d{8}T\d{6}Z)$", d.name)
        out.append((m.group(1) if m else "", d))
    return [d for _, d in sorted(out, key=lambda t: (t[0], t[1].name))]


def latest_result(pattern: str, root: str | Path = "results", include_smoke: bool = False):
    """Parsed ``results.json`` of the newest non-smoke run matching ``pattern`` (None if absent)."""
    for d in reversed(latest_result_dirs(pattern, root, include_smoke)):
        f = d / "results.json"
        if f.exists():
            try:
                return json.loads(f.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
    return None
