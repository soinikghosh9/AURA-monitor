"""Provenance stamping: every result directory records exactly how it was produced.

No result is trusted without a `provenance.json` recording the git commit, config, package
versions, platform, and seed. This is the operational backbone of the anti-fabrication policy:
a manuscript number is only admissible if its `results/` artifact carries a provenance stamp.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

# Packages whose versions materially affect numerical results.
_TRACKED = [
    "numpy", "scipy", "pandas", "scikit-learn", "statsmodels", "matplotlib",
    "torch", "ewstools", "filterpy", "particles", "lifelines", "mne",
]


def _run_git(args: list[str]) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, timeout=10
        )
        return out.stdout.strip()
    except Exception:  # pragma: no cover - git absent / not a repo
        return None


def git_commit(short: bool = False) -> str | None:
    return _run_git(["rev-parse", "--short", "HEAD"] if short else ["rev-parse", "HEAD"])


# Regenerable OUTPUT artifacts: pipelines rewrite these every run, so their modification does NOT mean
# the code that produced a result is uncommitted. Excluded from the code-dirtiness check.
_OUTPUT_PATHS = ("manuscript/figures", "results")


def git_dirty(exclude_outputs: bool = True) -> bool | None:
    """True iff the working tree has uncommitted changes.

    By default, regenerated output artifacts (``manuscript/figures``, ``results``) are EXCLUDED, so a
    True value means the *code/config* that produced a result is uncommitted — which is what
    provenance and the manuscript-grade gate actually care about. Pass ``exclude_outputs=False`` for a
    raw whole-tree check.
    """
    out = _git_status_porcelain(exclude_outputs)
    return None if out is None else bool(out)


def _git_status_porcelain(exclude_outputs: bool = True) -> str | None:
    args = ["status", "--porcelain"]
    if exclude_outputs:
        args += [".", *(f":(exclude){p}" for p in _OUTPUT_PATHS)]
    return _run_git(args)


def git_dirty_paths(exclude_outputs: bool = True) -> list[str] | None:
    """The paths making the tree dirty, so a reviewer can see WHAT was uncommitted.

    ``git_dirty()`` alone cannot distinguish "a source file was edited and not committed", which
    invalidates the provenance commit, from "an unrelated untracked document is sitting in the tree",
    which cannot affect any number. Recording the paths makes that check auditable after the fact
    instead of requiring trust in the run's operator.
    """
    out = _git_status_porcelain(exclude_outputs)
    if out is None:
        return None
    # porcelain lines are "XY <path>", but _run_git strips the output, so the first line has already
    # lost its leading status space -- split on whitespace rather than slicing a fixed offset.
    paths = []
    for ln in out.splitlines():
        parts = ln.strip().split(None, 1)
        if len(parts) == 2:
            paths.append(parts[1].strip())
    return paths


def package_versions(names: list[str] | None = None) -> dict[str, str | None]:
    names = names or _TRACKED
    versions: dict[str, str | None] = {}
    for n in names:
        try:
            versions[n] = importlib_metadata.version(n)
        except Exception:
            versions[n] = None
    return versions


def utc_stamp() -> str:
    """Compact UTC timestamp safe for directory names, e.g. 20260603T184500Z."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def provenance_dict(
    config: Any = None, seed: int | None = None, extra: dict | None = None
) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "git_dirty_paths": git_dirty_paths(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "seed": seed,
        "config": config,
        "extra": extra or {},
    }


def write_provenance(
    out_dir: str | Path, config: Any = None, seed: int | None = None, extra: dict | None = None
) -> Path:
    """Write provenance.json into an existing directory; return its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "provenance.json"
    path.write_text(
        json.dumps(provenance_dict(config, seed, extra), indent=2, default=str), encoding="utf-8"
    )
    return path


def new_results_dir(
    name: str,
    root: str | Path = "results",
    config: Any = None,
    seed: int | None = None,
    extra: dict | None = None,
) -> Path:
    """Create `results/<name>_<UTCstamp>/` stamped with provenance; return the directory.

    Emits a loud warning when the git tree is dirty (R1): a result generated on a dirty tree is not
    reconstructable from its recorded commit alone, so manuscript-grade runs must be on a clean
    checkout. Non-fatal so that exploratory/dev runs still work.
    """
    if git_dirty():
        import warnings

        paths = git_dirty_paths() or []
        code_like = [p for p in paths if p.endswith((".py", ".ps1", ".toml", ".cfg", ".yml",
                                                     ".yaml", ".json", ".txt"))]
        detail = (f" Uncommitted CODE/CONFIG: {code_like}." if code_like
                  else f" (no code/config among them; dirty paths: {paths})")
        warnings.warn(
            f"new_results_dir('{name}'): the git working tree is DIRTY, so the provenance commit "
            f"does not uniquely identify the code that produced these numbers.{detail} "
            "Commit/stash before a manuscript-grade run (R1).",
            stacklevel=2,
        )
    out_dir = Path(root) / f"{name}_{utc_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_provenance(out_dir, config=config, seed=seed, extra=extra)
    return out_dir
