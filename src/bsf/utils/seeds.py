"""Global seeding for reproducibility.

Two layers:
  * `set_global_seeds` pins the *global* RNG state (Python, NumPy legacy, PyTorch) and makes
    PyTorch deterministic. Call once at the start of every entry point.
  * `make_rng` returns an explicit, isolated `numpy.random.Generator` for a given seed — the
    preferred way to get reproducible randomness inside a function without touching global state.
"""

from __future__ import annotations

import os
import random

import numpy as np

DEFAULT_SEED = 1234


def set_global_seeds(seed: int = DEFAULT_SEED, *, deterministic_torch: bool = True) -> int:
    """Seed Python, NumPy (legacy global), and PyTorch (if installed). Returns the seed used.

    CPU runs are bit-reproducible. For GPU (the Blackwell/laptop tier) we also request deterministic
    CUDA kernels (R3): ``CUBLAS_WORKSPACE_CONFIG`` must be set before the first CUDA context, and
    ``use_deterministic_algorithms(warn_only=True)`` makes GEMMs deterministic without crashing on the
    few ops (e.g. cuDNN LSTM) that lack a deterministic implementation — those just warn.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before CUDA init; deterministic GEMM
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is an optional dependency (only needed for the deep detector)
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:  # pragma: no cover - very old torch
                pass
    except Exception:  # pragma: no cover - torch absent
        pass
    return seed


def make_rng(seed: int | None = DEFAULT_SEED) -> np.random.Generator:
    """Return an isolated NumPy Generator. Prefer this over the global `np.random`."""
    return np.random.default_rng(seed)
