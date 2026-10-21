"""Loader for the SWEC-ETHZ iEEG Seizure Detection dataset (Zenodo 14578383, Ghosh; CC-BY-4.0).

Structure (per the Zenodo record):
  short-term/<ID>/Sz<k>.mat  — (T, M) samples x channels, fs = 512 Hz. Each clip is
      3 min preictal + seizure + 3 min postictal, so onset = 180 s and offset = T/fs - 180 s.
  long-term/ID18/ID18_<h>h.mat — hourly segments, fs = 1024 Hz, 42 channels.
  long-term/ID18/ID18_info.mat — seizure begin/end times in seconds.

Verified against the real files (2026-06-03): the array is named ``EEG`` (short-term samples-major,
long-term channels-major — both handled); ``ID18_info.mat`` holds ``seizure_begin``/``seizure_end``
(absolute seconds) + ``fs``; short clips have exactly 180 s pre/post (computed ictal durations match
the Zenodo record: ID2 96-301 s, ID3 73-125 s, ID16 67-117 s). Arrays are returned samples-major
(T, M). v5/v7 ``.mat`` load via scipy; v7.3 (HDF5) would need h5py (the optional ``.[ieeg]`` extra).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

SHORT_FS = 512.0
LONG_FS = 1024.0
PRE_POST_S = 180.0  # 3 min preictal and postictal in the short-term clips


@dataclass
class SeizureClip:
    patient: str
    fs: float
    data: np.ndarray  # (T, M) samples x channels
    onset: int  # sample index of seizure onset
    offset: int  # sample index of seizure offset
    source: str

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[1])

    @property
    def duration_s(self) -> float:
        return self.data.shape[0] / self.fs


@dataclass
class LongRecording:
    patient: str
    fs: float
    data: np.ndarray  # (T, M)
    seizures: list[tuple[int, int]]  # (onset_sample, offset_sample)
    source: str


def load_mat(path: str | Path) -> dict[str, np.ndarray]:
    """Load a .mat file (v5/v7 via scipy; v7.3/HDF5 via h5py), dropping MATLAB meta keys."""
    path = Path(path)
    try:
        raw = loadmat(path)
        return {k: v for k, v in raw.items() if not k.startswith("__")}
    except NotImplementedError:  # v7.3 == HDF5
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                f"{path} is a v7.3 .mat (HDF5); install h5py: pip install -e '.[ieeg]'"
            ) from exc
        out: dict[str, np.ndarray] = {}
        with h5py.File(path, "r") as f:
            for k in f.keys():
                out[k] = np.array(f[k]).T  # h5py reads MATLAB arrays transposed
        return out


def _pick_data_array(d: dict[str, np.ndarray]) -> np.ndarray:
    """Pick the largest 2-D numeric array (the iEEG matrix) from a loaded .mat dict."""
    best, best_size = None, -1
    for v in d.values():
        a = np.asarray(v)
        if a.ndim == 2 and np.issubdtype(a.dtype, np.number) and a.size > best_size:
            best, best_size = a, a.size
    if best is None:
        raise ValueError("no 2-D numeric array found in .mat file")
    return best


def _as_samples_major(a: np.ndarray) -> np.ndarray:
    """Ensure (T, M) with T >= M (iEEG has far more samples than channels)."""
    a = np.asarray(a, dtype=float)
    return a if a.shape[0] >= a.shape[1] else a.T


def load_short_seizure(
    path: str | Path, patient: str | None = None, fs: float = SHORT_FS, pre_post_s: float = PRE_POST_S
) -> SeizureClip:
    """Load one short-term seizure clip; onset/offset from the fixed 3-min pre/post convention."""
    data = _as_samples_major(_pick_data_array(load_mat(path)))
    pre = int(round(pre_post_s * fs))
    onset = min(pre, data.shape[0] - 1)
    offset = max(onset + 1, data.shape[0] - pre)
    return SeizureClip(patient or Path(path).parent.name, fs, data, onset, offset, str(path))


def load_short_patient(folder: str | Path, fs: float = SHORT_FS) -> list[SeizureClip]:
    folder = Path(folder)
    return [load_short_seizure(f, patient=folder.name, fs=fs) for f in sorted(folder.glob("Sz*.mat"))]


def discover(root: str | Path) -> dict[str, list[Path]]:
    """Map patient ID -> sorted list of Sz*.mat files found anywhere under ``root``."""
    root = Path(root)
    out: dict[str, list[Path]] = {}
    for sub in sorted(p for p in root.rglob("*") if p.is_dir()):
        szs = sorted(sub.glob("Sz*.mat"))
        if szs:
            out[sub.name] = szs
    return out


def load_long_info(path: str | Path, default_fs: float = LONG_FS) -> tuple[float, list[tuple[float, float]]]:
    """Parse ID18_info.mat -> (fs, [(begin_s, end_s), ...]) in ABSOLUTE seconds from recording start.

    Real format: separate ``seizure_begin``/``seizure_end`` (N×1) + ``fs``. Falls back to a single
    2-column [begin,end] array if those keys are absent.
    """
    d = load_mat(path)
    if "seizure_begin" in d and "seizure_end" in d:
        beg = np.asarray(d["seizure_begin"], dtype=float).ravel()
        end = np.asarray(d["seizure_end"], dtype=float).ravel()
        fs = float(np.squeeze(d["fs"])) if "fs" in d else default_fs
        return fs, list(zip(beg.tolist(), end.tolist()))
    for v in d.values():  # fallback: a single 2-column array
        a = np.asarray(v, dtype=float)
        if a.ndim == 2 and 2 in a.shape:
            pairs = a if a.shape[1] == 2 else a.T
            return default_fs, [(float(b), float(e)) for b, e in pairs]
    return default_fs, []


def _hour_from_name(path: str | Path) -> int | None:
    """Parse the hour offset from a long-term segment name like ``ID18_183h.mat``."""
    m = re.search(r"_(\d+)h\.mat$", Path(path).name)
    return int(m.group(1)) if m else None


def load_long_segment(
    path: str | Path,
    seizures_abs_s: list[tuple[float, float]] | None = None,
    fs: float = LONG_FS,
    patient: str = "ID18",
) -> LongRecording:
    """Load one hourly long-term segment; map any whole-recording seizures into local samples.

    The segment ``ID18_<h>h.mat`` covers absolute time [h·3600, (h+1)·3600) s, so a seizure at
    absolute ``b`` s lands at local sample ``(b − h·3600)·fs``.
    """
    data = _as_samples_major(_pick_data_array(load_mat(path)))
    hour = _hour_from_name(path)
    local: list[tuple[int, int]] = []
    if seizures_abs_s and hour is not None:
        t0 = hour * 3600.0
        t1 = t0 + data.shape[0] / fs
        for b, e in seizures_abs_s:
            if e > t0 and b < t1:  # seizure overlaps this hour
                local.append((max(0, int(round((b - t0) * fs))),
                              min(data.shape[0], int(round((e - t0) * fs)))))
    return LongRecording(patient, fs, data, local, str(path))


def load_long(folder: str | Path) -> list[LongRecording]:
    """Load all ``ID18_*h`` segments + ``ID18_info.mat``; return segments with mapped seizures."""
    folder = Path(folder)
    info = folder / "ID18_info.mat"
    fs, seiz = load_long_info(info) if info.exists() else (LONG_FS, [])
    segments = sorted(p for p in folder.glob("*h.mat"))
    return [load_long_segment(p, seizures_abs_s=seiz, fs=fs) for p in segments]
