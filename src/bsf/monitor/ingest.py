"""Reading a patient's recordings onto one clock, on the common montage.

CHB-MIT and Siena recordings are located and their seizures placed on an absolute timeline by
``scripts/extract_scalp_raw.timeline`` (the same arbitration every analysis in the paper uses; the
scripts directory must be importable). Time is seconds on the patient's clock, so ``t % 86400`` is the
time of day.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bsf.monitor import signals as S


@dataclass
class Record:
    path: Path
    t0: float           # absolute start (s)
    dur: float          # duration (s)


def patient(cohort: str, pdir: Path) -> tuple[list[Record], list[tuple[float, float]]]:
    """Recordings in time order and annotated seizures [(on, off)] on the absolute clock."""
    import mne

    from extract_scalp_raw import timeline
    edfs = sorted(Path(pdir).glob("*.edf"))
    tl = timeline("chbmit" if cohort == "chbmit" else "siena", Path(pdir), edfs)
    recs = []
    for e in edfs:
        if e.name not in tl["starts"]:
            continue
        try:
            r = mne.io.read_raw_edf(e, preload=False, verbose="ERROR")
        except Exception:  # noqa: BLE001
            continue
        recs.append(Record(e, float(tl["starts"][e.name]), r.n_times / r.info["sfreq"]))
    recs.sort(key=lambda r: r.t0)
    return recs, sorted((float(a), float(b)) for a, b in tl["spans"])


def read(rec: Record, t_from: float | None = None, t_to: float | None = None) -> tuple[np.ndarray, float] | None:
    """(18, n) double banana at 250 Hz (volts) for [t_from, t_to] (absolute, clipped to the record),
    and the absolute time of its first sample; None if the montage cannot be formed."""
    import mne
    r = mne.io.read_raw_edf(rec.path, preload=False, verbose="ERROR")
    a = 0.0 if t_from is None else max(0.0, t_from - rec.t0)
    b = rec.dur if t_to is None else min(rec.dur, t_to - rec.t0)
    if b - a < 1.0:
        return None
    r.crop(tmin=a, tmax=min(b, r.times[-1]))
    x = S.db18(r.get_data(), r.ch_names)
    if x is None:
        return None
    return S.resample(x, r.info["sfreq"], S.FS).astype(np.float32), rec.t0 + a


def windows(x: np.ndarray, t0: float, win_s: float = 8.0, stride_s: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    n = int(round(win_s * S.FS)); st = int(round(stride_s * S.FS))
    idx = np.arange(0, x.shape[1] - n + 1, st)
    if not len(idx):
        return np.zeros((0, x.shape[0], n), np.float32), np.zeros(0)
    return np.stack([x[:, i:i + n] for i in idx]), t0 + idx / S.FS


def segment(recs: list[Record], t_from: float, t_to: float, win_s: float = 8.0, stride_s: float = 8.0):
    """Raw windows and their absolute start times covering [t_from, t_to] across recordings."""
    Ws, Ts = [], []
    for rec in recs:
        if rec.t0 + rec.dur <= t_from or rec.t0 >= t_to:
            continue
        got = read(rec, t_from, t_to)
        if got is None:
            continue
        w, t = windows(got[0], got[1], win_s, stride_s)
        Ws.append(w); Ts.append(t)
    if not Ws:
        return np.zeros((0, 18, int(win_s * S.FS)), np.float32), np.zeros(0)
    return np.concatenate(Ws), np.concatenate(Ts)
