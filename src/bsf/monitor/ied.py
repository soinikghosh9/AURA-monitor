"""Interictal epileptiform discharges: detection, type and channel map.

Trained on VEPISET (84 children, 29-channel routine EEG at 500 Hz, every 4-s epoch labelled IED of one
of five spatial types, or non-IED), converted to the monitor's common input: the 18-channel double
banana at 250 Hz. Only the development patients of the sealed split are used.

Patient adaptation is label-free: IEDs are a small share of any recording, so a patient's own median
epoch is a valid interictal reference, and every feature is robust-z-scored by it (in a stream, by the
preceding 30 min). The same reference makes the channel map interpretable: it shows where the
discharge departs from that patient's own background.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from bsf.monitor import signals as S

VEPISET_CH = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
              "Fz", "Cz", "Pz", "PG1", "PG2", "A1", "A2", "ECG1", "ECG2", "EMG1", "EMG2", "EMG3", "EMG4"]
CLASS = {0: "non_ied", 1: "generalized", 2: "frontal", 3: "temporal", 4: "centro_parietal", 5: "occipital"}
WIN_S = 4.0


def build_cache(root: str | Path, patients: set[str], out: str | Path, n_jobs: int = 8) -> dict:
    """Every labelled epoch of ``patients`` as raw db18 at 250 Hz (microvolts), with class, patient
    and start time (s) in its recording. Cached to ``out`` (npz)."""
    from joblib import Parallel, delayed

    from bsf.data.vepiset import _NAME_RE, discover
    out = Path(out)
    if out.exists():
        d = np.load(out, allow_pickle=False)
        return {k: d[k] for k in d.files}
    clips = [c for c in discover(root) if c.pid in patients]
    starts = [float(_NAME_RE.match(Path(c.path).name).group("start")) / c.fs for c in clips]

    def one(c):
        a = np.load(c.path).astype(np.float64)
        if a.shape[0] > a.shape[1]:
            a = a.T
        if a.shape[0] < 19:
            return None
        b = S.resample(S.db18(a[:19], VEPISET_CH[:19]), c.fs, S.FS)
        n = int(WIN_S * S.FS)
        return (b[:, :n].astype(np.float32) * 1e6) if b.shape[1] >= n - 2 and b.shape[1] >= n else None

    got = Parallel(n_jobs=n_jobs, batch_size=64)(delayed(one)(c) for c in clips)
    ok = [i for i, x in enumerate(got) if x is not None]
    X = np.stack([got[i] for i in ok])
    res = {"X": X, "cls": np.array([clips[i].cls for i in ok], np.int8), "pid": np.array([clips[i].pid for i in ok]),
           "t": np.array([starts[i] for i in ok], np.float64), "n_dropped": np.array(len(clips) - len(ok))}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **res)
    return res


def epoch_features(X: np.ndarray, n_jobs: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Raw epochs (n, 18, T) -> per-channel features (n, 18, F) and AURA's 18 global features (n, 18),
    both on the per-channel standardised epoch."""
    from joblib import Parallel, delayed

    from bsf.data.features import rich_plus

    def one(w):
        z = S.standardise(w)
        return S.channel_features(z), rich_plus(z.T, S.FS)
    out = Parallel(n_jobs=n_jobs, batch_size=128)(delayed(one)(w) for w in X)
    return (np.stack([o[0] for o in out]).astype(np.float32),
            np.nan_to_num(np.stack([o[1] for o in out])).astype(np.float32))


def amplitude_features(X: np.ndarray) -> np.ndarray:
    """Per-channel log peak-to-peak and log line length of the RAW epoch (amplitude relative to the
    patient's background is a defining IED property that per-epoch standardisation removes)."""
    ptp = np.log1p(X.max(-1) - X.min(-1))
    ll = np.log1p(np.abs(np.diff(X, axis=-1)).mean(-1))
    return np.stack([ptp, ll], -1).astype(np.float32)


def patient_z(F: np.ndarray, pid: np.ndarray) -> np.ndarray:
    """Robust z of every epoch's features by the same patient's median epoch (label-free)."""
    Z = np.empty_like(F, dtype=np.float32)
    flat = F.reshape(len(F), -1)
    for p in np.unique(pid):
        m = pid == p
        Z[m] = S.robust_z(flat[m], flat[m]).reshape(F[m].shape)
    return Z


def design(Fch: np.ndarray, Famp: np.ndarray, Fgl: np.ndarray) -> np.ndarray:
    """One row per epoch: per-channel features, channel max/median summaries, global features."""
    C = np.concatenate([Fch, Famp], -1)                       # (n, 18, F+2)
    return np.concatenate([C.reshape(len(C), -1), C.max(1), np.median(C, 1), Fgl], 1)


def channel_map(Zch: np.ndarray) -> np.ndarray:
    """(n, 18, F) patient-z per-channel features -> (n, 18) discharge evidence per derivation: the mean
    of the transient descriptors (line length, Teager energy, sharpness, peak-to-peak)."""
    return Zch[..., :4].mean(-1)


def lateralize(cmap: np.ndarray) -> dict:
    """Left/right/bilateral from one channel map (18,): share of positive evidence per side."""
    e = np.clip(np.asarray(cmap, float), 0, None)
    side = np.array(S.SIDE)
    L, R = e[side == "L"].sum(), e[side == "R"].sum()
    tot = L + R + 1e-9
    li = (L - R) / tot
    return {"index": round(float(li), 3), "side": "left" if li > 0.2 else "right" if li < -0.2 else "bilateral"}


def region_scores(cmap: np.ndarray) -> dict:
    e = np.clip(np.asarray(cmap, float), 0, None)
    out = {}
    for r in sorted(set(S.LOBE.values())):
        idx = [i for i, reg in enumerate(S.REGION) if S.LOBE[reg] == r]
        out[r] = round(float(e[idx].mean()), 3)
    return out


def gated_z(F: np.ndarray, pid: np.ndarray, p1: np.ndarray, q: float = 0.5) -> np.ndarray:
    """Robust z by the patient's quiet background: the epochs whose first-pass IED probability is at or
    below the patient's ``q`` quantile (label-free; approximates the non-IED reference)."""
    Z = np.empty_like(F, dtype=np.float32)
    flat = F.reshape(len(F), -1)
    for p in np.unique(pid):
        m = np.flatnonzero(pid == p)
        ref = m[p1[m] <= np.quantile(p1[m], q)]
        Z[m] = S.robust_z(flat[m], flat[ref]).reshape(F[m].shape)
    return Z


def rolling_gated_z(F: np.ndarray, t: np.ndarray, p1: np.ndarray, span_s: float = 1800.0, gap_s: float = 300.0,
                    min_ref: int = 20, q: float = 0.5) -> np.ndarray:
    """In a stream: robust z by the epochs 35 to 5 min earlier whose first-pass IED probability is at or
    below that window's ``q`` quantile; zeros until enough reference exists."""
    flat = F.reshape(len(F), -1)
    out = np.zeros_like(flat, dtype=np.float32)
    lo = np.searchsorted(t, t - gap_s - span_s, side="left")
    hi = np.searchsorted(t, t - gap_s, side="right")
    key = None
    for i in range(len(F)):
        if hi[i] - lo[i] < min_ref:
            continue
        if key != (lo[i], hi[i]):
            r = np.arange(lo[i], hi[i])
            r = r[p1[r] <= np.quantile(p1[r], q)]
            ref = flat[r]
            med = np.median(ref, 0); mad = np.median(np.abs(ref - med), 0) * 1.4826 + 1e-6
            key = (lo[i], hi[i])
        out[i] = np.clip((flat[i] - med) / mad, -20, 20)
    return out.reshape(F.shape)
