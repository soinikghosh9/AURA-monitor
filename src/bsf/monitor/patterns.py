"""Seizure fingerprints: where a seizure starts, how it spreads, what changes, and which earlier seizures
it resembles.

Everything is measured against the patient's own reference (the windows 35 to 5 min before onset), so a
fingerprint describes the departure from that patient's state rather than absolute EEG properties:

  ictal evidence   per derivation and window: the mean positive robust z of line length, Teager energy
                   and peak-to-peak, plus half the largest absolute z of the relative band powers
  onset map        ictal evidence over the first 16 s from onset (18 derivations)
  onset channels   the three derivations with the strongest onset evidence; side and region from the map
  latency          per derivation, the first window from 16 s before onset with evidence > 3 in two
                   consecutive windows
  spread           derivations above 3 per window; time until half of them are involved
  signature        AURA's 18 global features, robust z, median over the first three ictal windows

The fingerprint vector (unit-norm onset map and signature, concatenated) gives cosine similarity
between seizures and an agglomerative clustering into the patient's seizure types.
"""
from __future__ import annotations

import numpy as np

from bsf.monitor import ied as I
from bsf.monitor import signals as S

THR = 3.0


def evidence(Zch: np.ndarray) -> np.ndarray:
    """(n_w, 18, F) per-channel robust z -> (n_w, 18) ictal evidence."""
    pos = np.clip(Zch[..., [0, 1, 3]], 0, None).mean(-1)
    spec = np.abs(Zch[..., 5:9]).max(-1)
    return pos + 0.5 * spec


def fingerprint(Wref: np.ndarray, Wper: np.ndarray, t_rel: np.ndarray, duration_s: float | None = None) -> dict:
    """Wref (n_ref, 18, T) raw reference windows; Wper (n, 18, T) raw windows around the seizure with
    their start times relative to onset ``t_rel`` (s)."""
    from bsf.data.features import FEATURE_NAMES_PLUS, rich_plus
    zr = [S.standardise(w) for w in Wref]
    zp = [S.standardise(w) for w in Wper]
    Cref = np.stack([S.channel_features(w) for w in zr])
    Cper = np.stack([S.channel_features(w) for w in zp])
    Zch = np.stack([S.robust_z(Cper[:, c], Cref[:, c]) for c in range(Cper.shape[1])], 1)
    E = evidence(Zch)
    on = (t_rel >= 0) & (t_rel < 16)
    omap = E[on].mean(0) if on.any() else E[np.argmin(np.abs(t_rel))]
    lat = np.full(E.shape[1], np.nan)
    sel = np.flatnonzero(t_rel >= -16)
    for c in range(E.shape[1]):
        hit = [i for a, i in zip(sel[:-1], sel[1:]) if E[a, c] > THR and E[i, c] > THR]
        if hit:
            lat[c] = t_rel[hit[0] - 1]
    spread = (E > THR).sum(1)
    half = np.flatnonzero((t_rel >= 0) & (spread >= E.shape[1] / 2))
    Gref = np.stack([rich_plus(w.T, S.FS) for w in zr[:: max(1, len(zr) // 60)]])
    ict = np.flatnonzero(t_rel >= 0)[:3]
    Gp = np.stack([rich_plus(zp[i].T, S.FS) for i in ict]) if len(ict) else np.zeros((1, Gref.shape[1]))
    sig = np.median(S.robust_z(np.nan_to_num(Gp), np.nan_to_num(Gref)), 0)
    top = np.argsort(omap)[::-1][:3]
    return {"onset_map": omap.round(3).tolist(), "latency_s": [None if np.isnan(v) else float(v) for v in lat],
            "onset_channels": [S.DB18_NAMES[i] for i in top], "lateralization": I.lateralize(omap),
            "regions": I.region_scores(omap),
            "spread_curve": {"t": t_rel.round(1).tolist(), "n_channels": spread.astype(int).tolist()},
            "time_to_half_s": float(t_rel[half[0]]) if len(half) else None,
            "signature": dict(zip(FEATURE_NAMES_PLUS, np.round(sig, 3).tolist())),
            "duration_s": duration_s, "evidence": E.round(3).tolist()}


def vector(fp: dict) -> np.ndarray:
    m = np.clip(np.asarray(fp["onset_map"], float), 0, None)
    s = np.clip(np.asarray(list(fp["signature"].values()), float), -10, 10)
    return np.r_[m / (np.linalg.norm(m) + 1e-9), s / (np.linalg.norm(s) + 1e-9)] / np.sqrt(2)


def similarity(a: dict, b: dict) -> float:
    return float(vector(a) @ vector(b))


def cluster(fps: list[dict], threshold: float = 0.6) -> list[int]:
    """Seizure types within a patient: average-linkage clustering at cosine similarity ``threshold``."""
    if len(fps) < 2:
        return [0] * len(fps)
    from scipy.cluster.hierarchy import fcluster, linkage
    V = np.stack([vector(f) for f in fps])
    Z = linkage(V, "average", metric="cosine")
    return (fcluster(Z, 1 - threshold, "distance") - 1).tolist()
