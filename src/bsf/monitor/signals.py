"""Common montage, resampling, signal quality and per-channel features.

The double banana (18 longitudinal bipolar derivations) is the single input every monitor model sees:
taken directly from bipolar recordings, computed from referential ones by name. Channel k is the same
derivation for every patient and every cohort, which is what the paper's montage analysis showed the
networks need.
"""
from __future__ import annotations

import re
from math import gcd

import numpy as np

DB18 = [("FP1", "F7"), ("F7", "T7"), ("T7", "P7"), ("P7", "O1"),
        ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
        ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
        ("FP2", "F8"), ("F8", "T8"), ("T8", "P8"), ("P8", "O2"),
        ("FZ", "CZ"), ("CZ", "PZ")]
DB18_NAMES = [f"{a}-{b}".replace("FP", "Fp").replace("Z", "z") for a, b in DB18]
ALIAS = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
FS = 250.0

# regions and sides of each derivation, for channel maps and lateralization
REGION = ["frontal", "temporal", "temporal", "occipital",
          "frontal", "central", "parietal", "occipital",
          "frontal", "central", "parietal", "occipital",
          "frontal", "temporal", "temporal", "occipital",
          "midline", "midline"]
SIDE = ["L"] * 8 + ["R"] * 8 + ["M", "M"]
LOBE = {"frontal": "frontal", "temporal": "temporal", "central": "centro_parietal",
        "parietal": "centro_parietal", "occipital": "occipital", "midline": "centro_parietal"}


def _norm(name: str) -> str:
    n = name.upper().replace("EEG ", "").replace("-REF", "").replace("-LE", "").strip()
    n = re.sub(r"-\d+$", "", n) if n.count("-") >= 2 else n
    return "-".join(ALIAS.get(p, p) for p in n.split("-"))


def db18(data: np.ndarray, ch_names: list[str]) -> np.ndarray | None:
    """(n_ch, n_samples) in file order -> (18, n_samples), or None if a derivation is unavailable."""
    names = [_norm(c) for c in ch_names]
    bip = {n: i for i, n in reversed(list(enumerate(names))) if "-" in n}
    ref = {n: i for i, n in reversed(list(enumerate(names))) if "-" not in n}
    out = []
    for a, b in DB18:
        if f"{a}-{b}" in bip:
            out.append(data[bip[f"{a}-{b}"]])
        elif f"{b}-{a}" in bip:
            out.append(-data[bip[f"{b}-{a}"]])
        elif a in ref and b in ref:
            out.append(data[ref[a]] - data[ref[b]])
        else:
            return None
    return np.stack(out)


def resample(x: np.ndarray, fs_in: float, fs_out: float = FS) -> np.ndarray:
    from scipy.signal import resample_poly
    up, dn = int(round(fs_out)), int(round(fs_in))
    if up == dn:
        return x
    g = gcd(up, dn)
    return resample_poly(x, up // g, dn // g, axis=-1)


def standardise(w: np.ndarray) -> np.ndarray:
    """Per-channel robust standardisation of one window (C, T), as the training clips."""
    med = np.median(w, -1, keepdims=True)
    mad = np.median(np.abs(w - med), -1, keepdims=True) * 1.4826 + 1e-12
    return np.clip((w - med) / mad, -20, 20).astype(np.float32)


# ------------------------------------------------------------------ signal quality
def quality(w: np.ndarray, fs: float = FS, line_hz: float | None = None) -> dict:
    """Per-channel quality flags of one raw window (C, T) in volts or microvolts.

    flat: near-zero variance; clipped: >1% of samples at the window's extreme value; line: >50% of the
    power in a 2 Hz band at 50 or 60 Hz; artefact: peak-to-peak > 1 mV (assuming volts if values < 1).
    Returns per-channel booleans and a window score (share of clean channels).
    """
    x = np.asarray(w, float)
    uv = x * 1e6 if np.nanmax(np.abs(x)) < 1 else x
    sd = uv.std(-1)
    flat = sd < 0.5
    ext = np.abs(uv).max(-1, keepdims=True)
    clipped = (np.abs(uv) >= 0.999 * ext).mean(-1) > 0.01
    ptp = uv.max(-1) - uv.min(-1)
    art = ptp > 1000.0
    f = np.fft.rfftfreq(x.shape[-1], 1 / fs)
    P = np.abs(np.fft.rfft(uv - uv.mean(-1, keepdims=True), axis=-1)) ** 2
    tot = P[:, (f > 0.5) & (f < min(fs / 2, 100))].sum(-1) + 1e-12
    line = np.zeros(len(x), bool)
    for hz in ((line_hz,) if line_hz else (50.0, 60.0)):
        if hz < fs / 2:
            line |= P[:, (f > hz - 1) & (f < hz + 1)].sum(-1) / tot > 0.5
    bad = flat | clipped | art | line
    return {"flat": flat, "clipped": clipped, "artefact": art, "line": line,
            "score": float(1 - bad.mean())}


# ------------------------------------------------------------------ per-channel features
CH_FEATS = ["line_length", "teager_max", "sharpness", "ptp", "kurtosis", "delta_rel", "theta_rel",
            "alpha_rel", "beta_rel"]


def channel_features(w: np.ndarray, fs: float = FS) -> np.ndarray:
    """(C, T) standardised window -> (C, len(CH_FEATS)): transient (spike/sharp-wave) and spectral
    descriptors per derivation; the substrate of IED channel maps and seizure onset maps."""
    from scipy.stats import kurtosis
    x = np.asarray(w, float)
    d = np.diff(x, axis=-1)
    ll = np.abs(d).mean(-1)
    te = (x[:, 1:-1] ** 2 - x[:, :-2] * x[:, 2:]).max(-1)
    sharp = np.abs(np.diff(d, axis=-1)).max(-1) / (np.abs(np.diff(d, axis=-1)).mean(-1) + 1e-9)
    ptp = x.max(-1) - x.min(-1)
    ku = np.nan_to_num(kurtosis(x, axis=-1))
    f = np.fft.rfftfreq(x.shape[-1], 1 / fs)
    P = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    tot = P[:, (f >= 1) & (f < 30)].sum(-1) + 1e-12
    rel = [P[:, (f >= lo) & (f < hi)].sum(-1) / tot for lo, hi in ((1, 4), (4, 8), (8, 13), (13, 30))]
    return np.nan_to_num(np.column_stack([ll, te, sharp, ptp, ku, *rel])).astype(np.float32)


def robust_z(F: np.ndarray, ref: np.ndarray) -> np.ndarray:
    med = np.median(ref, 0)
    mad = np.median(np.abs(ref - med), 0) * 1.4826 + 1e-6
    return np.clip((F - med) / mad, -20, 20)
