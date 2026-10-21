"""Rich multichannel / multiband features — addressing the representation bottleneck (G2).

The audit found the real-data near-chance results trace to a feature representation that throws away
the signal the field actually uses: ``slow_feature`` keeps ONE channel, band-limits to 1-45 Hz
(excluding high-frequency oscillations), and collapses everything into a single band-RMS envelope.
This module computes a much richer per-window vector from ALL channels:

  * **multi-band log power** (delta/theta/alpha/beta/gamma/HFO) — the spectral evolution preictally,
    including the >45 Hz band the envelope discards;
  * **spatial spread** (std across channels per band) — seizures are network, not single-channel;
  * **cross-channel synchrony** (mean |correlation|) — a documented preictal network marker;
  * **line-length** and **spectral-edge frequency**.

~15 causal features per window. Drop-in for the continuous streamers and the clip evaluators; the
hypothesis is that this representation carries the preictal signal the envelope loses.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, sosfilt, welch

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30),
         "gamma": (30, 70), "hfo": (70, 150)}
FEATURE_NAMES = ([f"{b}_pow_mean" for b in BANDS] + [f"{b}_pow_spread" for b in BANDS]
                 + ["line_length", "synchrony", "spectral_edge"])
PHYS_NAMES = ["spike_rate", "kurtosis", "theta_gamma_pac"]
FEATURE_NAMES_PLUS = FEATURE_NAMES + PHYS_NAMES


def rich_features(window: np.ndarray, fs: float, bands=BANDS) -> np.ndarray:
    """(T, M) multichannel window -> rich feature vector (multi-band x spatial + synchrony + shape)."""
    x = np.nan_to_num(np.asarray(window, dtype=float))
    if x.ndim == 1:
        x = x[:, None]
    nyq = fs / 2.0
    nper = int(min(max(64, fs * 2), x.shape[0]))
    f, psd = welch(x, fs=fs, nperseg=nper, axis=0)          # (Fbins, M)
    means, spreads = [], []
    for lo, hi in bands.values():
        hi = min(hi, nyq - 1)
        if hi <= lo:
            means.append(0.0); spreads.append(0.0); continue
        bp = psd[(f >= lo) & (f < hi)].sum(0)               # (M,) band power per channel
        means.append(float(np.log10(bp.mean() + 1e-12)))    # average power in band
        spreads.append(float(np.log10(bp.std() + 1e-12)))   # spatial spread of the band
    ll = float(np.log10(np.mean(np.abs(np.diff(x, axis=0))) + 1e-12))
    if x.shape[1] > 1:
        c = np.nan_to_num(np.corrcoef(x.T))      # constant channels -> NaN corr; zero them
        n = c.shape[0]
        sync = float((np.abs(c).sum() - n) / (n * (n - 1) + 1e-9))  # mean |off-diagonal correlation|
    else:
        sync = 0.0
    tot = psd.mean(1)
    cum = np.cumsum(tot)
    sef = float(f[min(np.searchsorted(cum, 0.5 * cum[-1]), len(f) - 1)]) if cum[-1] > 0 else 0.0
    vec = np.asarray(means + spreads + [ll, sync, sef], dtype=np.float32)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)  # never emit NaN/inf downstream


def physiological_features(window: np.ndarray, fs: float, pac_max_s: float | None = None) -> np.ndarray:
    """Clinically-motivated preictal markers: spike/IED rate, kurtosis (peakiness), and theta-gamma
    phase-amplitude coupling. Especially relevant for SCALP, where HFOs are skull-attenuated.

    ``pac_max_s`` truncates the PAC computation to the LAST ``pac_max_s`` seconds of the window
    (``None`` = the full window, the production setting). This exists for the ablation that tests
    whether PAC needs the whole preictal context: the coupling evolves across the approach, so a short
    snippet should lose the physiology gain. Truncation is causal -- it keeps the samples nearest the
    forecast instant and drops the oldest.
    """
    from scipy.signal import decimate
    from scipy.stats import kurtosis as _kurt
    x = np.nan_to_num(np.asarray(window, dtype=float))
    if x.ndim == 1:
        x = x[:, None]
    nyq = fs / 2.0
    # spike/IED rate: high-pass >10 Hz, fraction of samples exceeding 4x MAD (sharp transients). Cheap
    # on the full signal.
    hsos = butter(4, min(10.0, 0.5 * nyq) / nyq, btype="high", output="sos")
    hp = sosfilt(hsos, x, axis=0)
    mad = np.median(np.abs(hp - np.median(hp, 0)), 0) + 1e-9
    spike = float(np.mean(np.mean(np.abs(hp) > 4 * mad, 0)))
    kurt = float(np.mean(np.nan_to_num(_kurt(x, axis=0))))  # peaky distribution ~ spikes
    # theta-gamma PAC (Canolty mean-vector-length) over the FULL window (it evolves across the
    # pre-ictal context — truncating loses it), but DECIMATED to ~160 Hz so the per-channel Hilbert
    # stays cheap on long clips. PAC bands (<=70 Hz) are preserved.
    q = max(1, int(fs // 160))
    xp = x
    if pac_max_s is not None:
        keep = int(round(pac_max_s * fs))
        if 0 < keep < xp.shape[0]:
            xp = xp[-keep:]                      # last pac_max_s seconds (causal truncation)
    xd = decimate(xp, q, axis=0, ftype="fir") if q > 1 else xp
    nyqd = (fs / q) / 2.0
    pac = 0.0
    ghi = min(70.0, 0.95 * nyqd)
    if nyqd > 9 and ghi > 31:
        sl = butter(4, [4.0 / nyqd, 8.0 / nyqd], btype="band", output="sos")
        sh = butter(4, [30.0 / nyqd, ghi / nyqd], btype="band", output="sos")
        pacs = []
        for ch in range(min(8, xd.shape[1])):
            ph = np.angle(hilbert(sosfilt(sl, xd[:, ch])))
            amp = np.abs(hilbert(sosfilt(sh, xd[:, ch])))
            pacs.append(np.abs(np.mean(amp * np.exp(1j * ph))) / (np.mean(amp) + 1e-12))
        pac = float(np.mean(pacs)) if pacs else 0.0
    return np.nan_to_num(np.asarray([spike, kurt, pac], dtype=np.float32))


def rich_plus(window: np.ndarray, fs: float, pac_max_s: float | None = None) -> np.ndarray:
    """Rich spectral/spatial features + physiological markers (18-D). Drop-in for rich_features."""
    return np.concatenate([rich_features(window, fs),
                           physiological_features(window, fs, pac_max_s=pac_max_s)])


def history_features(seizures: np.ndarray, dt_h: float = 1.0, halflives_h=(24.0, 168.0)) -> np.ndarray:
    """Causal seizure-history / self-excitation features (the hazard-model payload, Front C).

    Seizures cluster (self-excitation) and ride multidien cycles, so recent seizure activity is a
    day-ahead predictor a Hawkes/point-process would model. We compute, using only seizures STRICTLY
    before each time $t$: log time-since-last-seizure (normalized), and an exponentially-weighted recent
    seizure rate at each half-life in ``halflives_h`` (fast clustering vs. slow multidien load).
    Returns ``(n, 1+len(halflives_h))``; deterministic from observed past events, hence leak-free.
    """
    s = np.asarray(seizures); n = len(s)
    decays = [float(np.exp(-dt_h * np.log(2.0) / h)) for h in halflives_h]
    acc = [0.0] * len(decays)
    t_since = np.full(n, float(n)); last = -1
    ewma = np.zeros((n, len(decays)), dtype=float)
    for t in range(n):
        for i, d in enumerate(decays):          # decay the running rate to time t (past events only)
            acc[i] *= d
            ewma[t, i] = acc[i]
        t_since[t] = (t - last) if last >= 0 else n
        if s[t] > 0:                            # record AFTER reading features at t -> strictly causal
            last = t
            for i in range(len(decays)):
                acc[i] += 1.0
    ts = np.log1p(t_since) / np.log1p(n)
    return np.column_stack([ts, ewma]).astype(np.float32)


def connectivity_features(window: np.ndarray, fs: float, k: int = 3) -> np.ndarray:
    """Network/graph descriptor of the channel correlation matrix (Front A: seizures are network events).

    The scalar ``synchrony`` (mean $|$off-diagonal corr$|$) collapses the spatial signature; here we keep
    the top-``k`` eigenvalues of the channel correlation matrix (normalized by channel count, so
    $\\lambda_1$ is the dominant synchronous mode) plus the normalized eigenvalue entropy (network
    dispersion: low when one mode dominates, high when activity is distributed). Causal (within-window),
    bounded, zero for $<3$ channels. Returns ``k+1`` features.
    """
    x = np.nan_to_num(np.asarray(window, dtype=float))
    if x.ndim == 1 or x.shape[1] < 3:
        return np.zeros(k + 1, dtype=np.float32)
    C = np.nan_to_num(np.corrcoef(x.T))
    M = C.shape[0]
    w = np.sort(np.linalg.eigvalsh((C + C.T) * 0.5))[::-1]   # descending eigenvalues
    top = list(w[:k] / M) + [0.0] * max(0, k - len(w))
    p = np.clip(w, 1e-9, None); p = p / p.sum()
    ent = float(-(p * np.log(p)).sum() / np.log(M))          # normalized spectral entropy in [0,1]
    return np.nan_to_num(np.asarray(top[:k] + [ent], dtype=np.float32))


def clock_features(n: int, periods_h=(24.0, 168.0), dt_h: float = 1.0, t0: float = 0.0) -> np.ndarray:
    """Deterministic circadian/multidien CLOCK features: sin/cos at each period (Front C, day-ahead).

    Returns ``(n, 2*len(periods))`` of ``[sin(2pi t/P), cos(2pi t/P)]`` per period ``P`` (hours), where
    ``t = t0 + dt_h*arange(n)``. Purely a function of elapsed time, so it is exactly causal and
    leakage-free -- a deterministic clock, not an estimate. Time-of-day / multidien phase is the single
    most reliable day-ahead covariate (Karoly/Baud); the offset to true wall-clock is learnable. Assumes
    contiguous sampling at ``dt_h`` (canine/scalp streams are hourly/regular); gaps only blur the phase.
    """
    t = t0 + dt_h * np.arange(int(n), dtype=np.float64)
    cols = []
    for P in periods_h:
        w = 2.0 * np.pi / float(P)
        cols.append(np.sin(w * t)); cols.append(np.cos(w * t))
    return np.asarray(np.column_stack(cols), dtype=np.float32)
