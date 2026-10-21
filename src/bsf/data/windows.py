"""Real-iEEG window pipeline: recording -> labelled, detector-ready slow-feature windows.

A *window* is a multi-minute segment (long enough for a critical-slowing approach to be observable).
Its **slow feature** is the per-block band-RMS envelope decimated to ``target_len``, then causally
detrended and normalised by mean-abs (Bury preprocessing) — the same length-L normalised series the
simulation-trained detector consumes, so sim and real are commensurable.

Labels honour the forecasting conventions (prereg §5b):
  * ``ictal``      — window centre inside a seizure.
  * ``preictal``   — centre in [onset − preictal_s − sph_s, onset − sph_s] (SPH excluded: a forecast
                     must lead the seizure, not coincide with it).
  * ``postictal``  — centre within ``postictal_s`` after an offset.
  * ``interictal`` — centre ≥ ``interictal_guard_s`` from every seizure ("true interictal").
  * ``excluded``   — the buffer zone in between (not used for either class).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt

from bsf.detect.ews_classic import detrend_causal


@dataclass
class LabeledWindow:
    start: int  # sample index of window start
    label: str  # preictal | interictal | ictal | postictal | excluded
    feature: np.ndarray  # length target_len normalised slow-feature series


def _bandpass(x: np.ndarray, fs: float, band=(1.0, 45.0), causal: bool = True) -> np.ndarray:
    """Band-pass filter. ``causal=True`` (default, DEPLOYABLE) is a forward-only ``sosfilt`` whose
    output at sample t depends only on samples ≤ t; ``causal=False`` is the zero-phase ``sosfiltfilt``
    kept ONLY as a labelled non-causal positive control (it peeks into the future and the leakage guard
    flags it). Phase-1's causal-collapse control deliberately uses ``causal=False``.
    """
    nyq = fs / 2.0
    hi = min(band[1], 0.99 * nyq)
    sos = butter(4, [band[0] / nyq, hi / nyq], btype="band", output="sos")
    x = np.asarray(x, dtype=float)
    if not causal:
        return sosfiltfilt(sos, x)
    # forward-only; steady-state initial conditions scaled by x[0] suppress the startup transient
    # without using any future sample (strictly causal).
    zi = sosfilt_zi(sos) * (x[0] if len(x) else 0.0)
    y, _ = sosfilt(sos, x, zi=zi)
    return y


def slow_feature(
    window: np.ndarray, fs: float, target_len: int = 120, band=(1.0, 45.0),
    detrend: str = "ewm", span: float = 0.2, causal: bool = True,
) -> np.ndarray:
    """Per-block band-RMS envelope -> length ``target_len``, causally detrended + mean-abs normalised.

    With ``causal=True`` (default) every stage — band-pass, block-RMS envelope, detrend — uses only
    past samples, so the feature at the window's trailing edge (the forecast instant) is deployable.
    """
    bp = _bandpass(window, fs, band, causal=causal)
    env = np.array([np.sqrt(np.mean(b**2)) for b in np.array_split(bp, target_len)])
    r = detrend_causal(env, span=span, method=detrend) if detrend != "none" else env - env.mean()
    s = float(np.mean(np.abs(r)))
    return (r / s if s > 0 else r).astype(np.float32)


def label_window(
    center: int, fs: float, seizures: list[tuple[int, int]],
    sph_s: float = 60.0, preictal_s: float = 1800.0, postictal_s: float = 600.0,
    interictal_guard_s: float = 3600.0,
) -> str:
    for on, off in seizures:
        if on <= center <= off:
            return "ictal"
        if on - (preictal_s + sph_s) * fs <= center < on - sph_s * fs:
            return "preictal"
        if off < center <= off + postictal_s * fs:
            return "postictal"
    for on, off in seizures:
        if abs(center - on) < interictal_guard_s * fs or abs(center - off) < interictal_guard_s * fs:
            return "excluded"
    return "interictal"


def windows_from_recording(
    data: np.ndarray, fs: float, seizures: list[tuple[int, int]], channel: int | None = None,
    win_s: float = 480.0, step_s: float = 120.0, target_len: int = 120, **feat_kw,
) -> list[LabeledWindow]:
    """Slide a window over one channel, label each, and compute its detector-ready slow feature."""
    data = np.asarray(data, dtype=float)
    if data.ndim == 2:
        if channel is None:
            from bsf.dynamotype.labeling import select_seizure_channel

            channel = (
                select_seizure_channel(data, seizures[0][0], seizures[0][1]) if seizures else 0
            )
        x = data[:, channel]
    else:
        x = data
    w, step = int(win_s * fs), int(step_s * fs)
    out: list[LabeledWindow] = []
    for start in range(0, len(x) - w + 1, step):
        center = start + w // 2
        label = label_window(center, fs, seizures)
        feat = slow_feature(x[start : start + w], fs, target_len=target_len, **feat_kw)
        out.append(LabeledWindow(start=start, label=label, feature=feat))
    return out


def collect_swec_long_windows(root: str, **kw) -> list[LabeledWindow]:
    """Build labelled windows from the SWEC long-term ID18 segments (preictal + true interictal)."""
    from pathlib import Path

    from bsf.data.swec_ethz import load_long

    windows: list[LabeledWindow] = []
    for rec in load_long(Path(root) / "longterm"):
        windows.extend(windows_from_recording(rec.data, rec.fs, rec.seizures, **kw))
    return windows
