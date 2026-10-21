"""Classic critical-slowing-down early-warning signals — implemented CAUSALLY.

Indicators (Scheffer 2009; Dakos 2008; ewstools, Bury 2023):
  * rolling **variance** (ddof=1) and rolling **lag-1 autocorrelation** (AR(1) coefficient α),
  * **Kendall τ** trend of each indicator.

Near a fold the recovery rate λ→0, so α = e^{-λΔt} → 1 and Var = σ²/(2λ) → ∞ (rising AR(1)/variance).

Causality (the load-bearing point, see ``docs/method_notes/csd_ews_classic.md``):
the rolling var/AC estimators are already trailing/past-only — *all* leakage lives in the detrend.
``detrend_causal`` is forward-only (EWM / trailing). ``detrend_noncausal`` is the whole-series
Gaussian smoother (``mode="reflect"``) used by ewstools by default; we keep it ONLY as the
leakage-guard positive control, never in a deployed forecast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import kendalltau


def _abs_window(window: float, n: int) -> int:
    w = int(round(window * n)) if 0 < window <= 1 else int(window)
    return max(2, w)


# --------------------------------------------------------------------------- detrending
def detrend_causal(x: np.ndarray, span: float = 0.2, method: str = "ewm") -> np.ndarray:
    """Forward-only detrend: residuals = x − trailing_trend(x≤t). Returns residuals (len == x)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if method == "none":
        return x - np.nanmean(x)
    if method == "ewm":
        halflife = max(1.0, span * n / 2.0)  # causal analogue of a Lowess span
        trend = pd.Series(x).ewm(halflife=halflife, adjust=False).mean().to_numpy()
    elif method == "trailing_mean":
        w = _abs_window(span, n)
        trend = pd.Series(x).rolling(w, min_periods=1).mean().to_numpy()
    else:  # pragma: no cover
        raise ValueError(f"unknown causal detrend method {method!r}")
    return x - trend


def detrend_noncausal(x: np.ndarray, bandwidth: float = 0.2) -> np.ndarray:
    """Whole-series Gaussian smoother (ewstools default), ``mode='reflect'`` — NON-CAUSAL.

    Kept ONLY as the leakage-guard positive control. ``sigma = (0.25/0.675)·bandwidth·N``.
    """
    x = np.asarray(x, dtype=float)
    sigma = (0.25 / 0.675) * bandwidth * len(x)
    trend = ndimage.gaussian_filter(x, sigma=sigma, mode="reflect")
    return x - trend


# --------------------------------------------------------------------------- rolling indicators
def rolling_variance(r: np.ndarray, window: float = 0.25) -> np.ndarray:
    w = _abs_window(window, len(r))
    return pd.Series(np.asarray(r, dtype=float)).rolling(w).var(ddof=1).to_numpy()


def rolling_ac1(r: np.ndarray, window: float = 0.25) -> np.ndarray:
    w = _abs_window(window, len(r))
    s = pd.Series(np.asarray(r, dtype=float))
    return s.rolling(w).apply(lambda a: pd.Series(a).autocorr(lag=1), raw=True).to_numpy()


def kendall_tau(indicator: np.ndarray, time: np.ndarray | None = None) -> float:
    """Scalar Kendall τ of an indicator vs time, over its defined (non-NaN) support."""
    y = np.asarray(indicator, dtype=float)
    t = np.arange(len(y), dtype=float) if time is None else np.asarray(time, dtype=float)
    mask = np.isfinite(y) & np.isfinite(t)
    if mask.sum() < 3:
        return float("nan")
    return float(kendalltau(t[mask], y[mask]).correlation)


# --------------------------------------------------------------------------- top-level
@dataclass
class CSDResult:
    residuals: np.ndarray
    variance: np.ndarray
    ac1: np.ndarray
    ktau_variance: float
    ktau_ac1: float
    window: int
    detrend: str
    causal: bool


def compute_csd(
    x: np.ndarray,
    window: float = 0.25,
    detrend: str = "ewm",
    span: float = 0.2,
    causal: bool = True,
) -> CSDResult:
    """Full CSD computation. ``causal=False`` selects the non-causal detrend (positive control)."""
    x = np.asarray(x, dtype=float)
    r = detrend_causal(x, span=span, method=detrend) if causal else detrend_noncausal(x, bandwidth=span)
    var = rolling_variance(r, window)
    ac = rolling_ac1(r, window)
    return CSDResult(
        residuals=r,
        variance=var,
        ac1=ac,
        ktau_variance=kendall_tau(var),
        ktau_ac1=kendall_tau(ac),
        window=_abs_window(window, len(x)),
        detrend=("noncausal_gaussian" if not causal else detrend),
        causal=causal,
    )


# --- feature functions (x -> full-length indicator) for the leakage guard / forecasting ---
def ac1_feature(x: np.ndarray, window: float = 0.25, span: float = 0.2, causal: bool = True) -> np.ndarray:
    r = detrend_causal(x, span=span) if causal else detrend_noncausal(x, bandwidth=span)
    return rolling_ac1(r, window)


def variance_feature(x: np.ndarray, window: float = 0.25, span: float = 0.2, causal: bool = True) -> np.ndarray:
    r = detrend_causal(x, span=span) if causal else detrend_noncausal(x, bandwidth=span)
    return rolling_variance(r, window)
