"""Forecasting metrics (threshold-free probability scores + warning-budget operating curve).

See ``docs/method_notes/forecasting_metrics_and_chance_model.md``. Probability scores: Brier score,
Brier skill score (reference = climatological base rate ō(1−ō)), Murphy REL/RES/UNC decomposition,
calibration. Ranking: ROC-AUC. Warning-budget axis: sensitivity-vs-time-in-warning operating curve
and its AUC (random forecaster → 0.5, perfect → 1).
"""

from __future__ import annotations

import numpy as np

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # numpy 2.0 renamed trapz


def brier_score(f: np.ndarray, o: np.ndarray) -> float:
    f = np.asarray(f, dtype=float)
    o = np.asarray(o, dtype=float)
    return float(np.mean((f - o) ** 2))


def base_rate(o: np.ndarray) -> float:
    return float(np.mean(np.asarray(o, dtype=float)))


def brier_skill_score(f: np.ndarray, o: np.ndarray, ref_rate: float | None = None) -> float:
    """BSS = 1 − BS/BS_ref with BS_ref = ō(1−ō). ``ref_rate`` overrides ō (e.g. training base rate)."""
    o = np.asarray(o, dtype=float)
    obar = base_rate(o) if ref_rate is None else float(ref_rate)
    bs_ref = obar * (1.0 - obar)
    if bs_ref <= 0:
        return float("nan")
    return float(1.0 - brier_score(f, o) / bs_ref)


def brier_decomposition(f: np.ndarray, o: np.ndarray, n_bins: int = 20) -> dict[str, float]:
    """Murphy decomposition BS = REL − RES + UNC (binning-limited)."""
    f = np.asarray(f, dtype=float)
    o = np.asarray(o, dtype=float)
    n = len(o)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(f, edges) - 1, 0, n_bins - 1)
    obar = o.mean()
    rel = res = 0.0
    for k in range(n_bins):
        m = idx == k
        nk = int(m.sum())
        if nk == 0:
            continue
        rel += nk * (f[m].mean() - o[m].mean()) ** 2
        res += nk * (o[m].mean() - obar) ** 2
    return {"REL": rel / n, "RES": res / n, "UNC": float(obar * (1 - obar))}


def expected_calibration_error(f: np.ndarray, o: np.ndarray, n_bins: int = 10) -> float:
    """ECE = Σ_k (n_k/N)|f_k − ō_k| (a convention, not from a primary source; REL is the primary scalar)."""
    f = np.asarray(f, dtype=float)
    o = np.asarray(o, dtype=float)
    n = len(o)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(f, edges) - 1, 0, n_bins - 1)
    ece = 0.0
    for k in range(n_bins):
        m = idx == k
        nk = int(m.sum())
        if nk:
            ece += (nk / n) * abs(f[m].mean() - o[m].mean())
    return float(ece)


def reliability_curve(o: np.ndarray, f: np.ndarray, n_bins: int = 10, strategy: str = "quantile"):
    from sklearn.calibration import calibration_curve

    return calibration_curve(np.asarray(o, dtype=int), np.asarray(f, dtype=float),
                             n_bins=n_bins, strategy=strategy)


def roc_auc(o: np.ndarray, f: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    o = np.asarray(o, dtype=int)
    if len(np.unique(o)) < 2:
        return float("nan")
    return float(roc_auc_score(o, np.asarray(f, dtype=float)))


def time_in_warning(warning_mask: np.ndarray) -> float:
    return float(np.mean(np.asarray(warning_mask, dtype=float)))


def operating_curve(o: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sweep the high-risk threshold; return (time-in-warning, sensitivity) points.

    TiW(θ) = fraction of windows warned; S(θ) = fraction of positive windows warned (TPR).
    Runs from (0,0) at θ=∞ to (1,1) at θ=−∞.
    """
    o = np.asarray(o, dtype=int)
    f = np.asarray(f, dtype=float)
    thresholds = np.concatenate([[np.inf], np.unique(f)[::-1]])
    n_pos = max(int((o == 1).sum()), 1)
    tiw = np.empty(len(thresholds))
    sens = np.empty(len(thresholds))
    for i, th in enumerate(thresholds):
        warn = f >= th
        tiw[i] = warn.mean()
        sens[i] = (warn & (o == 1)).sum() / n_pos
    order = np.argsort(tiw)
    return tiw[order], sens[order]


def auc_s_vs_tiw(o: np.ndarray, f: np.ndarray) -> float:
    """Threshold-free skill on the warning-budget axis: ∫ S d(TiW). Random→0.5, perfect→1."""
    tiw, sens = operating_curve(o, f)
    return float(_trapz(sens, tiw))


def sensitivity_at_tiw(o: np.ndarray, f: np.ndarray, budget: float) -> float:
    """Sensitivity when the warning may be active a fraction ``budget`` of the time."""
    tiw, sens = operating_curve(o, f)
    return float(np.interp(budget, tiw, sens))


def event_level_counts(y_true: np.ndarray, warn: np.ndarray, max_gap: int = 1) -> tuple[int, int]:
    """Collapse overlapping/contiguous positive windows into independent seizure EVENTS (E2).

    A single clinical seizure spawns many correlated positive windows (here windows overlap 75%); the
    chance-model Bernoulli N must be the number of INDEPENDENT events, not windows. An event is a
    maximal run of positive (``y_true==1``) windows whose consecutive members are separated by at most
    ``max_gap`` negative windows. Returns ``(n_events, n_events_caught)`` where an event is *caught* if
    ANY of its windows is warned. Feeding window counts to ``above_chance`` instead overstates power by
    1–2 orders of magnitude on clustered real data.
    """
    y = np.asarray(y_true).astype(bool)
    warn = np.asarray(warn).astype(bool)
    idx = np.flatnonzero(y)
    if idx.size == 0:
        return 0, 0
    groups = np.split(idx, np.where(np.diff(idx) > (max_gap + 1))[0] + 1)
    n_events = len(groups)
    caught = sum(1 for g in groups if warn[g].any())
    return int(n_events), int(caught)


def cluster_bootstrap_ci(metric_fn, o: np.ndarray, f: np.ndarray, groups=None,
                         n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI for ``metric_fn(o, f)``, resampling CLUSTERS — not windows (E6/D1).

    ``groups`` assigns each sample to a cluster id (e.g. seizure-event or subject); the bootstrap
    resamples unique clusters with replacement, preserving within-cluster correlation, so the CI is
    not inflated by pseudo-replicated overlapping windows. ``groups=None`` falls back to per-sample
    resampling (valid only for genuinely independent samples). Deterministic given ``seed``. Returns
    ``(point, lo, hi)``.
    """
    o = np.asarray(o)
    f = np.asarray(f)
    point = float(metric_fn(o, f))
    rng = np.random.default_rng(seed)
    groups = np.arange(len(o)) if groups is None else np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_group = [np.flatnonzero(groups == g) for g in uniq]
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([idx_by_group[k] for k in pick])
        val = metric_fn(o[idx], f[idx])
        if np.isfinite(val):
            stats.append(val)
    if not stats:
        return point, float("nan"), float("nan")
    return point, float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1.0 - alpha / 2))
