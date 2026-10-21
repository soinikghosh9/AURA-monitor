"""Short-horizon seizure likelihood a clinician can read as a probability.

The quantity is P(a seizure starts within the next ``horizon`` minutes | everything known now), updated
every 10 min. Its inputs are causal:

  history     seizures already observed (counts in the last 1, 6 and 24 h, log time since the last one):
              seizures cluster, and the strongest short-term predictor of a seizure is a recent one
  own rate    the patient's own seizure rate so far, shrunk to the population rate (Gamma-Poisson with
              the population as prior), so it starts at the population value and becomes personal
  time of day sine and cosine of the clock phase
  IED burden  optional: the patient-referenced IED rate over the last 30 min (``ied_rate`` column)

A logistic model on these is fitted across training patients and evaluated leave-one-patient-out by the
caller; the output is calibrated by construction (logistic link) and checked with reliability bins.
Seizures are the annotated ones in evaluation, and the confirmed ones in deployment.
"""
from __future__ import annotations

import numpy as np

BIN_S = 600.0
COLS = ["n_1h", "n_6h", "n_24h", "log_since_last", "own_rate_log", "tod_sin", "tod_cos"]


def coverage_bins(recs, bin_s: float = BIN_S) -> np.ndarray:
    """Start times of bins at least half covered by recordings."""
    t0 = min(r.t0 for r in recs); t1 = max(r.t0 + r.dur for r in recs)
    edges = np.arange(np.floor(t0 / bin_s) * bin_s, t1, bin_s)
    cov = np.zeros(len(edges))
    for r in recs:
        a, b = r.t0, r.t0 + r.dur
        cov += np.clip(np.minimum(edges + bin_s, b) - np.maximum(edges, a), 0, None)
    return edges[cov >= 0.5 * bin_s]


def features(tb: np.ndarray, onsets: np.ndarray, t_start: float, prior_rate_per_h: float,
             prior_strength_h: float = 24.0, observed_h: np.ndarray | None = None) -> np.ndarray:
    """Causal features at bin starts ``tb`` given seizure ``onsets`` (only those before each tb count).
    ``observed_h``: monitored hours before each bin (defaults to wall time since ``t_start``)."""
    on = np.sort(np.asarray(onsets, float))
    F = np.zeros((len(tb), len(COLS)))
    for i, t in enumerate(tb):
        past = on[on < t]
        F[i, 0] = np.sum(past >= t - 3600); F[i, 1] = np.sum(past >= t - 6 * 3600); F[i, 2] = np.sum(past >= t - 86400)
        F[i, 3] = np.log1p((t - past[-1]) / 3600) if len(past) else np.log1p(72.0)
        obs = observed_h[i] if observed_h is not None else (t - t_start) / 3600
        rate = (len(past) + prior_rate_per_h * prior_strength_h) / (obs + prior_strength_h)
        F[i, 4] = np.log(rate + 1e-4)
        ph = 2 * np.pi * ((t % 86400) / 86400)
        F[i, 5] = np.sin(ph); F[i, 6] = np.cos(ph)
    return F


def labels(tb: np.ndarray, onsets: np.ndarray, horizon_s: float, bin_s: float = BIN_S) -> np.ndarray:
    """1 when a seizure starts within (t + bin, t + bin + horizon]: the next horizon after this bin."""
    on = np.asarray(onsets, float)
    return np.array([np.any((on > t + bin_s) & (on <= t + bin_s + horizon_s)) for t in tb], int)


def ictal_mask(tb: np.ndarray, spans, bin_s: float = BIN_S, post_s: float = 1800.0) -> np.ndarray:
    """Bins overlapping a seizure or its 30 min post-ictal period (not scored: the state is known)."""
    return np.array([any(t < b + post_s and t + bin_s > a for a, b in spans) for t in tb], bool)


class RiskModel:
    def __init__(self, C: float = 0.5):
        self.C = C

    def fit(self, F: np.ndarray, y: np.ndarray):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.m = make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=2000)).fit(F, y)
        return self

    def predict(self, F: np.ndarray) -> np.ndarray:
        return self.m.predict_proba(F)[:, 1]

    def explain(self, f: np.ndarray, cols=COLS) -> dict:
        """Contribution of each input to the log-odds (standardized coefficient x standardized value)."""
        sc, lr = self.m[0], self.m[1]
        z = (np.asarray(f, float) - sc.mean_) / sc.scale_
        return {c: round(float(w * v), 3) for c, w, v in zip(cols, lr.coef_.ravel(), z)}


def reliability(p: np.ndarray, y: np.ndarray, n_bins: int = 8) -> list[dict]:
    q = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    out = []
    for a, b in zip(q[:-1], q[1:]):
        m = (p >= a) & (p <= b)
        if m.sum():
            out.append({"p_mean": round(float(p[m].mean()), 4), "observed": round(float(y[m].mean()), 4), "n": int(m.sum())})
    return out
