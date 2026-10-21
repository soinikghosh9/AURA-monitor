"""Clinician-in-the-loop adaptation: confirmed seizures teach a patient head, rejected alarms tune the
threshold.

The monitor starts with the population ensemble and the development-data threshold for the chosen
false-alarm budget. Alarms are reviewed at the end of each hour:

  confirmed   its ictal windows become patient-specific positives; with the windows of rejected alarms
              and a sample of unremarkable windows as negatives, a small logistic head on the patient's
              refreshed AURA features is refitted. Its decision value, standardized on the patient's
              unremarkable windows (median 0, robust scale 1) and clipped to +-3, is ADDED to the ensemble
              logit. Unremarkable windows therefore keep their population score and the threshold keeps
              its meaning; windows resembling the patient's confirmed seizures rise and windows
              resembling rejected alarms fall.
  rejected    its windows become negatives; once at least ``min_hours`` have been reviewed, if the
              false-alarm rate exceeds the budget, the threshold is raised to the lowest value that meets
              it on the reviewed history, but never above the level at which a confirmed seizure of that
              history would have been missed.

A first version averaged the population and head probabilities and raised the threshold from the first
reviewed hour; the average changed the score scale under a fixed threshold and short histories drove the
threshold up (development replay: F1 fell in 5 of 7 streams). Both are corrected here.

``replay`` simulates this sequentially on a scored recording, adjudicating alarms against annotations
(a confirmed alarm overlaps an annotated seizure within the scoring tolerance), and returns the
threshold in force for every window so that scoring uses the decisions the monitor actually made.
"""
from __future__ import annotations

import numpy as np

from bsf.eval import szcore


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Personalizer:
    def __init__(self, threshold: float, budget_per_24h: float, win_s: float = 8.0, seed: int = 0,
                 min_hours: float = 6.0, clip: float = 3.0):
        self.thr0 = self.thr = threshold
        self.budget, self.win_s, self.min_hours, self.clip = budget_per_24h, win_s, min_hours, clip
        self.pos: list[int] = []; self.neg: list[int] = []
        self.head = None; self.d0, self.ds = 0.0, 1.0
        self.rng = np.random.default_rng(seed)
        self.log: list[dict] = []

    def evidence(self, Fz: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Standardized patient-head evidence (0 for every window before the first confirmed seizure)."""
        if self.head is None:
            return np.zeros(len(idx))
        d = self.head.decision_function(np.nan_to_num(Fz[idx]))
        return np.clip((d - self.d0) / self.ds, -self.clip, self.clip)

    def score(self, p: np.ndarray, Fz: np.ndarray, idx: np.ndarray) -> np.ndarray:
        if self.head is None:
            return p[idx]
        return 1.0 / (1.0 + np.exp(-(_logit(p[idx]) + self.evidence(Fz, idx))))

    def refit(self, Fz, upto: int):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        if not self.pos:
            return
        pool = np.setdiff1d(np.arange(upto), np.r_[self.pos, self.neg].astype(int))
        if len(pool) < 20:
            return
        bg = self.rng.choice(pool, min(len(pool), 10 * len(self.pos)), replace=False)
        neg = np.r_[np.asarray(self.neg, int), bg].astype(int)
        idx = np.r_[np.asarray(self.pos, int), neg]
        y = np.r_[np.ones(len(self.pos)), np.zeros(len(neg))]
        self.head = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000)).fit(
            np.nan_to_num(Fz[idx]), y)
        ref = self.rng.choice(pool, min(len(pool), 2000), replace=False)
        d = self.head.decision_function(np.nan_to_num(Fz[ref]))
        self.d0 = float(np.median(d)); self.ds = float(1.4826 * np.median(np.abs(d - self.d0))) or 1.0

    def replay(self, y: np.ndarray, p: np.ndarray, Fz: np.ndarray, t: np.ndarray, review_every_s: float = 3600.0,
               smooth_s: float = 120.0) -> dict:
        """Sequential simulation; returns per-window adapted scores, the threshold in force for each window,
        and the event log."""
        from run_stream_replication import smooth
        n = len(p)
        out = np.zeros(n); thr_t = np.zeros(n)
        start, hours, n_fa = 0, 0.0, 0
        conf: list[tuple[float, float]] = []
        while start < n:
            end = max(int(np.searchsorted(t, t[start] + review_every_s)), start + 1)
            idx = np.arange(start, end)
            out[idx] = self.score(p, Fz, idx); thr_t[idx] = self.thr
            s = smooth(out[:end], t[:end], smooth_s)[start:end]
            hyp = szcore._merge_split(szcore._runs(s >= self.thr, t[idx], self.win_s), szcore.SZCORE["merge_s"], szcore.SZCORE["max_s"])
            ref = szcore._merge_split(szcore._runs(y[:end] > 0, t[:end], self.win_s), szcore.SZCORE["merge_s"], szcore.SZCORE["max_s"])
            ext = [(a - szcore.SZCORE["tol_pre_s"], b + szcore.SZCORE["tol_post_s"]) for a, b in ref]
            for a, b in hyp:
                win = np.flatnonzero((t >= a) & (t < b))
                true = any(a < r1 and r0 < b for r0, r1 in ext)
                if true:
                    self.pos += [i for i in win if y[i] > 0] or list(win); conf.append((a, b))
                else:
                    self.neg += list(win); n_fa += 1
                self.log.append({"t_on": float(a), "t_off": float(b), "confirmed": bool(true), "threshold": round(self.thr, 4)})
            hours += (t[end - 1] - t[start] + self.win_s) / 3600
            if hyp and self.pos:
                self.refit(Fz, end)
            if hours >= self.min_hours and n_fa and 24 * n_fa / hours > self.budget:
                seen = smooth(self.score(p, Fz, np.arange(end)), t[:end], smooth_s)
                cand = szcore.threshold_for_budget([(y[:end], seen, t[:end])], self.budget, self.win_s)
                caps = [seen[(t[:end] >= a) & (t[:end] < b)].max() for a, b in conf if ((t[:end] >= a) & (t[:end] < b)).any()]
                if caps:
                    cand = min(cand, min(caps) - 1e-6)
                self.thr = max(self.thr, cand)
            start = end
        return {"scores": out, "thr_t": thr_t, "events": self.log, "final_threshold": self.thr}

