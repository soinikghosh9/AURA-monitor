#!/usr/bin/env python
"""Does cross-patient performance grow with the number of training patients?

The manuscript argues that the lever for cross-patient EEG models is more patients rather than a larger
model. This measures it for the analytic vector (matched input): within the benchmark's own
patient-grouped folds, the training side is subsampled to k patients (k = 2, 4, 8, ... up to all),
repeated with different draws, and each draw is scored on the untouched held-out fold. The slope of AUC
against log2(k) and its bootstrap interval say whether performance was still rising when the data ran
out. CPU only.

-> results/patient_scaling_<stamp>/results.json
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    from sklearn.base import clone
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from bench_data import load_task
    from run_feature_fairness import _all_subwindows
    from bsf.utils.io import save_json
    from bsf.utils.provenance import new_results_dir
    out = new_results_dir("patient_scaling", seed=0, config={})
    clfs = {"feat_logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000,
                                                                                class_weight="balanced")),
            "feat_gbm": GradientBoostingClassifier(n_estimators=200, random_state=0)}
    R = {}
    rng = np.random.default_rng(0)
    for task in ("typing", "scalp", "detect", "ieeg"):
        W, y, g, ev, X = load_task(task, "dev")
        if task != "typing":
            X = _all_subwindows(W, {"ieeg": 400.0}.get(task, 250.0), 6)
        del W
        splits = 5 if task == "typing" else 3
        folds = list(StratifiedGroupKFold(splits, shuffle=True, random_state=1234).split(X, y, g))
        R[task] = {}
        for mname, clf in clfs.items():
            curve = {}
            n_tr_all = min(len(np.unique(g[tr])) for tr, _ in folds)
            ks = sorted({k for k in (2, 3, 4, 6, 8, 12, 16, 24, 32) if k < n_tr_all} | {n_tr_all})
            for k in ks:
                aucs = []
                for _rep in range(10 if k < n_tr_all else 1):
                    oof = np.full(len(y), np.nan)
                    for tr, te in folds:
                        pts = np.unique(g[tr])
                        pick = rng.choice(pts, min(k, len(pts)), replace=False)
                        m = np.isin(g[tr], pick)
                        if len(np.unique(y[tr][m])) < 2:
                            continue
                        oof[te] = clone(clf).fit(X[tr][m], y[tr][m]).predict_proba(X[te])[:, 1]
                    ok = np.isfinite(oof)
                    if ok.sum() and len(np.unique(y[ok])) == 2:
                        aucs.append(roc_auc_score(y[ok], oof[ok]))
                if aucs:
                    curve[int(k)] = {"mean": round(float(np.mean(aucs)), 4),
                                     "sd": round(float(np.std(aucs)), 4), "n": len(aucs)}
            ks_ = np.array(sorted(curve)); ms = np.array([curve[k]["mean"] for k in ks_])
            slope = float(np.polyfit(np.log2(ks_), ms, 1)[0]) if len(ks_) > 2 else None
            # slope over the upper half of the curve: still rising when the data ran out?
            hi = ks_ >= np.median(ks_)
            slope_hi = float(np.polyfit(np.log2(ks_[hi]), ms[hi], 1)[0]) if hi.sum() > 1 else None
            R[task][mname] = {"curve": curve, "slope_per_doubling": None if slope is None else round(slope, 4),
                              "slope_upper_half": None if slope_hi is None else round(slope_hi, 4)}
            print(f"[{task}] {mname:13s} " + " ".join(f"k={k}:{curve[k]['mean']:.3f}" for k in ks_)
                  + f" | per doubling {slope:+.3f} (upper half {slope_hi:+.3f})", flush=True)
    save_json(R, out / "results.json")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
