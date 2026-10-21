#!/usr/bin/env python
"""Does the baseline-referenced gain survive placing the baseline BEFORE the scored data?

The alignment experiment reserves a random third of each patient's negative clips as that patient's
baseline. On the scalp tasks a patient's negatives are consecutive windows of one or a few
recordings, so a random third interleaves the baseline with the very windows it is used to score:
the scored negatives sit minutes from a reference clip, while a deployed monitor's baseline would
precede everything it scores. This reruns the analytic arms of that comparison with the baseline
taken as each patient's EARLIEST negatives in recording order (clip ids are assigned in time order
within a recording, recordings in file order), next to the random draw, each against the unaligned
model on its own identical clips. Analytic arms only (the networks' alignment runs are GPU-days).

-> results/baseline_placement_<stamp>/results.json
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


def early_mask(y, g, ev, frac=1 / 3, min_n=5):
    """Per patient, its earliest third of negative clips (by recording order), as the baseline."""
    m = np.zeros(len(y), bool)
    order = np.array([int(e.split("|")[1]) for e in ev])
    for s in np.unique(g):
        neg = np.where((g == s) & (y == 0))[0]
        k = max(min_n, int(round(frac * len(neg))))
        if len(neg) - k < min_n:
            continue
        m[neg[np.argsort(order[neg], kind="stable")[:k]]] = True
    return m


def main() -> int:
    from sklearn.base import clone
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from bench_data import baseline_mask, feature_align, load_task
    from run_final_stats import hierarchical_contrast, signflip
    from run_variant_stats import _per_subject_delta
    from bsf.utils.io import save_json
    from bsf.utils.provenance import new_results_dir
    out = new_results_dir("baseline_placement", seed=1234, config={})
    clfs = {"feat_logistic": make_pipeline(StandardScaler(), LogisticRegression(
                max_iter=2000, class_weight="balanced")),
            "feat_gbm": GradientBoostingClassifier(n_estimators=200, random_state=0)}
    R = {}
    for task in ("detect", "scalp", "ieeg"):
        W, y0, g0, ev0, X0 = load_task(task, "dev"); del W
        R[task] = {}
        for design, base in (("random", baseline_mask(y0, g0)), ("early", early_mask(y0, g0, ev0))):
            has_ref = {s for s in np.unique(g0) if (base & (g0 == s)).any()}
            keep = ~base & np.isin(g0, list(has_ref))
            ref = {s: X0[(g0 == s) & base] for s in has_ref}
            y, g, X = y0[keep], g0[keep], X0[keep]
            Xz = feature_align(X, g, ref, "z")
            folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=1234).split(X, y, g))
            R[task][design] = {"n_clips": int(keep.sum()), "n_patients": len(has_ref)}
            for m, clf in clfs.items():
                p = {}
                for arm, Xin in (("none", X), ("ea_base", Xz)):
                    o = np.zeros(len(y))
                    for tr, te in folds:
                        o[te] = clone(clf).fit(Xin[tr], y[tr]).predict_proba(Xin[te])[:, 1]
                    p[arm] = o
                r = hierarchical_contrast(y, g, [p["ea_base"]], p["none"], n_boot=2000)
                r["p_signflip"] = signflip(_per_subject_delta(y, g, p["ea_base"], p["none"]))
                r["auc_none"] = round(float(roc_auc_score(y, p["none"])), 4)
                r["auc_aligned"] = round(float(roc_auc_score(y, p["ea_base"])), 4)
                R[task][design][m] = r
                print(f"[{task}] {design:6s} {m:13s} none {r['auc_none']:.3f} -> base "
                      f"{r['auc_aligned']:.3f}  delta {r['delta']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}] "
                      f"p={r['p_two_sided']} sf={r['p_signflip']}", flush=True)
    save_json(R, out / "results.json")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
