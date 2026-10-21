#!/usr/bin/env python
"""How much of the EEG representation is the patient, and how much is the clinical label?

A mechanistic account of why cross-patient transfer is hard and why no architecture resolved it. For
each task, each feature of the 18-D analytic vector is decomposed (after global standardisation) into

  patient share   variance of patient means / total variance        (identity)
  label share     within-patient variance of class means / total     (the clinical signal)

and, as a whole-vector check, patient identity is decoded from the vector with a clip-level
cross-validated multinomial logistic regression (accuracy against the chance rate of the largest
patient). Both inputs are reported: the vector on the first-run input (whole clip, unnormalised on
iEEG) and on the networks' own standardised sub-windows, to test whether the per-window
standardisation the networks receive reduces the patient component -- the explanation offered in the
manuscript for why the matched-input vector transfers better.

-> results/identity_decomp_<stamp>/results.json
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))


def shares(X, y, g):
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    tot = (Z ** 2).mean(0)
    subs = np.unique(g)
    pm = np.stack([Z[g == s].mean(0) for s in subs])
    w = np.array([(g == s).mean() for s in subs])
    pat = (w[:, None] * pm ** 2).sum(0) / tot
    lab = np.zeros(Z.shape[1]); wsum = 0.0
    for s in subs:
        m = g == s
        if len(np.unique(y[m])) < 2:
            continue
        Zs = Z[m] - Z[m].mean(0)
        cm = np.stack([Zs[y[m] == c].mean(0) for c in (0, 1)])
        pc = np.array([(y[m] == c).mean() for c in (0, 1)])
        lab += m.mean() * (pc[:, None] * cm ** 2).sum(0)
        wsum += m.mean()
    lab = lab / max(wsum, 1e-9) / tot
    return pat, lab


def identity_accuracy(X, g):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    labs, gi = np.unique(g, return_inverse=True)
    n_split = min(5, int(np.bincount(gi).min()))
    if n_split < 2:
        return None, None
    pred = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
                             X, gi, cv=StratifiedKFold(n_split, shuffle=True, random_state=0))
    return float((pred == gi).mean()), float(np.bincount(gi).max() / len(gi))


def main() -> int:
    from bench_data import load_task
    from run_feature_fairness import _all_subwindows
    from bsf.utils.io import save_json
    from bsf.utils.provenance import new_results_dir
    out = new_results_dir("identity_decomp", seed=0, config={})
    R = {}
    for task in ("typing", "ieeg", "scalp", "detect"):
        W, y, g, ev, X = load_task(task, "dev")
        inputs = {"first_run": X}
        if task != "typing":
            inputs["matched"] = _all_subwindows(W, {"ieeg": 400.0}.get(task, 250.0), 6)
        del W
        R[task] = {}
        for name, F in inputs.items():
            F = np.nan_to_num(F)
            pat, lab = shares(F, y, g)
            acc, chance = identity_accuracy(F, g)
            R[task][name] = {"patient_share_median": round(float(np.median(pat)), 4),
                             "label_share_median": round(float(np.median(lab)), 4),
                             "ratio_median": round(float(np.median(pat / np.maximum(lab, 1e-6))), 2),
                             "identity_acc": None if acc is None else round(acc, 4),
                             "identity_chance": None if chance is None else round(chance, 4),
                             "n_patients": int(len(np.unique(g))),
                             "per_feature_patient": [round(float(v), 4) for v in pat],
                             "per_feature_label": [round(float(v), 4) for v in lab]}
            r = R[task][name]
            print(f"[{task}] {name:9s} patient share {r['patient_share_median']:.3f}  label share "
                  f"{r['label_share_median']:.3f}  (x{r['ratio_median']})  identity decoded "
                  f"{r['identity_acc']} vs chance {r['identity_chance']} ({r['n_patients']} pts)", flush=True)
    save_json(R, out / "results.json")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
