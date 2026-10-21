#!/usr/bin/env python
"""Threshold-based clinical metrics from the stored predictions; nothing is refitted.

AUC ranks; a clinician acts on a yes/no at a threshold. This script reports, for every model and task,
sensitivity, specificity, PPV, NPV, F1, balanced accuracy and the Matthews correlation coefficient at
a threshold that never sees the scored patient's labels:

  development   each patient is thresholded at the Youden-optimal cut-point of the out-of-fold
                predictions of the OTHER development patients (leave-one-patient-out cross-fitting);
  test          the arms of the post-seal addendum, whose test predictions are stored, are thresholded
                at the Youden cut-point of the same arm's development out-of-fold predictions, fixed
                before the test predictions are read.

Intervals are percentiles of 1,000 bootstrap draws of patients. Per-patient AUCs (for the
distribution figures) and the confusion counts behind every metric are stored as well.

  python scripts/run_clinical_metrics.py
-> results/clinical_metrics_<stamp>/results.json
"""
from __future__ import annotations

import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
TASKS = ("typing", "ieeg", "scalp", "detect")
METRICS = ("sens", "spec", "ppv", "npv", "f1", "bal_acc", "mcc")


def youden(y, p):
    from sklearn.metrics import roc_curve
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    t = float(thr[int(np.argmax(tpr - fpr))])
    return min(t, 1.0)


def counts(y, yhat):
    tp = int(((yhat == 1) & (y == 1)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum()); fn = int(((yhat == 0) & (y == 1)).sum())
    return tp, fp, tn, fn


def from_counts(tp, fp, tn, fn):
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den > 0 else 0.0
    return {"sens": sens, "spec": spec, "ppv": ppv, "npv": npv, "f1": f1,
            "bal_acc": 0.5 * (sens + spec), "mcc": mcc}


def panel(y, p, yhat, g, n_boot=1000, seed=0):
    """Point metrics, AUC/AUPRC, and patient-bootstrap intervals for every metric."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    tp, fp, tn, fn = counts(y, yhat)
    out = {k: round(float(v), 4) for k, v in from_counts(tp, fp, tn, fn).items()}
    out.update(tp=tp, fp=fp, tn=tn, fn=fn, n=int(len(y)), n_pos=int(y.sum()),
               auc=round(float(roc_auc_score(y, p)), 4), auprc=round(float(average_precision_score(y, p)), 4))
    rng = np.random.default_rng(seed)
    subs = np.unique(g); idx_by = {s: np.where(g == s)[0] for s in subs}
    B = {k: [] for k in (*METRICS, "auc", "auprc")}
    for _ in range(n_boot):
        idx = np.concatenate([idx_by[s] for s in rng.choice(subs, len(subs), replace=True)])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        for k, v in from_counts(*counts(yy, yhat[idx])).items():
            B[k].append(v)
        B["auc"].append(roc_auc_score(yy, p[idx])); B["auprc"].append(average_precision_score(yy, p[idx]))
    out["ci"] = {k: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]
                 for k, v in B.items() if len(v)}
    return out


def crossfit_threshold(y, p, g):
    """Each patient's prediction is thresholded at the Youden cut-point of the other patients."""
    yhat = np.zeros(len(y), int); thr = {}
    for s in np.unique(g):
        m = g == s
        t = youden(y[~m], p[~m]); thr[s] = t
        yhat[m] = (p[m] >= t).astype(int)
    return yhat, thr


def per_patient_auc(y, p, g):
    from sklearn.metrics import roc_auc_score
    return {str(s): round(float(roc_auc_score(y[g == s], p[g == s])), 4)
            for s in np.unique(g) if len(np.unique(y[g == s])) == 2}


def _variant(task, pattern):
    fs = sorted(glob.glob(str(REPO / "results" / "_oof_variants" / f"variant_{task}_{pattern}")))
    return fs[-1] if fs else None


def dev_source(task, arm, arms):
    """(y, p) of an addendum arm on development patients, for its test threshold."""
    from run_final_stats import gather  # noqa: F401  (arms come from gather)
    if arm in arms:
        y, _, A = arms[arm]
        return y, np.mean(A, 0)
    if arm.endswith("_base") or arm.endswith("_ea_base"):
        f = _variant(task, "subject_ea_base_db_e25_*_align.npz")
        name = {"feat_logistic_base": "feat_logistic", "feat_white_base": "feat_white",
                "SCCNet_ea_base": "SCCNet", "deep_cnn_raw_ea_base": "deep_cnn_raw"}[arm]
        with np.load(f) as z:
            return z["_y"].astype(int), z[f"{name}@s0"].astype(float)
    return None


def main() -> int:
    from bsf.utils.io import save_json
    from bsf.utils.provenance import new_results_dir
    from run_final_stats import gather

    out = new_results_dir("clinical_metrics", seed=0, config={"threshold": "youden, cross-fitted"})
    R = {"rule": {"dev": "leave-one-patient-out Youden threshold on other development patients' OOF",
                  "test": "Youden threshold on the arm's development OOF, applied unchanged to test"},
         "dev": {}, "per_patient_auc": {}, "test": {}}
    for task in TASKS:
        y, g, arms = gather(task)
        R["dev"][task] = {}; R["per_patient_auc"][task] = {}
        dev_arms = {}
        for m, A in arms.items():
            p = np.mean(A, 0)
            if not np.isfinite(p).all():
                continue
            yhat, _ = crossfit_threshold(y, p, g)
            R["dev"][task][m] = panel(y, p, yhat, g)
            R["per_patient_auc"][task][m] = per_patient_auc(y, p, g)
            dev_arms[m] = (y, g, A)
        print(f"[{task}] dev: {len(R['dev'][task])} models", flush=True)

        pf = sorted(glob.glob(str(REPO / "results" / "blind_addendum_2*" / f"preds_{task}.npz")))
        if not pf:
            continue
        R["test"][task] = {}
        with np.load(pf[-1], allow_pickle=True) as z:
            yt, gt = z["_y"].astype(int), z["_groups"].astype(str)
            P = {k: z[k].astype(float) for k in z.files if not k.startswith("_")}
        for arm, pt in P.items():
            src = dev_source(task, arm if not (task == "typing" and arm.endswith("_sub")) else arm[:-4],
                             dev_arms)
            if src is None:
                continue
            t = youden(*src)
            r = panel(yt, pt, (pt >= t).astype(int), gt)
            r["threshold"] = round(t, 5)
            R["test"][task][arm] = r
        print(f"[{task}] test: {len(R['test'][task])} arms", flush=True)
    save_json(R, out / "results.json")
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
