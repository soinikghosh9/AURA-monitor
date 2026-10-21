#!/usr/bin/env python
"""Duplicate animals in the iEEG development set, and what they did to the development comparison.

check_canine_identity.py found exact shared sample sequences between ieeg.org and Kaggle dogs: Dog_A0002b with
Kaggle Dog_3, and Dog_A0003 with Kaggle Dog_2 and Dog_5. The benchmark grouped folds by dataset label, so these
recordings sat in different folds, and every fold's test dogs had a duplicate in training. Two analyses:

  1. existing out-of-fold predictions, split into test dogs with and without a duplicate in their training folds:
     the AUC gap estimates how much each model gained from the leak (no retraining)
  2. the 18 AURA features re-run with folds grouped by animal (exact-evidence merge, and a conservative merge that
     also joins Dog_A0002a with Dog_A0002b), against the original folds (CPU only)

  python scripts/run_ieeg_dedup.py   -> results/ieeg_dedup_<stamp>/results.json
"""
from __future__ import annotations

import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

EXACT = {"canine_ieeg_Dog_A0002b": "animal_X", "kaggle_ieeg_Dog_3": "animal_X",
         "canine_ieeg_Dog_A0003": "animal_Y", "kaggle_ieeg_Dog_2": "animal_Y", "kaggle_ieeg_Dog_5": "animal_Y"}
CONSERVATIVE = {**EXACT, "canine_ieeg_Dog_A0002a": "animal_X"}


def folds_of(y, g, seed=1234, k=3):
    return list(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, g))


def main() -> int:
    import os
    os.chdir(ROOT)
    out = ROOT / "results" / f"ieeg_dedup_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out.mkdir(parents=True)
    f = sorted(glob.glob(str(ROOT / "results/_oof/unified_ieeg_*.npz")))[-1]
    z = np.load(f, allow_pickle=True)
    y, g = z["_y"].astype(int), z["_groups"].astype(str)
    folds = folds_of(y, g)
    R: dict = {"oof_file": Path(f).name, "fold_of": {}, "leak_split": {}, "aura_regrouped": {}}

    # 1. which test dogs had a duplicate in their training folds
    fold_of = {s: i for i, (_tr, te) in enumerate(folds) for s in np.unique(g[te])}
    R["fold_of"] = fold_of
    leaked = set()
    for s, i in fold_of.items():
        a = EXACT.get(s)
        if a and any(EXACT.get(o) == a and fold_of[o] != i for o in fold_of):
            leaked.add(s)
    clean = sorted(set(fold_of) - leaked)
    R["leaked_dogs"], R["clean_dogs"] = sorted(leaked), clean
    mc, ml = np.isin(g, clean), np.isin(g, sorted(leaked))
    zs = np.load(ROOT / "results/_oof_variants/featsub_ieeg.npz")      # the paper's reference: features on network input
    assert (zs["_groups"].astype(str) == g).all()
    preds = {**{k: z[k] for k in z.files if not k.startswith("_")},
             **{k.split("@")[0]: zs[k] for k in zs.files if not k.startswith("_")}}
    for m, p in preds.items():
        if p.shape != y.shape or not np.isfinite(p).all():
            continue
        R["leak_split"][m] = {"auc_all": round(float(roc_auc_score(y, p)), 4),
                              "auc_clean_dogs": round(float(roc_auc_score(y[mc], p[mc])), 4),
                              "auc_leaked_dogs": round(float(roc_auc_score(y[ml], p[ml])), 4)}
    print("clean:", clean, "| leaked:", sorted(leaked), flush=True)
    for m, v in sorted(R["leak_split"].items(), key=lambda kv: -kv[1]["auc_all"])[:12]:
        print(f"  {m:24s} all {v['auc_all']:.3f}  clean {v['auc_clean_dogs']:.3f}  leaked {v['auc_leaked_dogs']:.3f}", flush=True)

    # 2. AURA features with folds grouped by animal
    from run_feature_fairness import _all_subwindows
    from run_unified_benchmark import _load_ieeg
    W, y2, g2, Xclip = _load_ieeg(60, 6)
    g2 = np.asarray(g2).astype(str)
    assert (y2 == y).all() and (g2 == g).all(), "clip order differs from the benchmark OOF file"
    Xsub = _all_subwindows(W, 400.0, 12)
    models = {"feat_logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
              "feat_gbm": GradientBoostingClassifier(n_estimators=200, random_state=0)}
    for gname, gmap in (("original", {}), ("animal_exact", EXACT), ("animal_conservative", CONSERVATIVE)):
        ga = np.array([gmap.get(s, s) for s in g])
        R["aura_regrouped"][gname] = {"n_groups": int(len(np.unique(ga)))}
        for seed in (1234, 1, 2, 3, 4):
            fl = folds_of(y, ga, seed=seed)
            for xname, X in (("", Xclip), ("_sub", Xsub)):
                for mname, clf in models.items():
                    oof = np.zeros(len(y))
                    for tr, te in fl:
                        oof[te] = clone(clf).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
                    R["aura_regrouped"][gname].setdefault(mname + xname, []).append(round(float(roc_auc_score(y, oof)), 4))
        print(gname, {k: (np.mean(v).round(3) if isinstance(v, list) else v) for k, v in R["aura_regrouped"][gname].items()}, flush=True)
    (out / "results.json").write_text(json.dumps(R, indent=1))
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
