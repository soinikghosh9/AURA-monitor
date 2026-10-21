#!/usr/bin/env python
"""Score the analytic arms on exactly the input the deep models receive.

In the benchmark the deep models see six 8-s sub-windows of each clip (48 s), while the analytic arms
were given the 18-D vector computed over the WHOLE clip -- 600 s for a Kaggle iEEG clip, 60 s for a
scalp clip. That asymmetry favours the analytic arms, so "no architecture beats eighteen features"
could partly be "no architecture beats more data". This refits both analytic arms on the 18-D vector
averaged over exactly the sub-windows the deep models see (the cache's ``Xsub``), on the benchmark's
own folds, and stores the out-of-fold predictions beside the benchmark's so run_final_stats.py can use
them as the reference arm instead. Typing clips are one 4-s window, identical for both, and are left
out. Deterministic, CPU, minutes.

-> results/feature_fairness_<stamp>/results.json ; OOF -> results/_oof_variants/featsub_<task>.npz
"""

from __future__ import annotations

import argparse
import glob
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _all_subwindows(W: np.ndarray, fs: float, n_jobs: int) -> np.ndarray:
    """18-D vector per sub-window, averaged over the clip's sub-windows; cached, parallel."""
    import hashlib

    from joblib import Parallel, delayed

    from bsf.data.features import rich_plus
    key = hashlib.sha1(np.ascontiguousarray(W[:, :, 0, :64]).tobytes()).hexdigest()[:16]
    f = Path(f"results/_clip_cache/featsub6_{key}.npy")
    if f.exists():
        return np.load(f)
    n, k = W.shape[:2]
    flat = Parallel(n_jobs=n_jobs, batch_size=32)(
        delayed(rich_plus)(W[i, j].T, fs) for i in range(n) for j in range(k))
    X = np.nan_to_num(np.asarray(flat, np.float32)).reshape(n, k, -1).mean(1).astype(np.float32)
    np.save(f, X)
    return X


def main() -> int:
    from sklearn.base import clone
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from bench_data import load_task
    from bsf.utils.io import save_json
    from bsf.utils.provenance import new_results_dir
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default="ieeg,scalp,detect")
    ap.add_argument("--n-jobs", type=int, default=6)
    args = ap.parse_args()
    out = new_results_dir("feature_fairness", seed=1234, config=vars(args))
    clfs = {"feat_logistic": make_pipeline(StandardScaler(), LogisticRegression(
                max_iter=2000, class_weight="balanced")),
            "feat_gbm": GradientBoostingClassifier(n_estimators=200, random_state=0)}
    R = {}
    for task in [t for t in args.tasks.split(",") if t]:
        W, y, g, ev, Xsub = load_task(task, "dev", feat_key="Xsub")
        # The matched arm is defined identically for every task: the 18-D vector computed on EXACTLY
        # the arrays the networks receive (the pooled, standardised sub-windows), averaged over the
        # clip's sub-windows. The caches' own 'Xsub' does not meet that definition everywhere: on
        # scalp it is the first 8-s sub-window only (the analytic arms saw 8 s against the networks'
        # 48 s), and on iEEG it was computed from raw, unnormalised segments (absolute power the
        # networks never see). For detection a clip IS one 8-s window, repeated as its sub-windows.
        Xsub = _all_subwindows(W, {"ieeg": 400.0}.get(task, 250.0), args.n_jobs)
        del W
        bench = sorted(glob.glob(f"results/_oof/unified_{task}_*.npz"))[-1]
        with np.load(bench) as z:
            assert np.array_equal(z["_y"].astype(int), y) and np.array_equal(z["_groups"].astype(str), g), \
                "clip order differs from the benchmark's"
            ref = {k: z[k] for k in ("feat_logistic", "feat_gbm")}
        folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=1234).split(
            np.zeros(len(y)), y, g))
        store = {"_y": y.astype(np.int8), "_groups": g}
        R[task] = {}
        for m, clf in clfs.items():
            oof = np.zeros(len(y))
            for tr, te in folds:
                oof[te] = clone(clf).fit(Xsub[tr], y[tr]).predict_proba(Xsub[te])[:, 1]
            store[f"{m}_sub@s0"] = oof.astype(np.float32)
            R[task][m] = {"whole_clip": round(float(roc_auc_score(y, ref[m])), 4),
                          "same_subwindows": round(float(roc_auc_score(y, oof)), 4)}
            print(f"[{task}] {m:14s} whole clip {R[task][m]['whole_clip']:.3f} -> "
                  f"same sub-windows {R[task][m]['same_subwindows']:.3f}", flush=True)
        Path("results/_oof_variants").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f"results/_oof_variants/featsub_{task}.npz", **store)
    save_json(R, out / "results.json")
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
