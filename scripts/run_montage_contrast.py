#!/usr/bin/env python
"""Common-montage fairness check: each network against the physiological reference on identical clips.

The double-banana runs (run_benchmark_variants.py --montage db18) store every network's out-of-fold
predictions. Here the reference is fitted on the same clips and folds with its matched input (the 18
features of the standardized sub-windows the networks see, averaged over the clip; boosted and logistic),
and every network is contrasted with the better reference classifier: paired AUC difference with a
hierarchical bootstrap over patients (and seeds, when several were run; the contrast uses their mean
prediction) and the per-patient sign-flip test, Benjamini-Hochberg across networks.

-> results/montage_contrast_<stamp>/results.json
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


def main() -> int:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from bench_data import SFREQ, load_task
    from bsf.utils.io import save_json
    from bsf.utils.provenance import new_results_dir
    from run_benchmark_variants import _folds
    from run_final_stats import hierarchical_contrast, signflip
    from run_streaming_monitor import _train_window_features
    from run_variant_stats import _per_subject_delta

    out = new_results_dir("montage_contrast", seed=1234, config={"montage": "db18"})
    R = {"tasks": {}}
    for task in ("detect", "scalp"):
        oof = sorted((REPO / "results" / "_oof_variants").glob(f"variant_{task}_subject_none_e25_s3_n*_montage_db18.npz"))
        if not oof:
            continue
        W, y, g, ev, _ = load_task(task, "dev", montage="db18")
        n_t = W.shape[-1]
        F = _train_window_features(W, SFREQ[task]).mean(1)
        folds = _folds("subject", y, g, ev, 3)
        ref = {}
        for name, mk in (("feat_gbm", lambda: GradientBoostingClassifier(n_estimators=200, random_state=0)),
                         ("feat_logistic", lambda: make_pipeline(StandardScaler(), LogisticRegression(
                             max_iter=2000, class_weight="balanced")))):
            p = np.zeros(len(y))
            for tr, te in folds:
                p[te] = mk().fit(F[tr], y[tr]).predict_proba(F[te])[:, 1]
            ref[name] = p
        rname = max(ref, key=lambda k: roc_auc_score(y, ref[k]))
        res = {"reference": rname, "reference_auc": round(float(roc_auc_score(y, ref[rname])), 4),
               "reference_aucs": {k: round(float(roc_auc_score(y, v)), 4) for k, v in ref.items()},
               "n_clips": int(len(y)), "n_patients": int(len(np.unique(g))), "n_times": int(n_t), "networks": {}}
        with np.load(oof[-1]) as z:
            by_model: dict[str, list] = {}
            for key in z.files:
                m = key.split("@")[0]
                if m.startswith(("feat_", "_")) or len(z[key]) != len(y):
                    continue
                by_model.setdefault(m, []).append((key, z[key].astype(float)))
        for m, runs in by_model.items():
            p = np.mean([v for _k, v in runs], 0)          # seed-averaged prediction
            hc = hierarchical_contrast(y, g, [v for _k, v in runs], ref[rname], n_boot=2000)
            seeds = {k.split("@")[1]: round(float(roc_auc_score(y, v)), 4) for k, v in runs}
            per_seed = {k.split("@")[1]: round(float(roc_auc_score(y, v) - roc_auc_score(y, ref[rname])), 4) for k, v in runs}
            res["networks"][m] = {"auc": round(float(roc_auc_score(y, p)), 4), "delta": round(hc["delta"], 4),
                                  "lo": round(hc["lo"], 4), "hi": round(hc["hi"], 4), "p_boot": hc["p_two_sided"],
                                  "p_signflip": signflip(_per_subject_delta(y, g, p, ref[rname])),
                                  "n_seeds": len(runs), "seed_auc": seeds, "seed_delta": per_seed,
                                  "seeds_positive": int(sum(v > 0 for v in per_seed.values()))}
        # Benjamini-Hochberg over networks (bootstrap and sign-flip must both pass)
        names = list(res["networks"])
        for test in ("p_boot", "p_signflip"):
            ps = np.array([res["networks"][m][test] or 1.0 for m in names])
            order = np.argsort(ps); q = np.empty_like(ps)
            q[order] = np.minimum.accumulate((ps[order] * len(ps) / (np.arange(len(ps)) + 1))[::-1])[::-1]
            for m, v in zip(names, q):
                res["networks"][m][test + "_bh"] = round(float(min(v, 1.0)), 4)
        for m in names:
            v = res["networks"][m]
            v["significantly_better"] = bool(v["delta"] > 0 and v["p_boot_bh"] < 0.05 and v["p_signflip_bh"] < 0.05)
        R["tasks"][task] = res
        print(f"[{task}] reference {rname} {res['reference_auc']:.3f}; " + "; ".join(
            f"{m} {v['auc']:.3f} ({v['delta']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}])" for m, v in res["networks"].items()),
            flush=True)
    save_json(R, out / "results.json")
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
