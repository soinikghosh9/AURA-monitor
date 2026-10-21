#!/usr/bin/env python
"""Pairwise architecture inference for the whole benchmark family, with seeds, from stored OOF.

Supersedes run_paired_stats.py for the manuscript, which (a) compared only each task's top six models
against the best analytic arm while the text generalised to all twenty-three, (b) predated the
repaired AURA-Net, and (c) treated each model's single training run as the model -- although this
project has measured run-to-run spreads as large as the differences being claimed.

For every task and every model, against the task's best analytic arm on identical clips and folds:

  * the paired AUC difference, averaged over the model's available seeds;
  * a HIERARCHICAL bootstrap CI: resample subjects (the unit of generalisation), then, per draw,
    resample the model's seeds, so seed-to-seed variance is inside the interval rather than ignored;
  * two-sided p with Benjamini-Hochberg across all models within the task, and the one-sided p for
    "the deep model beats the analytic arm";
  * the seed spread (SD of per-seed AUC) next to each model's effect.

It also reports, per task, the largest advantage over the analytic arm that the data leave open: the
maximum over models of the upper 95% bound. That is the quantity a reader needs to bound "no model
beats eighteen features" by, since failing to reject is not evidence of equivalence.

Sources (all DEV-only, stored predictions; nothing is refitted):
  results/_oof/unified_<task>_*.npz                 benchmark run (the models' first draw)
  results/_oof_ab_<task>_frozen_s{0,1,2}.npy         frozen AURA-Net, three seeds
  results/_oof_variants/variant_<task>_subject_none_*_seeds.npz   seeds 1,2 for each task's top-8
  results/_oof_variants/variant_<task>_subject_none_*_fm.npz      pretrained FMs and controls

-> results/final_stats_<stamp>/results.json + paired.csv + seeds.csv
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

FEATURE_ARMS = ("feat_logistic", "feat_gbm", "feat_logistic_sub", "feat_gbm_sub")
# The reference arm is the analytic vector computed on the SAME sub-windows the deep models see
# (run_feature_fairness.py). The benchmark's own analytic arms saw a different amount of signal: the
# whole 10-min clip on iEEG, only the first 8-s sub-window on scalp. Typing clips are one window.
REFERENCE_ARMS = {"typing": ("feat_logistic", "feat_gbm")}
DEFAULT_REF = ("feat_logistic_sub", "feat_gbm_sub")
DISPLAY = {"auranet_hybrid": "AURA-Net", "auranet_pure": "AURA-Net (no features)",
           "deep_cnn_raw": "repo CNN"}


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)


def gather(task: str) -> tuple[np.ndarray, np.ndarray, dict[str, list[np.ndarray]]]:
    """All stored OOF vectors for one task on the benchmark's clip set, grouped by model."""
    base = sorted(glob.glob(str(REPO / f"results/_oof/unified_{task}_*.npz")))
    if not base:
        raise SystemExit(f"no benchmark OOF for {task}")
    with np.load(base[-1], allow_pickle=False) as z:
        y, g = z["_y"].astype(int), z["_groups"].astype(str)
        arms = {k: [z[k].astype(float)] for k in z.files if not k.startswith("_")}
    # the frozen AURA-Net: seed 0 IS the benchmark's current auranet_hybrid row (verified equal),
    # so the ablation's three seeds replace that single draw rather than being added to it
    fr = [REPO / f"results/_oof_ab_{task}_frozen_s{s}.npy" for s in range(3)]
    if all(f.exists() for f in fr):
        vs = [np.load(f).astype(float) for f in fr]
        if all(len(v) == len(y) for v in vs):
            arms["auranet_hybrid"] = vs
    arms.pop("auranet_anchor", None)          # the anchor ablation arm, not a competitor
    fs = REPO / f"results/_oof_variants/featsub_{task}.npz"
    if fs.exists():
        with np.load(fs, allow_pickle=False) as z:
            if np.array_equal(z["_y"].astype(int), y) and np.array_equal(z["_groups"].astype(str), g):
                for k in z.files:
                    if not k.startswith("_"):
                        arms[k.split("@")[0]] = [z[k].astype(float)]
    for f in sorted(glob.glob(str(REPO / f"results/_oof_variants/variant_{task}_subject_none_e25_*"))):
        if not (f.endswith("_seeds.npz") or f.endswith("_fm.npz")):
            continue
        with np.load(f, allow_pickle=False) as z:
            if len(z["_y"]) != len(y) or not (np.array_equal(z["_y"].astype(int), y)
                                              and np.array_equal(z["_groups"].astype(str), g)):
                print(f"  [skip] {Path(f).name}: clip set differs from the benchmark's", flush=True)
                continue
            for k in z.files:
                if k.startswith("_"):
                    continue
                name = k.split("@")[0]
                arms.setdefault(name, []).append(z[k].astype(float))
    return y, g, arms


def hierarchical_contrast(y, g, pa_list, pb, n_boot=2000, seed=0):
    """AUC(a) - AUC(b), a averaged over its seeds; bootstrap subjects, then seeds within each draw."""
    rng = np.random.default_rng(seed)
    subs = np.unique(g)
    idx_by = {s: np.where(g == s)[0] for s in subs}
    obs = float(np.mean([_auc(y, pa) for pa in pa_list]) - _auc(y, pb))
    diffs = []
    k = len(pa_list)
    for _ in range(n_boot):
        draw = rng.choice(subs, len(subs), replace=True)
        idx = np.concatenate([idx_by[s] for s in draw])
        if len(np.unique(y[idx])) < 2:
            continue
        seeds = rng.choice(k, k, replace=True) if k > 1 else [0]
        da = np.mean([_auc(y[idx], pa_list[j][idx]) for j in seeds])
        diffs.append(da - _auc(y[idx], pb[idx]))
    d = np.asarray(diffs)
    p2 = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    p_gt = float((d <= 0).mean())
    wins = 0; n = 0; per = []
    for s in subs:
        m = g == s
        if len(np.unique(y[m])) == 2:
            n += 1
            dd = np.mean([_auc(y[m], pa[m]) for pa in pa_list]) - _auc(y[m], pb[m])
            per.append(dd); wins += int(dd > 0)
    return {"delta": round(obs, 4), "lo": round(float(np.percentile(d, 2.5)), 4),
            "hi": round(float(np.percentile(d, 97.5)), 4), "p_two_sided": round(p2, 4),
            "p_model_better": round(p_gt, 4), "subjects_model_wins": wins, "subjects_scored": n,
            "p_signflip": signflip(per)}


def signflip(per, n_max=200000, seed=0):
    """Exact (or Monte-Carlo) paired sign-flip test on per-patient AUC differences.

    The subject bootstrap is the primary test, but a percentile bootstrap over eight clusters
    under-covers. The sign-flip test asks only whether the per-patient differences are symmetric about
    zero, is exact for n <= 17, and is the conservative check a small-cohort contrast needs.
    """
    per = np.asarray([v for v in per if np.isfinite(v)], float)
    n = len(per)
    if n < 2:
        return None
    obs = abs(per.mean())
    if n <= 17:
        signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    else:
        signs = np.random.default_rng(seed).choice([-1, 1], size=(n_max, n))
    null = np.abs((signs * per).mean(1))
    return round(float((null >= obs - 1e-12).mean()), 4)


def _bh(p):
    p = np.asarray(p, float); n = len(p)
    o = np.argsort(p); adj = np.empty(n); prev = 1.0
    for rank, i in enumerate(reversed(o), start=1):
        prev = min(prev, p[i] * n / (n - rank + 1)); adj[i] = prev
    return np.clip(adj, 0, 1)


def main() -> int:
    from bsf.utils.io import save_json
    from bsf.utils.provenance import new_results_dir
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default="typing,ieeg,scalp,detect")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    out = new_results_dir("final_stats", seed=0, config=vars(args))
    R, prow, srow = {}, ["task,model,n_seeds,auc_mean,auc_sd,delta,lo,hi,p_two,p_bh,p_model_better,"
                         "wins,scored,p_signflip,p_signflip_bh"], ["task,model,seed_aucs"]
    for task in [t for t in args.tasks.split(",") if t]:
        y, g, arms = gather(task)
        auc = {m: [float(_auc(y, p)) for p in ps] for m, ps in arms.items()}
        cands = [m for m in REFERENCE_ARMS.get(task, DEFAULT_REF) if m in auc]
        ref = max(cands, key=lambda m: auc[m][0])
        rows = {}
        for m, ps in arms.items():
            if m == ref:
                continue
            r = hierarchical_contrast(y, g, ps, arms[ref][0], n_boot=args.n_boot)
            r.update(n_seeds=len(ps), auc_mean=round(float(np.mean(auc[m])), 4),
                     auc_sd=round(float(np.std(auc[m], ddof=1)), 4) if len(ps) > 1 else None)
            rows[m] = r
        names = list(rows)
        adj = _bh([rows[m]["p_two_sided"] for m in names])
        for m, a in zip(names, adj):
            rows[m]["p_bh"] = round(float(a), 4)
        adj_sf = _bh([rows[m]["p_signflip"] if rows[m]["p_signflip"] is not None else 1.0
                      for m in names])
        for m, a in zip(names, adj_sf):
            rows[m]["p_signflip_bh"] = round(float(a), 4)
        deep = [m for m in names if m not in FEATURE_ARMS]
        best_upper = max(deep, key=lambda m: rows[m]["hi"])
        # a contrast is called only when the subject bootstrap AND the exact sign-flip test agree
        # after FDR (typing: most patients carry one class, so no per-patient AUC -> bootstrap only)
        def _sig(m):
            if rows[m]["p_bh"] >= 0.05:
                return False
            return task == "typing" or rows[m]["p_signflip_bh"] < 0.05
        sig_better = [m for m in deep if _sig(m) and rows[m]["delta"] > 0]
        sig_worse = [m for m in deep if _sig(m) and rows[m]["delta"] < 0]
        means = {m: float(np.mean(v)) for m, v in auc.items()}
        dm = sorted(means.values())
        seeded = {m: v for m, v in auc.items() if len(v) > 1}
        R[task] = {"reference_arm": ref, "reference_auc": round(auc[ref][0], 4),
                   "n_subjects": int(len(np.unique(g))), "n_clips": int(len(y)),
                   "n_models": len(arms), "contrasts": rows,
                   "sig_better_than_features": sig_better, "sig_worse_than_features": sig_worse,
                   "largest_open_advantage": {"model": best_upper, "upper95": rows[best_upper]["hi"],
                                              "delta": rows[best_upper]["delta"]},
                   "auc_range": [round(dm[0], 4), round(dm[-1], 4)],
                   "auc_sd_across_models": round(float(np.std(dm)), 4),
                   "median_seed_sd": (round(float(np.median([np.std(v, ddof=1) for v in seeded.values()])), 4)
                                      if seeded else None),
                   "seed_aucs": seeded}
        for m in sorted(rows, key=lambda k: -rows[k]["delta"]):
            r = rows[m]
            prow.append(f"{task},{m},{r['n_seeds']},{r['auc_mean']},{r['auc_sd']},{r['delta']},"
                        f"{r['lo']},{r['hi']},{r['p_two_sided']},{r['p_bh']},{r['p_model_better']},"
                        f"{r['subjects_model_wins']},{r['subjects_scored']},{r['p_signflip']},"
                        f"{r['p_signflip_bh']}")
        for m, v in seeded.items():
            srow.append(f"{task},{m},{' '.join(f'{a:.4f}' for a in v)}")
        print(f"[{task}] ref {ref} {auc[ref][0]:.3f} | {len(arms)} models | better(BH): {sig_better} "
              f"| worse(BH): {len(sig_worse)} | largest open advantage {best_upper} "
              f"{rows[best_upper]['delta']:+.3f} [..{rows[best_upper]['hi']:+.3f}] | "
              f"median seed SD {R[task]['median_seed_sd']}", flush=True)
    save_json(R, out / "results.json")
    (out / "paired.csv").write_text("\n".join(prow))
    (out / "seeds.csv").write_text("\n".join(srow))
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
