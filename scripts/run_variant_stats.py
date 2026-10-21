#!/usr/bin/env python
"""Statistics for the variant program: alignment, pretraining, and the protocol x model factorial.

Reads the stored out-of-fold predictions under results/_oof_variants/ (plus the benchmark's own for
the cross-patient cell of the factorial). Nothing is refitted; partial runs are summarised as far as
they go, so this can be re-run as units finish.

  ALIGN     per task and model: AUC with no alignment, label-free alignment (every clip of the
            patient) and baseline-referenced alignment (reserved negatives), all on identical clips;
            paired subject-bootstrap and exact sign-flip tests for each alignment against none, with
            Benjamini-Hochberg across the task's contrasts.
  PRETRAIN  per task and architecture: published weights vs the recipe-matched random-init control on
            identical 200 Hz input, seed-averaged, hierarchical bootstrap over subjects and seeds; and
            the frozen-encoder probe.
  FACTORIAL per task: AUC for each model under segment-random, event-grouped and subject-grouped
            folds, on the pooled metric; the inflation each protocol buys per model; whether the
            model ranking survives a change of protocol (Spearman); and a two-way decomposition of the
            AUC variance into protocol, model and residual (interaction) sums of squares.

-> results/variant_stats_<stamp>/results.json
"""

from __future__ import annotations

import glob
import re
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from run_final_stats import _auc, _bh, hierarchical_contrast, signflip  # noqa: E402

TASKS = ["typing", "ieeg", "scalp", "detect"]


def _load(path):
    with np.load(path, allow_pickle=False) as z:
        y = z["_y"].astype(int); g = z["_groups"].astype(str)
        arms: dict[str, list[np.ndarray]] = {}
        for k in z.files:
            if not k.startswith("_"):
                arms.setdefault(k.split("@")[0], []).append(z[k].astype(float))
    return y, g, arms


def _find(task, protocol, align, tag, drop=False, early=False):
    pat = (f"variant_{task}_{protocol}_{align}{'_db' if drop else ''}{'_early' if early else ''}"
           f"_e25_*_{tag}.npz")
    fs = sorted(glob.glob(str(REPO / "results/_oof_variants" / pat)))
    fs = [f for f in fs if "smoke" not in f]
    return fs[-1] if fs else None


def _per_subject_delta(y, g, pa, pb):
    out = []
    for s in np.unique(g):
        m = g == s
        if len(np.unique(y[m])) == 2:
            out.append(_auc(y[m], pa[m]) - _auc(y[m], pb[m]))
    return out


def align_stats(task, early=False):
    """``early``: the baseline is each patient's earliest third of negatives (it precedes every scored
    clip) instead of a random third; only the baseline-referenced arm exists for it."""
    files = {a: _find(task, "subject", a, "align", drop=True, early=early)
             for a in ("none", "ea_all", "ea_base")}
    if not files["none"]:
        return None
    base = _load(files["none"])
    y, g = base[0], base[1]
    res = {"files": {k: (Path(v).name if v else None) for k, v in files.items()}, "models": {}}
    contrasts = []
    for a in ("ea_all", "ea_base"):
        if not files[a]:
            continue
        ya, ga, arms_a = _load(files[a])
        if not (np.array_equal(ya, y) and np.array_equal(ga, g)):
            res[f"{a}_error"] = "clip set differs from the unaligned arm"
            continue
        for m, ps in arms_a.items():
            if m not in base[2]:
                continue
            r = hierarchical_contrast(y, g, ps, base[2][m][0], n_boot=2000)
            r["p_signflip"] = signflip(_per_subject_delta(y, g, ps[0], base[2][m][0]))
            r["auc_none"] = round(float(_auc(y, base[2][m][0])), 4)
            r["auc_aligned"] = round(float(_auc(y, ps[0])), 4)
            res["models"].setdefault(m, {})[a] = r
            contrasts.append((m, a))
    if contrasts:
        adj = _bh([res["models"][m][a]["p_two_sided"] for m, a in contrasts])
        for (m, a), v in zip(contrasts, adj):
            res["models"][m][a]["p_bh"] = round(float(v), 4)
    for a in ("ea_all", "ea_base"):
        ds = [v[a]["delta"] for m, v in res["models"].items() if a in v and not m.startswith("feat")]
        fs_ = [v[a]["delta"] for m, v in res["models"].items() if a in v and m.startswith("feat")]
        if ds or fs_:
            res[f"summary_{a}"] = {
                "deep_mean_delta": round(float(np.mean(ds)), 4) if ds else None,
                "deep_improved": int(sum(d > 0 for d in ds)), "deep_n": len(ds),
                "feature_deltas": {m: v[a]["delta"] for m, v in res["models"].items()
                                   if a in v and m.startswith("feat")},
                "significant_gains": [m for m, v in res["models"].items() if a in v
                                      and v[a].get("p_bh", 1) < 0.05 and v[a]["delta"] > 0],
                "significant_losses": [m for m, v in res["models"].items() if a in v
                                       and v[a].get("p_bh", 1) < 0.05 and v[a]["delta"] < 0]}
    return res


def pretrain_stats(task):
    f = _find(task, "subject", "none", "fm")
    if not f:
        return None
    y, g, arms = _load(f)
    res = {"file": Path(f).name, "auc": {m: [round(float(_auc(y, p)), 4) for p in ps]
                                         for m, ps in arms.items()}, "contrasts": {}}
    for arch in ("BIOT", "CBraMod"):
        pt, sc = arms.get(f"{arch}_pt"), arms.get(f"{arch}_scratch200")
        if pt and sc:
            # the scratch control is seed-averaged too: contrast the seed-mean prediction
            r = hierarchical_contrast(y, g, pt, np.mean(sc, 0), n_boot=2000)
            r["p_signflip"] = signflip(_per_subject_delta(y, g, np.mean(pt, 0), np.mean(sc, 0)))
            res["contrasts"][f"{arch}_pt_minus_scratch"] = r
    return res


def factorial_stats(task):
    bench = sorted(glob.glob(str(REPO / f"results/_oof/unified_{task}_*.npz")))
    if not bench:
        return None
    yb, gb, arms_b = _load(bench[-1])
    cells = {"subject": {m: arms_b[m][0] for m in arms_b}}
    for pr in ("segment", "event"):
        f = _find(task, pr, "none", "proto")
        if f:
            y, g, arms = _load(f)
            if np.array_equal(y, yb) and np.array_equal(g, gb):
                cells[pr] = {m: ps[0] for m, ps in arms.items()}
    prots = [p for p in ("segment", "event", "subject") if p in cells]
    models = sorted(set.intersection(*[set(cells[p]) for p in prots])) if len(prots) > 1 else []
    if len(prots) < 2 or not models:
        return {"protocols": prots, "models": models}
    A = np.array([[_auc(yb, cells[p][m]) for p in prots] for m in models])
    med = np.array([[np.nanmedian([_auc(yb[gb == s], cells[p][m][gb == s]) for s in np.unique(gb)
                                   if len(np.unique(yb[gb == s])) == 2]) for p in prots]
                    for m in models])
    grand = A.mean()
    ss_m = len(prots) * ((A.mean(1) - grand) ** 2).sum()
    ss_p = len(models) * ((A.mean(0) - grand) ** 2).sum()
    ss_tot = ((A - grand) ** 2).sum()
    ss_r = ss_tot - ss_m - ss_p
    from scipy.stats import spearmanr
    rank = {}
    for i, p in enumerate(prots):
        for j, q in enumerate(prots):
            if j > i:
                rank[f"{p}~{q}"] = round(float(spearmanr(A[:, i], A[:, j]).correlation), 3)
    infl = {}
    for p in prots:
        if p != "subject":
            d = A[:, prots.index(p)] - A[:, prots.index("subject")]
            infl[p] = {"mean": round(float(d.mean()), 4), "min": round(float(d.min()), 4),
                       "max": round(float(d.max()), 4),
                       "per_model": {m: round(float(v), 4) for m, v in zip(models, d)}}
    return {"protocols": prots, "models": models,
            "auc_pooled": {m: {p: round(float(A[i, k]), 4) for k, p in enumerate(prots)}
                           for i, m in enumerate(models)},
            "auc_median_subject": {m: {p: round(float(med[i, k]), 4) for k, p in enumerate(prots)}
                                   for i, m in enumerate(models)},
            "inflation_vs_subject": infl, "rank_spearman": rank,
            "variance_share": {"protocol": round(float(ss_p / ss_tot), 3),
                               "model": round(float(ss_m / ss_tot), 3),
                               "interaction_residual": round(float(ss_r / ss_tot), 3)},
            "spread_across_models": {p: round(float(A[:, k].max() - A[:, k].min()), 4)
                                     for k, p in enumerate(prots)}}


def recipe_stats(task):
    """Recipe sensitivity: lr 3e-4 and 50 epochs against the shared recipe, per model."""
    bench = sorted(glob.glob(str(REPO / f"results/_oof/unified_{task}_*.npz")))
    if not bench:
        return None
    yb, gb, base = _load(bench[-1])
    seeds_f = _find(task, "subject", "none", "seeds")
    if seeds_f:
        ys, gs, sa = _load(seeds_f)
        if np.array_equal(ys, yb) and np.array_equal(gs, gb):
            for m, ps in sa.items():
                base.setdefault(m, []).extend(ps)
    out = {}
    for tag, pat in (("lr3e-4", f"variant_{task}_subject_none_e25_lr0.0003_*_recipe.npz"),
                     ("e50", f"variant_{task}_subject_none_e50_*_recipe.npz")):
        fs = [f for f in sorted(glob.glob(str(REPO / "results/_oof_variants" / pat))) if "smoke" not in f]
        if not fs:
            continue
        y, g, arms = _load(fs[-1])
        if not (np.array_equal(y, yb) and np.array_equal(g, gb)):
            continue
        for m, ps in arms.items():
            if m not in base:
                continue
            d = [float(_auc(y, p)) for p in base[m]]
            out.setdefault(m, {"default": round(float(np.mean(d)), 4),
                               "default_sd": round(float(np.std(d, ddof=1)), 4) if len(d) > 1 else None})
            out[m][tag] = round(float(_auc(y, ps[0])), 4)
            out[m][f"{tag}_delta"] = round(out[m][tag] - out[m]["default"], 4)
    if not out:
        return None
    deltas = [v[k] for v in out.values() for k in ("lr3e-4_delta", "e50_delta") if k in v]
    return {"models": out, "mean_abs_delta": round(float(np.mean(np.abs(deltas))), 4),
            "max_delta": round(float(np.max(deltas)), 4), "min_delta": round(float(np.min(deltas)), 4)}


def main() -> int:
    from bsf.utils.io import save_json
    from bsf.utils.provenance import new_results_dir
    out = new_results_dir("variant_stats", seed=0, config={})
    R = {"align": {}, "align_early": {}, "pretrain": {}, "factorial": {}, "recipe": {}}
    for t in TASKS:
        R["recipe"][t] = recipe_stats(t)
        R["align"][t] = align_stats(t)
        R["align_early"][t] = align_stats(t, early=True)
        R["pretrain"][t] = pretrain_stats(t)
        R["factorial"][t] = factorial_stats(t)
        a = R["align"][t] or {}
        for k in ("summary_ea_all", "summary_ea_base"):
            if k in a:
                print(f"[{t}] {k}: {a[k]}", flush=True)
        pt = R["pretrain"][t] or {}
        for k, v in (pt.get("contrasts") or {}).items():
            print(f"[{t}] {k}: {v['delta']:+.3f} [{v['lo']:+.3f},{v['hi']:+.3f}] p={v['p_two_sided']}"
                  f" sf={v['p_signflip']}", flush=True)
        fa = R["factorial"][t] or {}
        if "variance_share" in fa:
            print(f"[{t}] factorial {fa['protocols']} n_models={len(fa['models'])} "
                  f"variance {fa['variance_share']} rank {fa['rank_spearman']} "
                  f"inflation { {k: v['mean'] for k, v in fa['inflation_vs_subject'].items()} }",
                  flush=True)
    save_json(R, out / "results.json")
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
