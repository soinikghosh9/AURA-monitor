#!/usr/bin/env python
"""Does tuning the networks change the comparison with AURA? Nested, patient-grouped tuning.

The benchmark trained every network with one recipe and no tuning, so that no architecture was favoured
by search effort. The obvious objection is that untuned networks are handicapped. Here the three best
trainable networks of a task are tuned inside the patient-independent folds of the benchmark:

  outer   the benchmark's patient-grouped folds (5 for IED classification, 3 otherwise), unchanged
  inner   within each outer training set, one patient-grouped split (25% of patients) for validation
  grid    learning rate {3e-4, 1e-3, 3e-3} x epochs {25, 50}; everything else the benchmark recipe

The configuration with the best inner validation AUC is refitted on the whole outer training set and
scored on the outer test fold (nested estimate); the benchmark recipe (1e-3, 25 epochs) is refitted on
the same folds as the default. AURA (boosted and logistic, the 18 features of the same standardized
sub-windows) is fitted on identical outer folds; contrasts are paired: hierarchical bootstrap over
patients and the per-patient sign-flip test (IED classification: bootstrap only). Resumable per unit.

  python scripts/run_tuned_networks.py --task typing --models EEGNeX,EEGInceptionERP,CTNet
  python scripts/run_tuned_networks.py --task detect --models ContraWR,BIOT,ATCNet

-> results/tuned_<task>_<stamp>/results.json  (+ results/_tuned/<task>/ units)
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

GRID = [(lr, ep) for lr in (3e-4, 1e-3, 3e-3) for ep in (25, 50)]


def main() -> int:
    import torch
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from bench_data import SFREQ, load_task
    from bsf.models import zoo
    from bsf.utils.compute import resolve_device
    from bsf.utils.io import save_json
    from bsf.utils.provenance import new_results_dir
    from run_benchmark_variants import _folds
    from run_final_stats import hierarchical_contrast, signflip
    from run_streaming_monitor import _train_window_features
    from run_variant_stats import _per_subject_delta

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=["typing", "ieeg", "scalp", "detect"])
    ap.add_argument("--models", required=True)
    ap.add_argument("--montage", default="native")
    ap.add_argument("--micro-bs", type=int, default=16)
    a = ap.parse_args()
    dev = resolve_device("auto")
    task, sf = a.task, SFREQ[a.task]
    W, y, g, ev, _ = load_task(task, "dev", montage=a.montage) if task in ("scalp", "detect") else load_task(task, "dev")
    n, k, C_, T_ = W.shape
    folds = _folds("subject", y, g, ev, 5 if task == "typing" else 3)
    state = REPO / "results" / "_tuned" / f"{task}{'' if a.montage == 'native' else '_' + a.montage}"
    state.mkdir(parents=True, exist_ok=True)
    out = new_results_dir(f"tuned_{task}", seed=1234, config={**vars(a), "grid": GRID})
    print(f"=== nested tuning [{task}] {n} clips / {len(np.unique(g))} patients, {len(folds)} outer folds ===", flush=True)

    def train_predict(m, tr, te, lr, ep):
        spec = zoo.build(m, C_, T_, sf, device=dev, seed=0)
        p = zoo.fit_predict(spec, W[tr].reshape(-1, C_, T_), np.repeat(y[tr], k), W[te].reshape(-1, C_, T_), dev,
                            epochs=ep, lr=lr, micro_bs=a.micro_bs, seed=0)
        del spec
        if dev == "cuda":
            torch.cuda.empty_cache()
        return p.reshape(len(te), k).mean(1)

    def unit(name, fn):
        f = state / f"{name}.npy"
        if f.exists():
            return np.load(f)
        v = fn()
        np.save(f, v)
        return v

    # AURA on identical folds (the 18 features of the standardized sub-windows the networks see)
    F = _train_window_features(W, sf).mean(1)
    aura = {}
    for hname, mk in (("feat_gbm", lambda: GradientBoostingClassifier(n_estimators=200, random_state=0)),
                      ("feat_logistic", lambda: make_pipeline(StandardScaler(), LogisticRegression(
                          max_iter=2000, class_weight="balanced")))):
        oof = np.zeros(n)
        for tr, te in folds:
            oof[te] = mk().fit(F[tr], y[tr]).predict_proba(F[te])[:, 1]
        aura[hname] = oof
    ref_name = max(aura, key=lambda h: roc_auc_score(y, aura[h]))
    R = {"task": task, "montage": a.montage, "n_patients": int(len(np.unique(g))), "grid": GRID,
         "aura": {h: round(float(roc_auc_score(y, v)), 4) for h, v in aura.items()}, "aura_reference": ref_name,
         "models": {}}
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    for m in models:
        nested = np.zeros(n); default = np.zeros(n); chosen = []
        d_ci = GRID.index((1e-3, 25))
        for fi, (tr, te) in enumerate(folds):
            gss = GroupShuffleSplit(1, test_size=0.25, random_state=fi)
            itr, iva = next(gss.split(tr, y[tr], g[tr]))
            itr, iva = tr[itr], tr[iva]
            val = []
            for ci, (lr, ep) in enumerate(GRID):
                if len(np.unique(y[iva])) == 2:
                    pv = unit(f"{m}_f{fi}_inner_c{ci}", lambda: train_predict(m, itr, iva, lr, ep))
                    val.append(float(roc_auc_score(y[iva], pv)))
                else:                                      # degenerate inner split: keep the recipe
                    val.append(1.0 if ci == d_ci else -1.0)
            best = int(np.argmax(val))
            default[te] = unit(f"{m}_f{fi}_outer_c{d_ci}", lambda: train_predict(m, tr, te, *GRID[d_ci]))
            nested[te] = (default[te] if best == d_ci else
                          unit(f"{m}_f{fi}_outer_c{best}", lambda: train_predict(m, tr, te, *GRID[best])))
            chosen.append({"fold": fi, "lr": GRID[best][0], "epochs": GRID[best][1],
                           "inner_auc": round(val[best], 4),
                           "inner_auc_default": round(val[d_ci], 4)})
            print(f"  {m} fold {fi}: inner " + " ".join(f"{v:.3f}" for v in val) +
                  f" -> lr {GRID[best][0]:g} ep {GRID[best][1]}", flush=True)
        res = {"default_auc": round(float(roc_auc_score(y, default)), 4),
               "nested_auc": round(float(roc_auc_score(y, nested)), 4), "chosen": chosen}
        for lab, pp in (("nested", nested), ("default", default)):
            hc = hierarchical_contrast(y, g, [pp], aura[ref_name], n_boot=2000)
            dd = {"delta": round(hc["delta"], 4), "lo": round(hc["lo"], 4), "hi": round(hc["hi"], 4),
                  "p_boot": hc["p_two_sided"]}
            if task != "typing":
                dd["p_signflip"] = signflip(_per_subject_delta(y, g, pp, aura[ref_name]))
            res[f"{lab}_vs_aura"] = dd
        hc = hierarchical_contrast(y, g, [nested], default, n_boot=2000)
        res["nested_vs_default"] = {"delta": round(hc["delta"], 4), "lo": round(hc["lo"], 4),
                                    "hi": round(hc["hi"], 4), "p_boot": hc["p_two_sided"]}
        R["models"][m] = res
        np.savez(out / f"oof_{m}.npz", nested=nested, default=default, y=y, g=g)
        print(f"[{m}] default {res['default_auc']:.3f} nested {res['nested_auc']:.3f} "
              f"| AURA ({ref_name}) {R['aura'][ref_name]:.3f}; nested - AURA {res['nested_vs_aura']['delta']:+.3f} "
              f"[{res['nested_vs_aura']['lo']:+.3f}, {res['nested_vs_aura']['hi']:+.3f}]", flush=True)
        save_json(R, out / "results.json")
    np.savez(out / "oof_aura.npz", **aura, y=y, g=g)
    save_json(R, out / "results.json")
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
