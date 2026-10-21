#!/usr/bin/env python
"""Score every model ONCE on the sealed blind TEST subjects, and report the optimism gap.

This is the pass the whole benchmark exists to earn. Each model is fitted on ALL DEV subjects -- one
fit, no cross-validation, no selection -- and scored on TEST subjects that no fit, no hyperparameter,
no architecture decision and no earlier run has ever touched.

The headline is not the TEST leaderboard. It is **DEV minus TEST, per model**. AURA-Net's DEV numbers
are optimistically biased in a way the CNN's and the zoo's are not, because six of its architecture
decisions were made in response to cross-validated numbers on this cohort, while the zoo models were
never tuned here at all. The gap measures how much each model's development consumed, and printing it
beside the raw scores is the difference between a benchmark and an advertisement.

Run this ONCE, after the DEV leaderboard is frozen. If a result here prompts a model change, that
change is DEV work and this TEST result is contaminated -- say so in the manuscript rather than
quietly re-running.

-> results/blind_test_<stamp>/results.json + blind_leaderboard.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_unified_benchmark import (  # noqa: E402
    _load_detect, _load_ieeg, _load_scalp, _load_typing, feature_model_size)

SEAL = Path("results/_blind_split.json")


def _match(A, B):
    """Crop a DEV and a TEST pool to a common (n_sub, n_chan, n_time).

    A model fitted on DEV must receive TEST inputs of the same shape, and the two pools were each
    reduced to their OWN common montage independently. Cropping both to the joint minimum is the only
    way the same weights can score both, and it is applied before any fitting so it cannot be tuned.
    """
    n_s = min(A.shape[1], B.shape[1]); n_c = min(A.shape[2], B.shape[2])
    n_t = min(A.shape[3], B.shape[3])
    return A[:, :n_s, :n_c, :n_t], B[:, :n_s, :n_c, :n_t]


def _frozen_dev(path: str | None) -> dict:
    """Load the frozen DEV leaderboard so the gap can be computed without refitting anything."""
    from bsf.utils.io import latest_result_dirs
    dirs = ([Path(path)] if path else latest_result_dirs("unified_benchmark_*")[-8:])
    dev = {}
    for d in dirs:
        f = Path(d) / "results.json"
        if not f.exists():
            continue
        for task, tr in json.loads(f.read_text()).get("tasks", {}).items():
            if not isinstance(tr, dict):
                continue
            for m, r in tr.items():
                if isinstance(r, dict) and "panel" in r:
                    # ASSIGN, not setdefault. latest_result_dirs returns newest LAST, so setdefault
                    # kept the OLDEST value for every model -- and the optimism gap is then measured
                    # against a development number that has since been superseded. It mattered here:
                    # AURA-Net was refit after its first DEV entries were written, so its gaps were
                    # computed against a model that no longer exists.
                    dev.setdefault(task, {})[m] = r["panel"]["discrimination"]["auc"]
    return dev


def main() -> int:
    import torch
    from sklearn.base import clone
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from bsf.eval import clinical as C
    from bsf.models import zoo
    from bsf.utils import resume
    from bsf.utils.compute import resolve_device
    from bsf.utils.io import save_json
    from bsf.utils.provenance import new_results_dir
    from bsf.utils.seeds import set_global_seeds

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default="typing,ieeg,scalp,detect")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--max-per-class", type=int, default=60)
    ap.add_argument("--n-subwin", type=int, default=6)
    ap.add_argument("--max-per-subject", type=int, default=40)
    ap.add_argument("--micro-bs", type=int, default=16)
    ap.add_argument("--dev-results", default="", help="frozen DEV dir; default = latest")
    ap.add_argument("--i-am-ready", action="store_true",
                    help="required. TEST is scoreable ONCE; this flag is the deliberate act.")
    args = ap.parse_args()
    if not args.i_am_ready:
        print("Refusing to score the blind set without --i-am-ready.\n"
              "TEST is evaluated once, after DEV is frozen. If you are re-running this because you\n"
              "did not like the first result, the result is contaminated and must be reported as\n"
              "such rather than replaced.")
        return 2

    set_global_seeds(1234)
    dev_name = resolve_device("auto")
    seal = json.loads(SEAL.read_text())
    out = new_results_dir("blind_test", seed=1234, config={**vars(args), "seal": seal["sha256"]})
    dev_auc = _frozen_dev(args.dev_results or None)
    print(f"=== BLIND TEST (seal {seal['sha256'][:12]}..., device={dev_name}) ===", flush=True)

    loaders = {"ieeg": lambda s: _load_ieeg(args.max_per_class, args.n_subwin, s),
               "scalp": lambda s: _load_scalp(args.max_per_class, args.n_subwin, s),
               "detect": lambda s: _load_detect(args.max_per_class, args.n_subwin, s),
               "typing": lambda s: _load_typing(args.max_per_subject, s)}
    sfreqs = {"ieeg": 400.0, "scalp": 250.0, "detect": 250.0, "typing": 500.0}
    results = {}

    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        D, T = loaders[task]("dev"), loaders[task]("test")
        if D is None or T is None:
            print(f"  [{task}] missing {'DEV' if D is None else 'TEST'} data -- skipped", flush=True)
            results[task] = {"skipped": "no data"}
            continue
        Wd, yd, gd, Xd = D
        Wt, yt, gt, Xt = T
        Wd, Wt = _match(Wd, Wt)
        n_sub, n_ch, n_t = Wd.shape[1], Wd.shape[2], Wd.shape[3]
        sf = sfreqs[task]
        print(f"  [{task}] DEV {len(yd)} clips / {len(np.unique(gd))} subj -> "
              f"TEST {len(yt)} clips / {len(np.unique(gt))} subj | {n_ch}ch x {n_t} @ {sf}Hz",
              flush=True)
        if len(np.unique(yt)) < 2:
            print(f"  [{task}] TEST has one class only -- not scoreable", flush=True)
            results[task] = {"skipped": "single-class TEST"}
            continue

        unit = f"blind_{task}_e{args.epochs}_n{len(yd)}_{len(yt)}"
        task_res = dict(resume.done_units(unit))

        def record(name, p, extra):
            task_res[name] = {"panel": C.full_panel(yt, p, groups=gt, n_boot=400),
                              "dev_auc": dev_auc.get(task, {}).get(name), **extra}
            a = task_res[name]["panel"]["discrimination"]["auc"]
            d = task_res[name]["dev_auc"]
            gap = f"{d - a:+.3f}" if d is not None else "  n/a"
            resume.update_unit(unit, name, task_res[name])
            print(f"    {name:22s} TEST {a:.3f}   DEV {d if d is None else f'{d:.3f}'}   "
                  f"gap {gap}", flush=True)

        for mname, clf in (("feat_logistic", make_pipeline(StandardScaler(), LogisticRegression(
                                max_iter=1000, class_weight="balanced"))),
                           ("feat_gbm", GradientBoostingClassifier(n_estimators=200,
                                                                   random_state=0))):
            if mname in task_res:
                continue
            m = clone(clf).fit(Xd, yd)
            record(mname, m.predict_proba(Xt)[:, 1], {"n_params": feature_model_size(m, Xd.shape[1]),
                                                      "input": "features_18d", "recipe": "shared"})

        runnable, skipped = zoo.available(n_ch, n_t, sf, device=dev_name)
        for mname in runnable:
            if mname in task_res:
                print(f"    (resume) {mname}", flush=True); continue
            try:
                spec = zoo.build(mname, n_ch, n_t, sf, device=dev_name)
                p = zoo.fit_predict(spec, Wd.reshape(-1, n_ch, n_t), np.repeat(yd, n_sub),
                                    Wt.reshape(-1, n_ch, n_t), dev_name,
                                    epochs=args.epochs, micro_bs=args.micro_bs)
                record(mname, p.reshape(len(yt), n_sub).mean(1),
                       {"n_params": spec.n_params, "input": "raw", "recipe": "shared",
                        "family": zoo.family_of(mname)})
                del spec, p
            except Exception as e:  # noqa: BLE001
                task_res[mname] = {"error": f"{type(e).__name__}: {str(e)[:100]}"}
                print(f"    {mname:22s} FAILED {str(e)[:70]}", flush=True)
            if dev_name == "cuda":
                torch.cuda.empty_cache()

        for mname, kw in (("auranet_hybrid", dict(d_feat=Xd.shape[1])), ("auranet_pure", {})):
            if mname in task_res:
                continue
            try:
                from bsf.models.auranet_train import AURANetClassifier
                clf = AURANetClassifier(fs=sf, epochs=args.epochs, device=dev_name, seed=0, **kw)
                clf.fit(Wd, yd, gd, feats=Xd if kw else None)
                p = clf.predict_proba(Wt, gt, feats=Xt if kw else None)
                record(mname, p, {"n_params": sum(q.numel() for q in clf.model.parameters()),
                                  "input": "raw+features" if kw else "raw",
                                  "recipe": "auranet_native", "family": "repo_ours"})
                del clf
            except Exception as e:  # noqa: BLE001
                task_res[mname] = {"error": f"{type(e).__name__}: {str(e)[:100]}"}
                print(f"    {mname:22s} FAILED {str(e)[:70]}", flush=True)
            if dev_name == "cuda":
                torch.cuda.empty_cache()

        task_res["_skipped_models"] = skipped
        task_res["_meta"] = {"n_dev": int(len(yd)), "n_test": int(len(yt)),
                             "n_test_subjects": int(len(np.unique(gt))),
                             "n_channels": int(n_ch), "n_times": int(n_t)}
        results[task] = task_res

    R = {"protocol": "fit on ALL DEV subjects, score ONCE on sealed TEST subjects",
         "blind_seal": seal["sha256"], "rules": seal["rules"], "tasks": results}
    save_json(R, out / "results.json")

    rows = ["task,model,family,input,recipe,n_params,test_auc,test_auc_lo,test_auc_hi,dev_auc,"
            "optimism_gap,test_auprc,test_brier_skill,test_ece"]
    for task, tr in results.items():
        if not isinstance(tr, dict):
            continue
        for m, r in tr.items():
            if not isinstance(r, dict) or "panel" not in r:
                continue
            pn = r["panel"]; d = pn["discrimination"]; ci = pn.get("ci", {}).get("auc", {}) or {}
            dv = r.get("dev_auc")
            rows.append(f"{task},{m},{r.get('family','')},{r.get('input','')},{r.get('recipe','')},"
                        f"{r.get('n_params',0)},{d['auc']},{ci.get('lo','')},{ci.get('hi','')},"
                        f"{'' if dv is None else dv},"
                        f"{'' if dv is None else round(dv - d['auc'], 4)},{d['auprc']},"
                        f"{pn['calibration']['brier_skill']},{pn['calibration']['ece']}")
    (out / "blind_leaderboard.csv").write_text("\n".join(rows))
    print(f"\n-> {out}  ({len(rows) - 1} rows). TEST is now SPENT for this seal.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
