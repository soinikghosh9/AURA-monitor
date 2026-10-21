#!/usr/bin/env python
"""Score the post-seal arms ONCE on the sealed TEST patients, as pre-declared in the manifest.

The arms here were introduced after the 2026-08-10 seal and listed, with their scoring rule, in
results/_blind_addendum_manifest.json before their development runs were complete. They are scored
together, once, whatever their development results; nothing is dropped. The journal
results/_blind_addendum_journal.json refuses a second real pass.

Scoring rule (from the manifest): fit on ALL development patients; baseline-referenced arms reserve one
third of each patient's negative clips as its baseline (typing: its non-discharge background), and every
arm -- including the original analytic arms, re-scored deterministically -- is evaluated on that same
reduced TEST set, so the arms are paired. Per arm: TEST AUC with a patient-bootstrap CI, and the paired
difference from the matched-input analytic arm.

  --selftest   run the identical code on a DEVELOPMENT-internal split (no TEST patient is read), to
               verify the pipeline before the single real pass
  --i-am-ready required for the real pass

-> results/blind_addendum_<stamp>/results.json (real) or blind_addendum_selftest_<stamp>/ (self-test)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "results" / "_blind_addendum_manifest.json"
JOURNAL = REPO / "results" / "_blind_addendum_journal.json"
SFREQ = {"ieeg": 400.0, "scalp": 250.0, "detect": 250.0, "typing": 500.0}


def _verify_manifest():
    m = json.loads(MANIFEST.read_text())
    sha = m.pop("sha256")
    body = json.dumps(m, indent=2, sort_keys=True)
    if hashlib.sha256(body.encode()).hexdigest() != sha:
        raise SystemExit("manifest digest mismatch: it has been edited since it was declared")
    return m, sha


def _split_selftest(W, y, g, ev, X, frac=0.3, seed=7):
    """A development-internal patient split standing in for DEV/TEST in the self-test."""
    subs = np.unique(g)
    rng = np.random.default_rng(seed)
    te = set(rng.choice(subs, max(2, int(round(frac * len(subs)))), replace=False))
    mt = np.isin(g, list(te))
    return (W[~mt], y[~mt], g[~mt], ev[~mt], X[~mt]), (W[mt], y[mt], g[mt], ev[mt], X[mt])


def _auc_ci(y, p, g, n_boot=2000, seed=0):
    from sklearn.metrics import roc_auc_score
    a = float(roc_auc_score(y, p))
    subs = np.unique(g); idx = {s: np.where(g == s)[0] for s in subs}
    rng = np.random.default_rng(seed); bs = []
    for _ in range(n_boot):
        ii = np.concatenate([idx[s] for s in rng.choice(subs, len(subs), replace=True)])
        if len(np.unique(y[ii])) == 2:
            bs.append(roc_auc_score(y[ii], p[ii]))
    return a, [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main() -> int:
    import torch
    from sklearn.base import clone
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from bench_data import baseline_mask, euclidean_align, feature_align, load_task, typing_baseline
    from run_benchmark_variants import _resample200
    from run_blind_test import _match
    from run_feature_fairness import _all_subwindows
    from run_final_stats import hierarchical_contrast
    from bsf.models import zoo
    from bsf.utils.compute import resolve_device
    from bsf.utils.io import save_json
    from bsf.utils.provenance import new_results_dir
    from bsf.utils.seeds import set_global_seeds

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default="typing,ieeg,scalp,detect")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--i-am-ready", action="store_true")
    ap.add_argument("--skip-deep", action="store_true", help="self-test only: analytic arms only")
    args = ap.parse_args()
    manifest, sha = _verify_manifest()
    real = not args.selftest
    if real:
        if not args.i_am_ready:
            print("Refusing the real pass without --i-am-ready (run --selftest first)."); return 2
        if JOURNAL.exists() and json.loads(JOURNAL.read_text()).get("status") == "done":
            print("The addendum has already been scored once; it is not re-run."); return 2
        JOURNAL.write_text(json.dumps({"status": "running", "manifest": sha}))
    set_global_seeds(1234)
    dev = resolve_device("auto")
    out = new_results_dir("blind_addendum" if real else "blind_addendum_selftest", seed=1234,
                          config={**vars(args), "manifest": sha})
    tasks = [t for t in args.tasks.split(",") if t]
    R = {"manifest_sha256": sha, "real_pass": real, "tasks": {}}
    for task in tasks:
        sf = SFREQ[task]
        D = load_task(task, "dev")
        if real:
            T = load_task(task, "test")
        else:
            D, T = _split_selftest(*D)
        Wd, yd, gd, evd, Xd = D; Wt, yt, gt, evt, Xt = T
        Wd, Wt = _match(Wd, Wt)
        # matched-input analytic vector (typing: one window, the pool's own vector)
        if task == "typing":
            Fd, Ft = Xd, Xt
        else:
            Fd = _all_subwindows(Wd, sf, 6); Ft = _all_subwindows(Wt, sf, 6)
        # baseline reservation (typing: external Non-IED background)
        if task == "typing":
            Wbd, gbd, Xbd = typing_baseline("dev")
            Wbt, gbt, Xbt = typing_baseline("dev" if not real else "test")
            nb = min(Wd.shape[2], Wbd.shape[2]); tb = min(Wd.shape[3], Wbd.shape[3])
            Wd, Wt = Wd[:, :, :nb, :tb], Wt[:, :, :nb, :tb]
            refWd = {s: Wbd[gbd == s][:, :, :nb, :tb] for s in np.unique(gd)}
            refWt = {s: Wbt[gbt == s][:, :, :nb, :tb] for s in np.unique(gt)}
            refXd = {s: Xbd[gbd == s] for s in np.unique(gd)}; refXt = {s: Xbt[gbt == s] for s in np.unique(gt)}
            keep_d = np.array([len(refWd.get(s, [])) >= 3 for s in gd])
            keep_t = np.array([len(refWt.get(s, [])) >= 3 for s in gt])
        else:
            bd, bt = baseline_mask(yd, gd), baseline_mask(yt, gt)
            refWd = {s: Wd[(gd == s) & bd] for s in np.unique(gd)}
            refWt = {s: Wt[(gt == s) & bt] for s in np.unique(gt)}
            refXd = {s: Fd[(gd == s) & bd] for s in np.unique(gd)}
            refXt = {s: Ft[(gt == s) & bt] for s in np.unique(gt)}
            keep_d = ~bd & np.array([len(refWd[s]) > 0 for s in gd])
            keep_t = ~bt & np.array([len(refWt[s]) > 0 for s in gt])
        sel = lambda a, k: a[k]  # noqa: E731
        Wd_, yd_, gd_, Fd_, Xd_ = (sel(a, keep_d) for a in (Wd, yd, gd, Fd, Xd))
        Wt_, yt_, gt_, Ft_, Xt_ = (sel(a, keep_t) for a in (Wt, yt, gt, Ft, Xt))
        print(f"[{task}] {'TEST' if real else 'SELFTEST'}: dev {len(yd_)} clips/{len(np.unique(gd_))} pts"
              f" -> scored {len(yt_)} clips/{len(np.unique(gt_))} pts", flush=True)
        preds = {}
        log = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000,  # noqa: E731
                                                                         class_weight="balanced"))
        gbm = lambda: GradientBoostingClassifier(n_estimators=200, random_state=0)  # noqa: E731
        preds["feat_logistic_sub"] = log().fit(Fd_, yd_).predict_proba(Ft_)[:, 1]
        preds["feat_gbm_sub"] = gbm().fit(Fd_, yd_).predict_proba(Ft_)[:, 1]
        preds["feat_logistic"] = log().fit(Xd_, yd_).predict_proba(Xt_)[:, 1]
        preds["feat_gbm"] = gbm().fit(Xd_, yd_).predict_proba(Xt_)[:, 1]
        Zd = feature_align(Fd_, gd_, refXd, "z"); Zt = feature_align(Ft_, gt_, refXt, "z")
        preds["feat_logistic_base"] = log().fit(Zd, yd_).predict_proba(Zt)[:, 1]
        Vd = feature_align(Fd_, gd_, refXd, "white"); Vt = feature_align(Ft_, gt_, refXt, "white")
        preds["feat_white_base"] = log().fit(Vd, yd_).predict_proba(Vt)[:, 1]
        if not args.skip_deep:
            n_sub, n_ch, n_t = Wd_.shape[1], Wd_.shape[2], Wd_.shape[3]
            W2d, W2t = _resample200(Wd_, sf), _resample200(Wt_, sf)
            nt2 = W2d.shape[-1]
            for m in ("BIOT_pt", "CBraMod_pt"):
                spec = zoo.build(m, n_ch, nt2, 200.0, device=dev, seed=0)
                p = zoo.fit_predict(spec, W2d.reshape(-1, n_ch, nt2), np.repeat(yd_, n_sub),
                                    W2t.reshape(-1, n_ch, nt2), dev, epochs=args.epochs, micro_bs=16,
                                    seed=0, backbone_lr=1e-4)
                preds[m] = p.reshape(len(yt_), n_sub).mean(1)
                del spec
                torch.cuda.empty_cache() if dev == "cuda" else None
            for m, base in (("BIOT_probe", "BIOT_pt"), ("CBraMod_probe", "CBraMod_pt")):
                spec = zoo.build(base, n_ch, nt2, 200.0, device=dev, seed=0)
                Ed = zoo.frozen_embed(spec, W2d.reshape(-1, n_ch, nt2), dev).reshape(len(yd_), n_sub, -1).mean(1)
                Et = zoo.frozen_embed(spec, W2t.reshape(-1, n_ch, nt2), dev).reshape(len(yt_), n_sub, -1).mean(1)
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1,
                                                                         class_weight="balanced"))
                preds[m] = clf.fit(Ed, yd_).predict_proba(Et)[:, 1]
                del spec
            Ad = euclidean_align(Wd_, gd_, refWd); At = euclidean_align(Wt_, gt_, refWt)
            for m in ("SCCNet", "deep_cnn_raw"):
                spec = zoo.build(m, n_ch, n_t, sf, device=dev, seed=0)
                p = zoo.fit_predict(spec, Ad.reshape(-1, n_ch, n_t), np.repeat(yd_, n_sub),
                                    At.reshape(-1, n_ch, n_t), dev, epochs=args.epochs, micro_bs=16,
                                    seed=0)
                preds[f"{m}_ea_base"] = p.reshape(len(yt_), n_sub).mean(1)
                del spec
                torch.cuda.empty_cache() if dev == "cuda" else None
        ref = "feat_gbm_sub" if task != "typing" else "feat_gbm_sub"
        res = {"n_clips": int(len(yt_)), "n_patients": int(len(np.unique(gt_))), "reference": ref}
        for m, p in preds.items():
            a, ci = _auc_ci(yt_, p, gt_)
            res[m] = {"auc": round(a, 4), "ci": [round(ci[0], 4), round(ci[1], 4)]}
            if m != ref:
                c = hierarchical_contrast(yt_, gt_, [p], preds[ref], n_boot=2000)
                res[m]["vs_ref"] = {k: c[k] for k in ("delta", "lo", "hi", "p_two_sided")}
            print(f"    {m:22s} AUC {a:.3f} [{ci[0]:.2f}, {ci[1]:.2f}]", flush=True)
        np.savez_compressed(out / f"preds_{task}.npz", _y=yt_.astype(np.int8), _groups=gt_,
                            **{k: np.asarray(v, np.float32) for k, v in preds.items()})
        R["tasks"][task] = res
        save_json(R, out / "results.json")
    if real:
        JOURNAL.write_text(json.dumps({"status": "done", "manifest": sha, "results": str(out)}))
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
