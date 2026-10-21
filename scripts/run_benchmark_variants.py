#!/usr/bin/env python
"""The benchmark's missing factors: protocol, patient alignment, seed, and real pretraining.

The unified benchmark trains 25 models once each, under one protocol, from random initialisation.
That design can support a claim about architecture only if the other things that move AUC are
measured on the same footing. This script varies them, one run per (task, protocol, alignment), with
every model inside a run scored on identical clips and folds so any two rows can be compared pairwise:

  --protocol  subject  cross-patient, subject-grouped folds (the benchmark's own protocol)
              event    patients shared between folds, but one seizure/hour never straddles a fold
              segment  clips split at random, ignoring patient and event -- the leaky protocol much of
                       the literature reports
  --align     none     per-clip standardisation only (the benchmark as run)
              ea_all   label-free per-patient alignment from every clip of that patient (transductive)
              ea_base  per-patient alignment from a reserved baseline of that patient's negative clips
                       (Non-IED background for typing); the reserved clips are dropped from scoring
  --seeds     weight-initialisation / batch-order seeds; folds are fixed across seeds, so the spread
              is model variance, not split variance
  models      any zoo name, plus BIOT_pt / CBraMod_pt (published weights, 200 Hz, backbone lr 1e-4),
              their *_scratch200 random-init controls, *_probe (frozen pretrained encoder + logistic),
              and the analytic arms feat_logistic, feat_gbm, feat_white (2nd-order reference whitening)

Deep models see the aligned waveform; the analytic arms see the 18-D vector z-scored (or whitened) by
the same patient reference, so an alignment effect can be read across both representations.

With --drop-baseline the reserved baseline clips are removed from EVERY arm, including align=none,
so the three alignment modes are paired on identical clips. Resumable per (model, seed).

-> results/bench_variant_<stamp>/results.json + leaderboard.csv ; OOF -> results/_oof_variants/
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_data import (SFREQ, baseline_mask, euclidean_align, feature_align,  # noqa: E402
                        load_task, seal_sha, typing_baseline)

FEATS = {"feat_logistic", "feat_gbm", "feat_white"}
PROBES = {"BIOT_probe": "BIOT_pt", "CBraMod_probe": "CBraMod_pt"}


def _resample200(W: np.ndarray, sf: float) -> np.ndarray:
    from math import gcd

    from scipy.signal import resample_poly
    a, b = 200, int(round(sf)); k = gcd(a, b)
    out = resample_poly(W, a // k, b // k, axis=-1).astype(np.float32)
    return out


def _folds(protocol, y, g, ev, splits):
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    if protocol == "segment":
        return list(StratifiedKFold(splits, shuffle=True, random_state=1234).split(y, y))
    groups = g if protocol == "subject" else ev
    return list(StratifiedGroupKFold(splits, shuffle=True, random_state=1234).split(
        np.zeros(len(y)), y, groups))


def main() -> int:
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
    ap.add_argument("--task", required=True, choices=["typing", "ieeg", "scalp", "detect"])
    ap.add_argument("--protocol", default="subject", choices=["subject", "event", "segment"])
    ap.add_argument("--align", default="none", choices=["none", "ea_all", "ea_base"])
    ap.add_argument("--drop-baseline", action="store_true")
    ap.add_argument("--baseline", default="random", choices=["random", "early"],
                    help="which third of each patient's negatives is reserved: a random draw, or the "
                         "earliest in recording order, so the baseline precedes every scored clip")
    ap.add_argument("--models", required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--splits", type=int, default=0, help="0 = 5 for typing, else 3")
    ap.add_argument("--max-per-class", type=int, default=60)
    ap.add_argument("--n-subwin", type=int, default=6)
    ap.add_argument("--max-per-subject", type=int, default=40)
    ap.add_argument("--micro-bs", type=int, default=16)
    ap.add_argument("--ft-backbone-lr", type=float, default=1e-4)
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="head/network learning rate (recipe-sensitivity units change it; the "
                         "benchmark recipe is 1e-3)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--montage", default="native", choices=["native", "db18"],
                    help="scalp tasks: db18 = the 18-channel bipolar double banana for every patient "
                         "(scripts/montage.py; extract_scalp_raw.py --montage db18 first)")
    ap.add_argument("--group-map", default="none", choices=["none", "animal"],
                    help="ieeg: animal = folds grouped by animal, merging dogs released under two labels")
    args = ap.parse_args()
    splits = args.splits or (5 if args.task == "typing" else 3)
    set_global_seeds(1234)
    dev = resolve_device("auto")
    out = new_results_dir("bench_variant", seed=1234, config={**vars(args), "device": dev})
    task = args.task
    sf = SFREQ[task]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    print(f"=== variant [{task}] protocol={args.protocol} align={args.align} "
          f"drop_baseline={args.drop_baseline} seeds={seeds} (seal {seal_sha()[:12]}) ===", flush=True)

    W, y, g, ev, X = load_task(task, "dev", args.max_per_class, args.n_subwin, args.max_per_subject,
                               **({"montage": args.montage} if args.montage != "native" else {}))

    # ---- the patient reference, and which clips are reserved for it ----
    base = np.zeros(len(y), bool)
    ref_W, ref_X = {}, {}
    if task == "typing":
        if args.align == "ea_base" or args.drop_baseline:
            # the baseline is external (Non-IED background), so nothing is reserved from the pool;
            # patients without one are dropped from every paired arm alike
            Wb, gb, Xb = typing_baseline("dev")
            n_ch = min(W.shape[2], Wb.shape[2]); n_t = min(W.shape[3], Wb.shape[3])
            W = W[:, :, :n_ch, :n_t]; Wb = Wb[:, :, :n_ch, :n_t]
            if args.align == "ea_base":
                for s in np.unique(g):
                    ref_W[s] = Wb[gb == s]; ref_X[s] = Xb[gb == s]
            missing = [s for s in np.unique(g) if (gb == s).sum() < 3]
            if missing:
                keep = ~np.isin(g, missing)
                print(f"  dropping {len(missing)} patients with no Non-IED baseline", flush=True)
                W, y, g, ev, X = W[keep], y[keep], g[keep], ev[keep], X[keep]
    else:
        if args.drop_baseline or args.align == "ea_base":
            if args.baseline == "early":
                from run_baseline_placement import early_mask
                base = early_mask(y, g, ev)
            else:
                base = baseline_mask(y, g)
        if args.align == "ea_base":
            for s in np.unique(g):
                ref_W[s] = W[(g == s) & base]; ref_X[s] = X[(g == s) & base]
            no_ref = [s for s in np.unique(g) if len(ref_W[s]) == 0]
            if no_ref:
                keep = ~np.isin(g, no_ref)
                print(f"  dropping {len(no_ref)} patients with too few negatives for a baseline",
                      flush=True)
                W, y, g, ev, X, base = W[keep], y[keep], g[keep], ev[keep], X[keep], base[keep]
    if base.any():
        keep = ~base
        if args.drop_baseline and args.align != "ea_base":
            # the none / ea_all arms of a paired comparison: drop the same reserved clips
            no_ref = [s for s in np.unique(g) if not ((g == s) & base).any()]
            keep &= ~np.isin(g, no_ref)
        W, y, g, ev, X = W[keep], y[keep], g[keep], ev[keep], X[keep]
    if args.align == "ea_all":
        for s in np.unique(g):
            ref_W[s] = W[g == s]; ref_X[s] = X[g == s]
    if args.align != "none":
        W = euclidean_align(W, g, ref_W)
    n_clip, n_sub, n_ch, n_t = W.shape
    g_fold = g
    if args.group_map == "animal":
        # recordings of one animal released under two dataset labels (check_canine_identity.py) must share
        # a fold; only the fold assignment changes, every per-subject analysis keeps the dataset label
        from run_ieeg_dedup import CONSERVATIVE
        g_fold = np.array([CONSERVATIVE.get(s, s) for s in np.asarray(g).astype(str)])
    folds = _folds(args.protocol, y, g_fold, ev, splits)
    print(f"  {n_clip} clips / {len(np.unique(g))} subjects / {len(np.unique(ev))} events | "
          f"{n_ch}ch x {n_t} @ {sf}Hz | {splits} folds", flush=True)

    lr_tag = "" if abs(args.lr - 1e-3) < 1e-12 else f"_lr{args.lr:g}"
    unit = (f"variant_{task}_{args.protocol}_{args.align}{'_db' if args.drop_baseline else ''}"
            f"{'_early' if args.baseline == 'early' else ''}"
            f"_e{args.epochs}{lr_tag}_s{splits}_n{n_clip}{('_' + args.tag) if args.tag else ''}"
            f"{('_' + args.montage) if args.montage != 'native' else ''}")
    done = resume.done_units(unit)
    oof_dir = Path("results/_oof_variants"); oof_dir.mkdir(parents=True, exist_ok=True)
    oof_path = oof_dir / f"{unit}.npz"
    store: dict[str, np.ndarray] = {}
    if oof_path.exists():
        with np.load(oof_path, allow_pickle=False) as z:
            store = {k: z[k] for k in z.files}
    store.update({"_y": y.astype(np.int8), "_groups": g.astype(str), "_events": ev.astype(str)})
    res = dict(done)

    def keep(key, p, extra):
        store[key] = np.asarray(p, np.float32)
        np.savez_compressed(oof_path, **store)
        res[key] = {"panel": C.full_panel(y, p, groups=g, n_boot=400), **extra}
        resume.update_unit(unit, key, res[key])
        d = res[key]["panel"]["discrimination"]
        print(f"    {key:28s} AUC {d['auc']:.3f} AUPRC {d['auprc']:.3f}", flush=True)

    # ---- analytic arms (deterministic: one seed) ----
    Xa = X
    if args.align != "none":
        Xa = feature_align(X, g, ref_X, "z")
    for m in [m for m in models if m in FEATS]:
        key = f"{m}@s0"
        if key in res:
            continue
        if m == "feat_white":
            Xin = feature_align(X, g, ref_X, "white") if args.align != "none" else X
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000,
                                                                     class_weight="balanced"))
        elif m == "feat_logistic":
            Xin, clf = Xa, make_pipeline(StandardScaler(), LogisticRegression(
                max_iter=2000, class_weight="balanced"))
        else:
            Xin, clf = Xa, GradientBoostingClassifier(n_estimators=200, random_state=0)
        oof = np.zeros(len(y))
        for tr, te in folds:
            oof[te] = clone(clf).fit(Xin[tr], y[tr]).predict_proba(Xin[te])[:, 1]
        keep(key, oof, {"input": "features_18d", "model": m, "seed": 0})

    # ---- deep models ----
    import torch
    deep = [m for m in models if m not in FEATS]
    need200 = any(m in zoo.PRETRAINED or m in PROBES for m in deep)
    W200 = _resample200(W, sf) if need200 else None
    for m in deep:
        for sd in (seeds if m not in PROBES else [0]):
            key = f"{m}@s{sd}"
            if key in res:
                print(f"    (resume) {key}", flush=True); continue
            try:
                oof = np.zeros(len(y)); npar = 0
                if m in PROBES:
                    spec = zoo.build(PROBES[m], n_ch, W200.shape[-1], 200.0, device=dev, seed=sd)
                    npar = spec.n_params
                    E = zoo.frozen_embed(spec, W200.reshape(-1, n_ch, W200.shape[-1]), dev)
                    E = E.reshape(n_clip, n_sub, -1).mean(1)
                    for tr, te in folds:
                        clf = make_pipeline(StandardScaler(), LogisticRegression(
                            max_iter=3000, C=0.1, class_weight="balanced"))
                        oof[te] = clf.fit(E[tr], y[tr]).predict_proba(E[te])[:, 1]
                    del spec
                    keep(key, oof, {"input": "frozen_pretrained_embedding", "model": m,
                                    "seed": sd, "n_params_encoder": npar, "embed_dim": E.shape[1]})
                    continue
                pre = m in zoo.PRETRAINED
                Win = W200 if pre else W
                sfi, nt = (200.0, W200.shape[-1]) if pre else (sf, n_t)
                for tr, te in folds:
                    spec = zoo.build(m, n_ch, nt, sfi, device=dev, seed=sd)
                    npar = spec.n_params
                    p = zoo.fit_predict(spec, Win[tr].reshape(-1, n_ch, nt), np.repeat(y[tr], n_sub),
                                        Win[te].reshape(-1, n_ch, nt), dev, epochs=args.epochs,
                                        micro_bs=args.micro_bs, seed=sd, lr=args.lr,
                                        backbone_lr=args.ft_backbone_lr if pre else None)
                    oof[te] = p.reshape(len(te), n_sub).mean(1)
                    del spec, p
                    if dev == "cuda":
                        torch.cuda.empty_cache()
                keep(key, oof, {"input": "raw@200Hz" if pre else "raw", "model": m, "seed": sd,
                                "n_params": npar, "family": zoo.family_of(m)
                                if not pre else ("pretrained_fm" if zoo.PRETRAINED[m][1]
                                                 else "fm_scratch")})
            except Exception as e:  # noqa: BLE001
                res[key] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}
                print(f"    {key:28s} FAILED {str(e)[:120]}", flush=True)
                if dev == "cuda":
                    torch.cuda.empty_cache()

    res["_meta"] = {"task": task, "protocol": args.protocol, "align": args.align,
                    "drop_baseline": args.drop_baseline, "n_clips": int(n_clip),
                    "n_subjects": int(len(np.unique(g))), "n_events": int(len(np.unique(ev))),
                    "splits": splits, "unit": unit, "oof": str(oof_path)}
    save_json({"blind_seal": seal_sha(), "results": res}, out / "results.json")
    rows = ["key,model,seed,auc,auc_lo,auc_hi,auprc,brier_skill,ece"]
    for k, r in res.items():
        if isinstance(r, dict) and "panel" in r:
            d = r["panel"]["discrimination"]; ci = r["panel"].get("ci", {}).get("auc", {}) or {}
            rows.append(f"{k},{r.get('model','')},{r.get('seed','')},{d['auc']},{ci.get('lo','')},"
                        f"{ci.get('hi','')},{d['auprc']},{r['panel']['calibration']['brier_skill']},"
                        f"{r['panel']['calibration']['ece']}")
    (out / "leaderboard.csv").write_text("\n".join(rows))
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
