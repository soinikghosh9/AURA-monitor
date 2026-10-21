#!/usr/bin/env python
"""Every deep model, both tasks, one protocol, DEV subjects only.

The comparison this project has never actually run. Prior benchmarks pitted our models against a
handful of baselines under recipes that differed in ways that turned out to matter more than the
architectures: a convenience-vs-causal support protocol was worth +0.227 AUC, a crippled input
pipeline turned 0.855 into 0.54 for the same model class, and an unequal optimiser budget gave the
baseline 2.2x the parameter updates. So here every model -- 21 from braindecode including the BIOT,
EEGPT and CBraMod foundation models this manuscript could previously only cite, plus the repo CNN and
AURA-Net, plus the analytic feature baselines -- is trained by ONE recipe on IDENTICAL folds.

Three clinical questions, all cross-patient, because that is the regime a deployed model faces:
  T1 FORECASTING  pre-ictal vs inter-ictal, subject-grouped CV, iEEG (``ieeg``) and scalp (``scalp``)
  T3 DETECTION    ictal vs non-ictal scalp windows (``detect``) -- the task clinics deploy TODAY, kept
                  separate from forecasting rather than relabelled from it
  T4 TYPING       generalized vs focal discharges (``typing``), subject-grouped CV
T5 RISK QUANTIFICATION is not a separate sweep: calibration, reliability, conformal coverage and
decision value are in the panel computed for every model on every task, because a probability's
truthfulness is a property of each row, not a fifth column.

DEV SUBJECTS ONLY. The blind TEST subjects are untouched here and scored once, later, by
run_blind_test.py. Every model gets the full clinical panel (discrimination, calibration, decision
value, event-level clinical rates, bootstrap CIs resampled over subjects), not a bare AUC.

-> results/unified_benchmark_<stamp>/results.json
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
SEAL = Path("results/_blind_split.json")


def dev_subjects(cohort: str, side: str = "dev") -> set[str]:
    """Subjects on one side of the sealed split. ``side='test'`` is the blind set: only
    run_blind_test.py may pass it, once, after DEV results are frozen."""
    if not SEAL.exists():
        raise SystemExit("no blind split sealed; run scripts/make_blind_split.py first")
    return set(json.loads(SEAL.read_text())["cohorts"].get(cohort, {}).get(side, []))


def _pool(W, y, g, X):
    """Harmonise ragged per-subject arrays and normalise per clip.

    Cross-patient pooling forces two compromises, both stated rather than hidden. Channel counts differ
    between subjects, so the montage is truncated to the common minimum -- channel k is then not the
    same electrode across subjects, which is a real limitation of every cross-patient raw model, ours
    included. Amplitude scale differs by orders of magnitude between rigs, so each clip is robustly
    standardised **using only its own samples**: no statistic crosses a clip, let alone a fold, so this
    cannot leak, and it is applied identically to every model so it cannot bias the comparison.
    """
    if not W:
        return None
    n_s = min(w.shape[1] for w in W)
    n_ch = min(w.shape[2] for w in W)
    n_t = min(w.shape[3] for w in W)
    Wa = np.concatenate([w[:, :n_s, :n_ch, :n_t] for w in W]).astype(np.float32)
    med = np.median(Wa, axis=-1, keepdims=True)
    mad = np.median(np.abs(Wa - med), axis=-1, keepdims=True) * 1.4826 + 1e-6
    Wa = np.nan_to_num(np.clip((Wa - med) / mad, -20, 20))
    return (Wa, np.concatenate(y), np.concatenate(g),
            np.nan_to_num(np.concatenate(X)).astype(np.float32))


def _load_ieeg(max_per_class: int, n_subwin: int, side: str = "dev"):
    """Pooled iEEG clips from DEV subjects only."""
    from bsf.data.clip_cache import build_or_load
    from bsf.data.kaggle_seizure import discover_kaggle
    W, y, g, X = [], [], [], []
    for root, cohort in (("datasets/kaggle_aes", "kaggle_ieeg"),
                         ("datasets/ieeg_canine", "canine_ieeg")):
        dev = dev_subjects(cohort, side)
        disc = discover_kaggle(root)
        for subj in sorted(disc):
            if subj not in dev:
                continue
            c = build_or_load(root, subj, disc[subj], n_subwin=n_subwin,
                              max_per_class=max_per_class, verbose=False)
            if c is None or len(np.unique(c["y"])) < 2:
                continue
            W.append(c["W"]); y.append(c["y"]); X.append(c["Xplus"])
            g.append(np.array([f"{cohort}_{subj}"] * len(c["y"])))
    return _pool(W, y, g, X)


def _load_scalp(max_per_class: int, n_subwin: int, side: str = "dev"):
    """Pooled raw scalp clips from DEV subjects only (needs extract_scalp_raw.py to have run)."""
    from bsf.data.clip_cache import load
    params = dict(n_subwin=n_subwin, win_s=8.0, target_fs=250.0,
                  max_per_class=max_per_class, with_features=True, v=5, cohort="scalp_raw")
    W, y, g, X = [], [], [], []
    for coh, tag in (("chbmit", "scalp_chbmit"), ("siena", "scalp_siena")):
        for s in sorted(dev_subjects(tag, side)):
            c = load("scalp_raw", f"{coh}_{s}", **params)
            if c is None or len(np.unique(c["y"])) < 2:
                continue
            W.append(c["W"]); y.append(c["y"]); X.append(c["Xplus"])
            g.append(np.array([f"{coh}_{s}"] * len(c["y"])))
    return _pool(W, y, g, X)


def _load_detect(max_per_class: int, n_subwin: int, side: str = "dev"):
    """Pooled ictal-vs-non-ictal scalp windows: T3, the task clinics actually deploy today.

    Deliberately a separate cache from the forecasting arm rather than a relabelling of it. Detection
    asks whether a seizure is happening now; forecasting asks whether one is coming. Conflating them
    is how a detector gets reported as a forecaster.
    """
    from bsf.data.clip_cache import load
    params = dict(n_subwin=n_subwin, win_s=8.0, target_fs=250.0,
                  max_per_class=max_per_class, with_features=True, v=5,
                  cohort="scalp_det", task="detection", buffer_s=120.0, det_v=2)
    W, y, g, X = [], [], [], []
    for coh, tag in (("chbmit", "scalp_chbmit"), ("siena", "scalp_siena")):
        for s in sorted(dev_subjects(tag, side)):
            c = load("scalp_det", f"{coh}_{s}", **params)
            if c is None or len(np.unique(c["y"])) < 2:
                continue
            W.append(c["W"]); y.append(c["y"]); X.append(c["Xplus"])
            g.append(np.array([f"{coh}_{s}"] * len(c["y"])))
    return _pool(W, y, g, X)


def _load_typing(max_per_subject: int, side: str = "dev"):
    """VEPISET generalized-vs-focal, DEV subjects only.

    Cached like the other cohorts. Uncached this was the slowest loader in the program by a wide
    margin -- the phase-amplitude coupling term in ``rich_plus`` costs roughly a second per clip, so
    ~1400 clips meant ~25 minutes of identical recomputation on every run that touches typing, of
    which there are several (benchmark, deployability, blind test).
    """
    from bsf.data.clip_cache import load as cache_load, save as cache_save
    from bsf.data.features import rich_plus
    from bsf.data.vepiset import by_patient, discover, load_clip
    cp = dict(max_per_subject=max_per_subject, side=side, v=1, cohort="vepiset_typing")
    hit = cache_load("vepiset_typing", "pooled", **cp)
    if hit is not None:
        # stored under "Wf", not "W": clip_cache.save force-downcasts "W" to float16, and these clips
        # are already z-scored, so fp16 would make a cached run differ slightly from an uncached one
        return hit["Wf"], hit["y"], hit["groups"].astype(str), hit["Xplus"]
    dev = dev_subjects("vepiset_typing", side)
    rng = np.random.default_rng(1234)
    W, y, g = [], [], []
    for pid, clips in sorted(by_patient(discover("datasets/vepiset-dataset")).items()):
        if pid not in dev:
            continue
        ied = [c for c in clips if c.cls in (1, 2, 3, 5)]
        if not ied:
            continue
        if len(ied) > max_per_subject:
            ied = [ied[i] for i in rng.choice(len(ied), max_per_subject, replace=False)]
        for c in ied:
            a = np.nan_to_num(np.asarray(load_clip(c.path), dtype=np.float32))
            if a.ndim != 2:
                continue
            if a.shape[0] < a.shape[1]:
                a = a.T
            W.append(a.T); y.append(int(c.cls == 1)); g.append(pid)
    if not W or len(set(y)) < 2:
        return None
    n_ch = min(w.shape[0] for w in W); n_t = min(w.shape[1] for w in W)
    Wa = np.stack([w[:n_ch, :n_t] for w in W]).astype(np.float32)
    mu = Wa.mean(-1, keepdims=True); sd = Wa.std(-1, keepdims=True) + 1e-6
    Wa = np.nan_to_num((Wa - mu) / sd)
    X = np.stack([np.nan_to_num(rich_plus(Wa[i].T, 500.0)) for i in range(len(Wa))])
    Wa, y, g, X = Wa[:, None], np.asarray(y), np.asarray(g), X.astype(np.float32)
    cache_save("vepiset_typing", "pooled",
               {"Wf": Wa, "y": y, "groups": np.asarray(g).astype(str), "Xplus": X}, **cp)
    return Wa, y, g, X


def feature_model_size(clf, n_feat: int) -> int:
    """Free parameters of a fitted feature model.

    The logistic arm has n_feat weights plus a bias. A gradient-boosted ensemble does not: every tree
    stores a threshold per split and a value per leaf, so its size is its node count. An earlier
    version reported n_feat + 1 for both, which put "19 parameters" on a model with thousands.
    """
    est = clf[-1] if hasattr(clf, "steps") else clf
    trees = getattr(est, "estimators_", None)
    if trees is None:
        return int(n_feat + 1)
    return int(sum(t.tree_.node_count for t in np.ravel(trees)))


def main() -> int:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
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
    ap.add_argument("--tasks", default="ieeg,typing,scalp,detect")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--max-per-class", type=int, default=60)
    ap.add_argument("--n-subwin", type=int, default=6)
    ap.add_argument("--max-per-subject", type=int, default=40)
    ap.add_argument("--micro-bs", type=int, default=16,
                    help="gradient-accumulation chunk; bounds memory for the 25M-param models")
    ap.add_argument("--models", default="", help="comma-separated subset; default = all runnable")
    ap.add_argument("--refit-models", default="",
                    help="comma-separated names to DROP from this config's checkpoints and refit, "
                         "leaving every other model's stored result untouched. Used when one model "
                         "changes and refitting the other twenty-four would be waste.")
    ap.add_argument("--force", action="store_true",
                    help="discard this config's checkpoints and refit. Needed to regenerate runs "
                         "made before out-of-fold predictions were kept, since without those a "
                         "paired comparison is impossible after the fact.")
    args = ap.parse_args()
    set_global_seeds(1234)
    dev = resolve_device("auto")
    out = new_results_dir("unified_benchmark", seed=1234, config={**vars(args), "device": dev})
    seal = json.loads(SEAL.read_text())
    print(f"=== unified benchmark (DEV only, seal {seal['sha256'][:12]}..., device={dev}) ===",
          flush=True)

    loaders = {"ieeg": lambda: _load_ieeg(args.max_per_class, args.n_subwin),
               "scalp": lambda: _load_scalp(args.max_per_class, args.n_subwin),
               "detect": lambda: _load_detect(args.max_per_class, args.n_subwin),
               "typing": lambda: _load_typing(args.max_per_subject)}
    sfreqs = {"ieeg": 400.0, "scalp": 250.0, "detect": 250.0, "typing": 500.0}
    results = {}

    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        data = loaders[task]()
        if data is None:
            print(f"  [{task}] no DEV data available -- skipped", flush=True)
            results[task] = {"skipped": "no data"}
            continue
        W, y, g, X = data
        # sub-windows are flattened into the sample axis for the zoo (all models take (B, C, T));
        # predictions are pooled back to the clip by mean, identically for every model.
        n_clip, n_sub, n_ch, n_t = W.shape
        sf = sfreqs[task]
        names = ([m.strip() for m in args.models.split(",") if m.strip()]
                 or zoo.ALL_BRAINDECODE + ["deep_cnn_raw"])
        runnable, skipped = zoo.available(n_ch, n_t, sf, device=dev, names=names)
        print(f"  [{task}] {n_clip} clips, {len(np.unique(g))} DEV subjects, {n_ch}ch x {n_t} "
              f"@ {sf}Hz | {len(runnable)} models runnable, {len(skipped)} skipped", flush=True)

        # The resume key carries the CONFIG, not just the task. Keyed on the task alone, a 2-epoch
        # smoke run leaves checkpoints that a later 25-epoch run silently adopts and reports as its
        # own -- the same smoke-shadowing failure this project has already been bitten by once.
        unit = (f"{task}_e{args.epochs}_s{args.splits}_c{args.max_per_class}"
                f"_w{args.n_subwin}_p{args.max_per_subject}_n{n_clip}")
        if args.force:
            resume.reset(f"unified_{unit}")
        refit = {m.strip() for m in args.refit_models.split(",") if m.strip()}
        if refit:
            st = resume.load_state(f"unified_{unit}")
            u = st.get("units", st)
            dropped = [m for m in list(u) if m in refit]
            for m in dropped:
                u.pop(m, None)
            if dropped:
                resume.save_state(f"unified_{unit}", st)
                print(f"    refitting {dropped} (other results kept)", flush=True)
        done = resume.done_units(f"unified_{unit}")
        # Keep the out-of-fold predictions, not just the summary panels. Two models are compared on
        # the SAME clips, so the honest test is a PAIRED bootstrap over subjects, which cancels the
        # subject-level variance that dominates here; judging by whether marginal CIs overlap is both
        # the wrong test and a much weaker one. Without the per-clip vectors that test is impossible
        # after the fact, and refitting to recover them costs hours.
        oof_dir = Path("results/_oof"); oof_dir.mkdir(parents=True, exist_ok=True)
        oof_path = oof_dir / f"unified_{unit}.npz"
        oof_store: dict[str, np.ndarray] = {}
        if oof_path.exists():
            with np.load(oof_path, allow_pickle=False) as z:
                oof_store = {k: z[k] for k in z.files}
        oof_store.update({"_y": y.astype(np.int8), "_groups": np.asarray(g).astype(str)})

        def _keep_oof(name, p):
            oof_store[name] = np.asarray(p, np.float32)
            np.savez_compressed(oof_path, **oof_store)
        task_res = dict(done)
        skf = StratifiedGroupKFold(args.splits, shuffle=True, random_state=1234)
        folds = list(skf.split(np.zeros(len(y)), y, g))

        # ---- analytic feature baselines on the SAME folds ----
        for mname, clf in (("feat_logistic", make_pipeline(StandardScaler(), LogisticRegression(
                                max_iter=1000, class_weight="balanced"))),
                           ("feat_gbm", GradientBoostingClassifier(n_estimators=200,
                                                                   random_state=0))):
            if mname in task_res:
                continue
            oof = np.zeros(len(y)); fitted = None
            for tr, te in folds:
                from sklearn.base import clone
                fitted = clone(clf).fit(X[tr], y[tr])
                oof[te] = fitted.predict_proba(X[te])[:, 1]
            task_res[mname] = {"panel": C.full_panel(y, oof, groups=g, n_boot=400),
                               "n_params": feature_model_size(fitted, X.shape[1]),
                               "input": "features_18d"}
            _keep_oof(mname, oof)
            resume.update_unit(f"unified_{unit}", mname, task_res[mname])
            print(f"    {mname:22s} AUC {task_res[mname]['panel']['discrimination']['auc']:.3f} "
                  f"AUPRC {task_res[mname]['panel']['discrimination']['auprc']:.3f}", flush=True)

        # ---- every deep model, one recipe ----
        import torch
        for mname in runnable:
            if mname in task_res:
                print(f"    (resume) {mname}", flush=True); continue
            try:
                oof = np.zeros(len(y)); npar = 0
                for tr, te in folds:
                    spec = zoo.build(mname, n_ch, n_t, sf, device=dev)
                    npar = spec.n_params
                    p = zoo.fit_predict(spec, W[tr].reshape(-1, n_ch, n_t),
                                        np.repeat(y[tr], n_sub),
                                        W[te].reshape(-1, n_ch, n_t), dev,
                                        epochs=args.epochs, micro_bs=args.micro_bs)
                    oof[te] = p.reshape(len(te), n_sub).mean(1)
                    del spec, p
                    if dev == "cuda":
                        torch.cuda.empty_cache()
                task_res[mname] = {"panel": C.full_panel(y, oof, groups=g, n_boot=400),
                                   "n_params": npar, "input": "raw",
                                   "recipe": "shared", "family": zoo.family_of(mname)}
                _keep_oof(mname, oof)
                resume.update_unit(f"unified_{unit}", mname, task_res[mname])
                d = task_res[mname]["panel"]["discrimination"]
                print(f"    {mname:22s} AUC {d['auc']:.3f} AUPRC {d['auprc']:.3f} "
                      f"({npar:,} params)", flush=True)
            except Exception as e:  # noqa: BLE001
                task_res[mname] = {"error": f"{type(e).__name__}: {str(e)[:100]}"}
                print(f"    {mname:22s} FAILED {str(e)[:70]}", flush=True)
                if dev == "cuda":
                    torch.cuda.empty_cache()

        # ---- AURA-Net, flagged because its recipe genuinely differs ----
        # It consumes the sub-windows jointly rather than as independent samples, so it cannot share
        # the zoo's flattened recipe. The row is marked ``recipe: auranet_native`` so the difference is
        # visible in the table rather than buried; without that flag the comparison would be the exact
        # kind of unmatched contrast this benchmark exists to eliminate.
        # Three arms, because the anchor is the design's distinctive claim and the evidence on it is
        # task-dependent: it ranks AURA-Net 1st of 25 on detection, where a disjoint per-patient
        # reference state genuinely exists, and costs 0.093 AUC on cross-patient scalp forecasting even
        # after the population fallback made it work at all. Reporting both configurations puts that
        # contrast in the table instead of resolving it by choosing a default.
        # Names say what each arm IS. The anchor is off by default now, so the arm that carries it is
        # the one that needs labelling -- it is retained as the ablation that shows the design's
        # distinctive component costing AUC on every task tested.
        for mname, kw in (("auranet_hybrid", dict(d_feat=X.shape[1])), ("auranet_pure", {}),
                          ("auranet_anchor", dict(anchor=True))):
            if mname in task_res:
                continue
            try:
                from bsf.models.auranet_train import AURANetClassifier
                oof = np.zeros(len(y)); npar = 0
                for tr, te in folds:
                    clf = AURANetClassifier(fs=sf, epochs=args.epochs, device=dev, seed=0, **kw)
                    fe = X if kw else None
                    clf.fit(W[tr], y[tr], g[tr], feats=fe[tr] if fe is not None else None)
                    oof[te] = clf.predict_proba(W[te], g[te],
                                                feats=fe[te] if fe is not None else None)
                    npar = sum(p.numel() for p in clf.model.parameters())
                    del clf
                    if dev == "cuda":
                        torch.cuda.empty_cache()
                task_res[mname] = {"panel": C.full_panel(y, oof, groups=g, n_boot=400),
                                   "n_params": npar,
                                   "input": "raw+features" if kw else "raw",
                                   "recipe": "auranet_native", "family": "repo_ours"}
                _keep_oof(mname, oof)
                resume.update_unit(f"unified_{unit}", mname, task_res[mname])
                d = task_res[mname]["panel"]["discrimination"]
                print(f"    {mname:22s} AUC {d['auc']:.3f} AUPRC {d['auprc']:.3f} "
                      f"({npar:,} params)", flush=True)
            except Exception as e:  # noqa: BLE001
                task_res[mname] = {"error": f"{type(e).__name__}: {str(e)[:100]}"}
                print(f"    {mname:22s} FAILED {str(e)[:70]}", flush=True)
        task_res["_skipped_models"] = skipped
        task_res["_meta"] = {"n_clips": int(n_clip), "n_subjects": int(len(np.unique(g))),
                             "n_channels": int(n_ch), "n_times": int(n_t), "sfreq": sf}
        results[task] = task_res

    R = {"protocol": "DEV subjects only; subject-grouped CV; one training recipe for every model",
         "blind_seal": seal["sha256"], "tasks": results}
    save_json(R, out / "results.json")

    # a flat leaderboard, because a nested JSON is not something anyone reads
    rows = ["task,model,family,input,recipe,n_params,auc,auc_lo,auc_hi,auprc,auprc_skill,"
            "brier_skill,ece,sens_at_spec95,net_benefit_max"]
    for task, tr_ in results.items():
        if not isinstance(tr_, dict):
            continue
        for m, r in sorted(tr_.items(), key=lambda kv: -(kv[1].get("panel", {})
                           .get("discrimination", {}).get("auc", 0) if isinstance(kv[1], dict) else 0)):
            if not isinstance(r, dict) or "panel" not in r:
                continue
            pn = r["panel"]; d = pn["discrimination"]; cal = pn["calibration"]
            ci = pn.get("ci", {}).get("auc", {}) or {}
            nb = pn["decision"]["net_benefit"]
            # net benefit relative to the better of the two default strategies, at the threshold where
            # the model helps most: a model that never beats "treat all" has no decision value at all
            nbmax = max((v["model"] - max(v["treat_all"], v["treat_none"]) for v in nb.values()),
                        default=0.0)
            rows.append(f"{task},{m},{r.get('family','')},{r.get('input','')},"
                        f"{r.get('recipe','')},{r.get('n_params',0)},{d['auc']},"
                        f"{ci.get('lo','')},{ci.get('hi','')},{d['auprc']},{d['auprc_skill']},"
                        f"{cal['brier_skill']},{cal['ece']},{d['sens_at_spec95']},{round(nbmax, 5)}")
    (out / "leaderboard.csv").write_text("\n".join(rows))
    print(f"\n-> {out}  ({len(rows) - 1} model-task rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
