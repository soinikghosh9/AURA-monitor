#!/usr/bin/env python
"""AURA-Ens on continuous EEG: common montage (R1), patient-baseline normalization (R2/R3) and the
stacked ensemble with clinical event scoring (R4) of docs/plans/2026-10-05-aura-monitor.md.

Same 25 development streams, scored windows and three patient-grouped folds as ``run_stream_tta.py``;
``--montage db18`` builds every input (training clips and streams) on the 18-channel double banana.
Per fold, on the other folds' patients:

  features   AURA's 18 features, boosted and logistic heads, raw (none) or robust-z-scored by the
             patient's reference (refreshed: training patients by their earliest-negative clips, streams
             by the windows 35 to 5 min earlier)
  networks   trained once (weights banked); scored as trained (none) and through a boosted head on the
             frozen embedding, raw or refreshed (probe_gbm_none / probe_gbm_refreshed: post-hoc
             representation-level adaptation)
  pbn        the same backbone fine-tuned with patient-baseline normalization inside training
             (bsf.models.pbn), scored with the stream's refreshed reference
  pbn_adv    pbn with a gradient-reversal patient adversary on the normalized embedding
  AURA-Net   hybrid, trained and scored with raw or refreshed features
  ensembles  mean of probabilities and a stacked logistic on logits fitted on the other folds' streams,
             over the post-hoc members (post), the PBN members (pbn) or both (all)

Every arm gets per-stream and pooled AUC, paired contrasts (sign-flip over streams), and SzCORE-style
event scores (sensitivity, precision, F1, false alarms per 24 h, latency) at false-alarm budgets of 2, 6
and 24 per 24 h, each threshold chosen on the other folds' streams.

-> results/<out>/{results.json, preds.npz, units/}
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

NETS = ["deep_cnn_raw", "SCCNet", "ATCNet", "ContraWR", "BIOT", "EEGNet"]
BUDGETS = (2.0, 6.0, 24.0)
SMOOTH_S = (8.0, 40.0, 120.0)          # causal smoothing spans considered (8 s = none)


def main() -> int:
    import torch
    from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from bench_data import load_task
    from bsf.eval import szcore
    from bsf.models import pbn as PBN
    from bsf.models import zoo
    from bsf.models.auranet_train import AURANetClassifier
    from bsf.utils.compute import resolve_device
    from bsf.utils.io import save_json
    from run_baseline_placement import early_mask
    from run_final_stats import signflip
    from run_stream_nets import N_FOLDS, load_streams, predict
    from run_stream_replication import RULE, _rolling, _zr, events_at, smooth
    from bsf.monitor.embedding import embed
    from run_stream_tta import identity_label
    from run_streaming_monitor import _stream_features, _train_window_features

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--montage", default="db18", choices=["native", "db18"])
    ap.add_argument("--models", default=",".join(NETS))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--pbn-epochs", type=int, default=10)
    ap.add_argument("--adv-weight", type=float, default=0.3)
    ap.add_argument("--micro-bs", type=int, default=16)
    ap.add_argument("--skip-auranet", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--smoke", action="store_true", help="1 epoch, first network only, separate output")
    a = ap.parse_args()
    if a.smoke:
        a.epochs, a.pbn_epochs, a.models, a.out = 1, 1, a.models.split(",")[0], f"aura_ens_{a.montage}_smoke"
    OUT = REPO / "results" / (a.out or f"aura_ens_{a.montage}")
    U = OUT / "units"; U.mkdir(parents=True, exist_ok=True)
    dev = resolve_device("auto")
    fs, T_ = RULE["fs"], 2000
    span, gap, mref = RULE["rolling_min"] * 60, RULE["gap_min"] * 60, RULE["min_ref_windows"]

    W, y, g, ev, _ = load_task("detect", "dev", montage=a.montage)
    C_ = 18 if a.montage == "db18" else RULE["channels"]
    W = W[:, :, :C_, :T_]
    k = W.shape[1]
    base = early_mask(y, g, ev)
    no_ref = [s for s in np.unique(g) if not ((g == s) & base).any()]
    keep = ~base & ~np.isin(g, no_ref)
    Ftr = _train_window_features(W, fs)
    Fz = np.empty_like(Ftr)
    for s in np.unique(g):
        m = g == s
        r = Ftr[m & base].reshape(-1, Ftr.shape[-1])
        Fz[m] = _zr(Ftr[m], r) if len(r) else np.nan
    subjects = sorted(np.unique(g))
    fold_of = {s: i % N_FOLDS for i, s in enumerate(subjects)}
    streams = load_streams(subjects, RULE["max_hours"], a.montage)
    subj = sorted(streams)
    Y = {s: streams[s]["y"][streams[s]["cut"]:] for s in subj}
    T = {s: streams[s]["t"][streams[s]["cut"]:] for s in subj}

    def stream_X(s):
        st = streams[s]
        return np.load(st["file"])["X"][st["order"]][:, :C_, :T_]

    Fst = {}
    for s in subj:
        st = streams[s]
        F = _stream_features({"subject": f"{s}_{a.montage}", "X": np.load(st["file"])["X"]}, "detect", fs, C_, T_)[st["order"]]
        Fst[s] = {"none": F[st["cut"]:], "refreshed": _rolling(F, st["t"], st["cut"], span, gap, mref)}
    print(f"=== AURA-Ens [{a.montage}]: {len(subj)} streams, {N_FOLDS} folds, {C_} channels, device {dev} ===", flush=True)

    P: dict = {}
    E7: dict = {}

    def put(m, arm, s, v):
        P.setdefault(m, {}).setdefault(arm, {})[s] = np.asarray(v, np.float32)

    # ---------------------------------------------------------------- features
    heads_f = {"feat_gbm": lambda: GradientBoostingClassifier(n_estimators=200, random_state=0),
               "feat_logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))}
    for f in range(N_FOLDS):
        uf = U / f"features_f{f}.npz"
        te = [s for s in subj if fold_of[s] == f]
        if uf.exists():
            d = np.load(uf)
            for key in d.files:
                m, arm, s = key.split("|"); put(m, arm, s, d[key])
            continue
        tr = keep & np.array([fold_of[s] != f for s in g])
        out, models = {}, {}
        for m, mk in heads_f.items():
            for arm, Fin in (("none", Ftr), ("refreshed", Fz)):
                X_ = np.nan_to_num(Fin[tr].reshape(-1, Fin.shape[-1])); y_ = np.repeat(y[tr], k)
                clf = mk().fit(X_, y_); models[(m, arm)] = clf
                for s in te:
                    out[f"{m}|{arm}|{s}"] = clf.predict_proba(np.nan_to_num(Fst[s][arm]))[:, 1]
        with open(U / f"features_f{f}.pkl", "wb") as fh:
            pickle.dump(models, fh)
        np.savez(uf, **out)
        for key, v in out.items():
            m, arm, s = key.split("|"); put(m, arm, s, v)
        print(f"  features fold {f} done", flush=True)

    # ---------------------------------------------------------------- networks, post hoc and PBN
    nets = [m.strip() for m in a.models.split(",") if m.strip()]
    for mname in nets:
        for f in range(N_FOLDS):
            te = [s for s in subj if fold_of[s] == f]
            uf = U / f"{mname}_f{f}.npz"
            if uf.exists():
                d = np.load(uf, allow_pickle=True)
                for key in d.files:
                    if key.startswith("E7|"):
                        E7[f"{mname}|f{f}"] = json.loads(str(d[key])); continue
                    arm, s = key.split("|"); put(mname, arm, s, d[key])
                print(f"  (resume) {mname} fold {f}", flush=True)
                continue
            trp = np.array([fold_of[s] != f for s in g])
            tr = keep & trp
            try:
                spec = zoo.build(mname, C_, T_, fs, device=dev, seed=0)
                wf = U / f"{mname}_f{f}.pt"
                if wf.exists():
                    spec.module.load_state_dict(torch.load(wf, map_location=dev))
                else:
                    zoo.fit_predict(spec, W[tr].reshape(-1, C_, T_), np.repeat(y[tr], k), W[tr][:1, 0], dev,
                                    epochs=a.epochs, micro_bs=a.micro_bs, seed=0)
                    torch.save(spec.module.state_dict(), wf)
                mod = spec.module.eval()
                out = {}
                # post-hoc representation-level adaptation
                Etr = embed(mod, W[tr].reshape(-1, C_, T_), dev)
                gtr = np.repeat(g[tr], k); ytr = np.repeat(y[tr], k)
                Etr_z = np.empty_like(Etr)
                for s in np.unique(g[tr]):
                    Eb = embed(mod, W[(g == s) & base].reshape(-1, C_, T_), dev)
                    Etr_z[gtr == s] = _zr(Etr[gtr == s], Eb)
                mk = lambda: HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.1, random_state=0)  # noqa: E731
                h_none, h_ref = mk().fit(Etr, ytr), mk().fit(Etr_z, ytr)
                with open(U / f"{mname}_f{f}_heads.pkl", "wb") as fh:
                    pickle.dump({"none": h_none, "refreshed": h_ref}, fh)
                del Etr, Etr_z
                # PBN and PBN + adversary, fine-tuned from the trained backbone
                ftm = trp & (keep | base)
                pbns = {}
                for arm, adv in (("pbn", 0.0), ("pbn_adv", a.adv_weight)):
                    pf = U / f"{mname}_f{f}_{arm}.pt"
                    net = PBN.PBNNet(copy.deepcopy(mod), PBN.embed_dim(mod, C_, T_, dev),
                                     len(PBN.trainable_patients(g[ftm], base[ftm])) if adv > 0 else 0).to(dev)
                    if pf.exists():
                        net.load_state_dict(torch.load(pf, map_location=dev))
                    else:
                        net = PBN.train_pbn(copy.deepcopy(mod), W[ftm], y[ftm], g[ftm], base[ftm], dev,
                                            epochs=a.pbn_epochs, adv_weight=adv, seed=0)
                        torch.save(net.state_dict(), pf)
                    pbns[arm] = net.eval()
                for s in te:
                    st = streams[s]
                    X = stream_X(s)
                    out[f"none|{s}"] = predict(mod, X[st["cut"]:], dev)
                    Es = embed(mod, X, dev)
                    out[f"probe_gbm_none|{s}"] = h_none.predict_proba(Es[st["cut"]:])[:, 1]
                    out[f"probe_gbm_refreshed|{s}"] = h_ref.predict_proba(
                        _rolling(Es, st["t"], st["cut"], span, gap, mref))[:, 1]
                    for arm, net in pbns.items():
                        Ep = PBN.embed_np(net, X, dev)
                        out[f"{arm}|{s}"] = PBN.classify_np(net, PBN.rolling_reference(Ep, st["t"], st["cut"], span, gap, mref), dev)
                    del X, Es
                    if dev == "cuda":
                        torch.cuda.empty_cache()
                hm = keep & ~trp
                Ec = embed(mod, W[hm].reshape(-1, C_, T_), dev).reshape(hm.sum(), k, -1).mean(1)
                e7 = {"net": identity_label(Ec, g[hm], y[hm]), "aura": identity_label(Ftr[hm].mean(1), g[hm], y[hm]),
                      "n_patients": int(len(np.unique(g[hm]))), "embed_dim": int(Ec.shape[1])}
                E7[f"{mname}|f{f}"] = e7
                np.savez(uf, **out, **{"E7|json": json.dumps(e7)})
                for key, v in out.items():
                    arm, s = key.split("|"); put(mname, arm, s, v)
                print(f"  {mname} fold {f}: " + ", ".join(
                    f"{arm} {np.median([roc_auc_score(Y[s], P[mname][arm][s]) for s in te]):.3f}"
                    for arm in sorted({kk.split('|')[0] for kk in out})) + f" | id acc {e7['net']['id_acc']:.2f}", flush=True)
                del spec, mod, pbns
            except Exception as e:  # noqa: BLE001
                print(f"  {mname} fold {f} FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
            if dev == "cuda":
                torch.cuda.empty_cache()

    # ---------------------------------------------------------------- AURA-Net hybrid
    if not a.skip_auranet:
        for arm, Ftrain in (("none", Ftr), ("refreshed", Fz)):
            for f in range(N_FOLDS):
                te = [s for s in subj if fold_of[s] == f]
                uf = U / f"auranet_hybrid_{arm}_f{f}.npz"
                if uf.exists():
                    d = np.load(uf)
                    for s in te:
                        put("auranet_hybrid", arm, s, d[s])
                    continue
                tr = keep & np.array([fold_of[s] != f for s in g])
                clf = AURANetClassifier(fs=fs, epochs=a.epochs, device=dev, seed=0, d_feat=Ftr.shape[-1])
                clf.fit(W[tr], y[tr], g[tr], feats=np.nan_to_num(Ftrain[tr].mean(1)))
                res = {}
                for s in te:
                    st = streams[s]
                    X = stream_X(s)[st["cut"]:][:, None].astype(np.float32)
                    res[s] = clf.predict_proba(X, np.array([s] * len(X)), feats=np.nan_to_num(Fst[s][arm]))
                    del X
                np.savez(uf, **res)
                try:                                     # banked for the monitor (bsf.monitor.detector)
                    torch.save(clf, U / f"auranet_hybrid_{arm}_f{f}.clf")
                except Exception as e:  # noqa: BLE001
                    print(f"  (auranet not saved: {e})", flush=True)
                for s, v in res.items():
                    put("auranet_hybrid", arm, s, v)
                print(f"  auranet_hybrid {arm} fold {f}: median AUC {np.median([roc_auc_score(Y[s], res[s]) for s in te]):.3f}", flush=True)
                del clf
                if dev == "cuda":
                    torch.cuda.empty_cache()

    # ---------------------------------------------------------------- ensembles
    complete = lambda m, arm: m in P and arm in P[m] and all(s in P[m][arm] for s in subj)  # noqa: E731
    sets = {"post": [("feat_gbm", "refreshed"), ("auranet_hybrid", "refreshed")] + [(m, "probe_gbm_refreshed") for m in nets],
            "pbn": [("feat_gbm", "refreshed"), ("auranet_hybrid", "refreshed")] + [(m, "pbn") for m in nets]}
    sets["all"] = sets["post"] + [(m, "pbn") for m in nets]
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731
    stack_w = {}
    for name, comp in sets.items():
        comp = [c for c in comp if complete(*c)]
        if len(comp) < 2:
            continue
        for s in subj:
            put("ens_mean", name, s, np.mean([P[m][arm][s] for m, arm in comp], 0))
        Xs = {s: np.column_stack([lg(P[m][arm][s]) for m, arm in comp]) for s in subj}
        for f in range(N_FOLDS):
            tr_s = [s for s in subj if fold_of[s] != f]
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1).fit(
                np.concatenate([Xs[s] for s in tr_s]), np.concatenate([Y[s] for s in tr_s]))
            stack_w[f"{name}|f{f}"] = {"components": [f"{m}|{arm}" for m, arm in comp],
                                       "coef": clf.coef_.ravel().round(4).tolist(), "intercept": float(clf.intercept_[0])}
            for s in subj:
                if fold_of[s] == f:
                    put("aura_ens", name, s, clf.predict_proba(Xs[s])[:, 1])

    # ---------------------------------------------------------------- summary
    R = {"montage": a.montage, "n_streams": len(subj), "folds": {s: fold_of[s] for s in subj},
         "stacking": stack_w, "models": {}, "E7": E7, "budgets_per_24h": list(BUDGETS),
         "event_scoring": szcore.SZCORE, "smoothing_candidates_s": list(SMOOTH_S)}
    Ps = {m: {arm: {s: smooth(d[s], T[s]) for s in subj} for arm, d in P[m].items() if all(s in d for s in subj)} for m in P}
    Pss = {sp: {m: {arm: {s: smooth(d[s], T[s], sp) for s in subj} for arm, d in P[m].items() if all(s in d for s in subj)} for m in P}
           for sp in SMOOTH_S}
    ref = {s: roc_auc_score(Y[s], P["feat_gbm"]["refreshed"][s]) for s in subj}
    for m in P:
        res = {}
        for arm, d in P[m].items():
            if not all(s in d for s in subj):
                continue
            per = {s: round(float(roc_auc_score(Y[s], d[s])), 4) for s in subj}
            r = {"auc": round(float(roc_auc_score(np.concatenate([Y[s] for s in subj]), np.concatenate([d[s] for s in subj]))), 4),
                 "median_stream_auc": round(float(np.median(list(per.values()))), 4), "per_stream_auc": per}

            def contrast(other):
                dd = [per[s] - roc_auc_score(Y[s], other[s]) for s in subj]
                return {"mean": round(float(np.mean(dd)), 4), "n_better": int(sum(x > 0 for x in dd)), "p_signflip": signflip(dd)}
            if arm != "none" and "none" in P[m]:
                r["delta_vs_none"] = contrast(P[m]["none"])
            if not (m == "feat_gbm" and arm == "refreshed"):
                r["delta_vs_aura_refreshed"] = contrast(P["feat_gbm"]["refreshed"])
            if arm in ("pbn", "pbn_adv") and "probe_gbm_refreshed" in P[m]:
                r["delta_vs_posthoc"] = contrast(P[m]["probe_gbm_refreshed"])
            if arm == "pbn_adv" and "pbn" in P[m]:
                r["delta_vs_pbn"] = contrast(P[m]["pbn"])
            # events: legacy 10% time-in-warning rule and SzCORE-style budgets, thresholds from other folds
            cau = nev = fa = 0; hrs = 0.0
            for s in subj:
                oth = np.concatenate([Ps[m][arm][o] for o in subj if o != s])
                e = events_at(Y[s], Ps[m][arm][s], T[s], float(np.quantile(oth, 0.9)), RULE["window_s"])
                cau += e["caught"]; nev += e["n_events"]; fa += e["false_alarms"]; hrs += len(Y[s]) * RULE["window_s"] / 3600
            r["legacy_events"] = {"caught": cau, "n_events": nev, "fa_per_24h": round(24 * fa / hrs, 2)}
            r["szcore"] = {}
            for b in BUDGETS:
                counts, spans_used = [], []
                for f in range(N_FOLDS):
                    # post-processing chosen on the other folds: smoothing span, then the budget threshold
                    best = None
                    for sp in SMOOTH_S:
                        trs = [(Y[o], Pss[sp][m][arm][o], T[o]) for o in subj if fold_of[o] != f]
                        thr = szcore.threshold_for_budget(trs, b, RULE["window_s"])
                        f1 = szcore.summarize([szcore.events(y_, p_, t_, thr, RULE["window_s"]) for y_, p_, t_ in trs])["f1"]
                        if best is None or f1 > best[0]:
                            best = (f1, sp, thr)
                    _f1, sp, thr = best
                    spans_used.append(sp)
                    counts += [szcore.events(Y[s], Pss[sp][m][arm][s], T[s], thr, RULE["window_s"]) for s in subj if fold_of[s] == f]
                r["szcore"][str(int(b))] = {**szcore.summarize(counts), "smoothing_s": spans_used}
            res[arm] = r
        R["models"][m] = res
        print(f"[{m}] " + "  ".join(f"{arm} {v['auc']:.3f}/{v['median_stream_auc']:.3f}" for arm, v in res.items()), flush=True)
    np.savez_compressed(OUT / "preds.npz", _y=np.concatenate([Y[s] for s in subj]).astype(np.int8),
                        _groups=np.concatenate([[s] * len(Y[s]) for s in subj]),
                        _t=np.concatenate([T[s] for s in subj]).astype(np.float64),
                        **{f"{m}|{arm}": np.concatenate([P[m][arm][s] for s in subj]).astype(np.float32)
                           for m in P for arm in P[m] if all(s in P[m][arm] for s in subj)})
    save_json(R, OUT / "results.json")
    print(f"-> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
