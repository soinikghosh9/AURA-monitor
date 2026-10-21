#!/usr/bin/env python
"""Networks in continuous monitoring, with and without a refreshed baseline (development patients).

Companion to ``run_stream_replication.py``, on the same streams and the same scored windows (the first
30 min of non-ictal windows of each stream are removed as in that analysis). The question is whether the
refreshed baseline that helps AURA also helps the networks, so the comparison is like for like.

Arms
  none       the network as benchmarked (per-window standardization only)
  refreshed  input-level counterpart of AURA's refreshed baseline: each scored window is mapped by
             R^{-1/2}, R the mean spatial covariance of the stream's windows from 35 to 5 min before it
             (Euclidean alignment to a moving reference, no labels); training patients are aligned to
             the covariance of their earliest third of negative clips (as in the early-baseline runs)

Networks: EEGNet, SCCNet, BIOT (random initialization) and the repository CNN, the four networks of the
early-baseline analysis, under the benchmark recipe. To keep the cost at 24 trainings, the development
patients are split into three patient-grouped folds (sorted, round-robin); each fold's streams are
scored by models trained on the other two folds' clips. AURA (both heads, none and refreshed) is refitted
on the same folds, so every contrast is paired on identical windows. Resumable per (model, arm, fold).

-> results/stream_nets/{results.json, preds.npz, units/*.npz}
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

OUT = REPO / "results" / "stream_nets"   # --out overrides (smoke tests)
NETS = ["EEGNet", "SCCNet", "BIOT", "deep_cnn_raw"]
N_FOLDS = 3


def _covs(X):
    X = X.astype(np.float32)
    return np.einsum("nct,ndt->ncd", X, X) / X.shape[-1]


def refreshed_ea(X, t, start, span_s, gap_s, min_ref=10):
    """Each window i >= start mapped by the inverse square root of the mean covariance of the windows
    whose start lies in [t_i - gap - span, t_i - gap]; identity when fewer than ``min_ref`` exist."""
    from bench_data import _inv_sqrt
    C = _covs(X).astype(np.float64)
    cs = np.concatenate([np.zeros((1,) + C.shape[1:]), np.cumsum(C, 0)])
    lo = np.searchsorted(t, t - gap_s - span_s, side="left")
    hi = np.searchsorted(t, t - gap_s, side="right")
    out = np.empty((len(X) - start,) + X.shape[1:], np.float32)
    for k, i in enumerate(range(start, len(X))):
        n = hi[i] - lo[i]
        if n >= min_ref:
            P = _inv_sqrt((cs[hi[i]] - cs[lo[i]]) / n).astype(np.float32)
            out[k] = P @ X[i].astype(np.float32)
        else:
            out[k] = X[i]
    return out


def predict(module, X, device, bs=128):
    import torch
    module.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.tensor(np.asarray(X[i:i + bs], np.float32), device=device)
            out.append(torch.softmax(module(xb), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def load_streams(subjects, max_hours, montage="native"):
    """The replication's cached streams, cut after the fixed baseline exactly as there."""
    from run_stream_replication import CACHE, RULE
    n_base = int(round(RULE["fixed_baseline_min"] * 60 / RULE["stride_s"]))
    out = {}
    for s in subjects:
        f = CACHE / f"streamrep_{s}_h{int(max_hours)}{'' if montage == 'native' else '_' + montage}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        o = np.argsort(d["t0s"], kind="stable")
        y, t = d["y"][o], d["t0s"][o]
        neg = np.where(y == 0)[0]
        if len(neg) < n_base:
            continue
        cut = neg[n_base - 1] + 1
        if y[cut:].sum() == 0:
            continue
        out[s] = {"order": o, "cut": cut, "y": y, "t": t, "file": f}
    return out


def main() -> int:
    import torch
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from bench_data import euclidean_align, load_task
    from run_baseline_placement import early_mask
    from run_final_stats import signflip
    from run_stream_replication import RULE, _rolling, _zr, events_at, smooth
    from run_streaming_monitor import _stream_features, _train_window_features
    from bsf.models import zoo
    from bsf.utils.compute import resolve_device
    from bsf.utils.io import save_json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default=",".join(NETS))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--micro-bs", type=int, default=16)
    ap.add_argument("--out", default="stream_nets")
    a = ap.parse_args()
    global OUT
    OUT = REPO / "results" / a.out
    dev = resolve_device("auto")
    (OUT / "units").mkdir(parents=True, exist_ok=True)
    fs, C_, T_ = RULE["fs"], RULE["channels"], 2000
    span, gap = RULE["rolling_min"] * 60, RULE["gap_min"] * 60

    W, y, g, ev, _ = load_task("detect", "dev")
    W = W[:, :, :C_, :T_]
    n, k = W.shape[:2]
    base = early_mask(y, g, ev)
    no_ref = [s for s in np.unique(g) if not ((g == s) & base).any()]
    keep = ~base & ~np.isin(g, no_ref)
    ref_W = {s: W[(g == s) & base] for s in np.unique(g) if s not in no_ref}
    Ftr = _train_window_features(W, fs)
    Fz = np.empty_like(Ftr)
    for s in np.unique(g):
        m = g == s
        r = Ftr[m & base].reshape(-1, Ftr.shape[-1])
        Fz[m] = _zr(Ftr[m], r) if len(r) else np.nan
    subjects = sorted(np.unique(g))
    fold_of = {s: i % N_FOLDS for i, s in enumerate(subjects)}
    streams = load_streams(subjects, RULE["max_hours"])
    print(f"=== stream nets: {len(streams)} streams, {N_FOLDS} folds, device {dev} ===", flush=True)

    def stream_arrays(s):
        st = streams[s]
        d = np.load(st["file"])
        X = d["X"][st["order"]][:, :C_, :T_]
        return X

    # ---- AURA on the same folds (CPU) ----
    P = {}
    heads = {"feat_gbm": lambda: GradientBoostingClassifier(n_estimators=200, random_state=0),
             "feat_logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(
                 max_iter=2000, class_weight="balanced"))}
    Fst = {}
    for s, st in streams.items():
        Xfull = np.load(st["file"])["X"]      # as cached: the feature cache is keyed on this array
        F = _stream_features({"subject": s, "X": Xfull}, "detect", fs, C_, T_)[st["order"]]
        del Xfull
        Fst[s] = {"none": F[st["cut"]:], "refreshed": _rolling(F, st["t"], st["cut"], span, gap,
                                                              RULE["min_ref_windows"])}
    for f in range(N_FOLDS):
        tr = keep & np.array([fold_of[s] != f for s in g])
        te = [s for s in streams if fold_of[s] == f]
        for m, mk in heads.items():
            for arm, Xf in (("none", Ftr), ("refreshed", Fz)):
                clf = mk().fit(Xf[tr].reshape(-1, Ftr.shape[-1]), np.repeat(y[tr], k))
                for s in te:
                    P.setdefault(m, {}).setdefault(arm, {})[s] = clf.predict_proba(Fst[s][arm])[:, 1]
    print("  AURA refitted on the folds", flush=True)

    # ---- networks (GPU), resumable per unit ----
    W_none = W[keep]; y_k, g_k = y[keep], g[keep]
    del W
    W_ea = euclidean_align(W_none, g_k, ref_W)
    nets = [m.strip() for m in a.models.split(",") if m.strip()]
    for m in nets:
        for arm in ("none", "refreshed"):
            for f in range(N_FOLDS):
                uf = OUT / "units" / f"{m}_{arm}_f{f}.npz"
                te = [s for s in streams if fold_of[s] == f]
                if uf.exists():
                    d = np.load(uf)
                    for s in te:
                        P.setdefault(m, {}).setdefault(arm, {})[s] = d[s]
                    print(f"  (resume) {m} {arm} fold {f}", flush=True)
                    continue
                trm = np.array([fold_of[s] != f for s in g_k])
                Xtr = (W_ea if arm == "refreshed" else W_none)[trm]
                spec = zoo.build(m, C_, T_, fs, device=dev, seed=0)
                zoo.fit_predict(spec, Xtr.reshape(-1, C_, T_), np.repeat(y_k[trm], k), Xtr[:1, 0], dev,
                                epochs=a.epochs, micro_bs=a.micro_bs, seed=0)
                del Xtr
                preds = {}
                for s in te:
                    st = streams[s]
                    X = stream_arrays(s)
                    Xs = (refreshed_ea(X, st["t"], st["cut"], span, gap, RULE["min_ref_windows"])
                          if arm == "refreshed" else X[st["cut"]:])
                    preds[s] = predict(spec.module, Xs, dev)
                    P.setdefault(m, {}).setdefault(arm, {})[s] = preds[s]
                    del X, Xs
                np.savez(uf, **preds)
                print(f"  {m} {arm} fold {f}: " + ", ".join(
                    f"{s} {roc_auc_score(streams[s]['y'][streams[s]['cut']:], preds[s]):.3f}" for s in te),
                    flush=True)
                del spec
                if dev == "cuda":
                    torch.cuda.empty_cache()

    # pre-specified combination: the mean of boosted AURA and of the networks' average, per arm
    if "feat_gbm" in P and nets and all(m in P for m in nets):
        P["ensemble"] = {arm: {s: 0.5 * P["feat_gbm"][arm][s] + 0.5 * np.mean([P[m][arm][s] for m in nets], 0)
                               for s in streams} for arm in ("none", "refreshed")}

    # ---- summary ----
    subj = sorted(streams)
    Y = {s: streams[s]["y"][streams[s]["cut"]:] for s in subj}
    T = {s: streams[s]["t"][streams[s]["cut"]:] for s in subj}
    R = {"n_streams": len(subj), "folds": {s: fold_of[s] for s in subj}, "models": {}}
    from bsf.eval import clinical as C
    for m in P:
        res = {"per_stream_auc": {s: {arm: round(float(roc_auc_score(Y[s], P[m][arm][s])), 4)
                                      for arm in P[m]} for s in subj}}
        Yc = np.concatenate([Y[s] for s in subj])
        for arm in P[m]:
            p = np.concatenate([P[m][arm][s] for s in subj])
            res[arm] = {"auc": round(float(roc_auc_score(Yc, p)), 4)}
            for sm in (False, True):
                cau = nev = fa = 0; hrs = 0.0; lat = []
                for s in subj:
                    oth = [smooth(P[m][arm][o], T[o]) if sm else P[m][arm][o] for o in subj if o != s]
                    thr = float(np.quantile(np.concatenate(oth), 0.90))
                    e = events_at(Y[s], smooth(P[m][arm][s], T[s]) if sm else P[m][arm][s], T[s], thr,
                                  RULE["window_s"])
                    cau += e["caught"]; nev += e["n_events"]; fa += e["false_alarms"]; lat += e["latency_s"]
                    hrs += len(Y[s]) * RULE["window_s"] / 3600
                res[arm]["events_smoothed" if sm else "events_raw"] = {
                    "sensitivity": round(cau / max(nev, 1), 4), "caught": cau, "n_events": nev,
                    "fa_per_24h": round(24 * fa / hrs, 2),
                    "median_latency_s": round(float(np.median(lat)), 1) if lat else None}
        if "refreshed" in P[m]:
            d = [res["per_stream_auc"][s]["refreshed"] - res["per_stream_auc"][s]["none"] for s in subj]
            res["delta_refreshed"] = {"mean": round(float(np.mean(d)), 4), "n_improved": int(sum(x > 0 for x in d)),
                                      "n": len(d), "p_signflip": signflip(d)}
        R["models"][m] = res
        print(f"[{m}] AUC none {res['none']['auc']:.3f} refreshed {res.get('refreshed', {}).get('auc', float('nan')):.3f}"
              f" delta {res.get('delta_refreshed', {}).get('mean')}", flush=True)
    # every model against refreshed boosted AURA, per stream
    ref = R["models"]["feat_gbm"]["per_stream_auc"]
    R["vs_aura_refreshed"] = {}
    for m in R["models"]:
        for arm in ("none", "refreshed"):
            if m == "feat_gbm" and arm == "refreshed":
                continue
            d = [R["models"][m]["per_stream_auc"][s][arm] - ref[s]["refreshed"] for s in subj]
            R["vs_aura_refreshed"][f"{m}|{arm}"] = {"mean": round(float(np.mean(d)), 4),
                                                    "n_better": int(sum(x > 0 for x in d)),
                                                    "p_signflip": signflip(d)}
    np.savez_compressed(OUT / "preds.npz", _y=np.concatenate([Y[s] for s in subj]).astype(np.int8),
                        _groups=np.concatenate([[s] * len(Y[s]) for s in subj]),
                        _t=np.concatenate([T[s] - T[s][0] for s in subj]).astype(np.float32),
                        **{f"{m}|{arm}": np.concatenate([P[m][arm][s] for s in subj]).astype(np.float32)
                           for m in P for arm in P[m]})
    save_json(R, OUT / "results.json")
    print(f"-> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
