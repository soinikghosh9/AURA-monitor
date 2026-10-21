#!/usr/bin/env python
"""Test-time adaptation of the networks in continuous EEG, and what their representations encode.

Same 25 development streams, scored windows, three patient-grouped folds and training recipe as
``run_stream_nets.py`` (so every contrast is paired with its rows). Per network and fold the model is
trained once on the clips of the other folds' patients (weights banked), then scored on the fold's
streams under each adaptation, none of which uses a label:

  E1  none            the network as trained
      adabn_stream    AdaBN: BatchNorm statistics re-estimated from all of the stream's windows
                      (transductive; the textbook form)
      adabn_refreshed BatchNorm statistics re-estimated every 5 min from the stream's windows 35 to
                      5 min earlier (the refreshed baseline, applied inside the network)
      tent            TENT: BatchNorm/LayerNorm affine parameters updated online by entropy
                      minimisation over consecutive 32-window batches, reset per stream
  E2  probe_none      logistic / boosted head on the frozen embedding (input to the task head)
      probe_refreshed the same head on embeddings robust-z-scored by the stream's windows 35 to 5 min
                      earlier (training patients: by their earliest-negative clips) -- AURA's
                      normalization applied to the learned representation
  E3  AURA-Net (hybrid, gated fusion of the 18 features) trained with raw or baseline-normalized
      features and scored with raw or refreshed features; a stacked ensemble of every model's output
      fitted on the other folds' streams
  E7  in each fold's held-out patients, how well the frozen embedding identifies the patient and the
      clinical label, against the AURA features of the same clips (identification accuracy, and the
      share of variance explained by patient and by label in the first 20 principal components)

-> results/stream_tta/{results.json, preds.npz, units/}
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

OUT = REPO / "results" / "stream_tta"
NETS = ["EEGNet", "SCCNet", "BIOT", "deep_cnn_raw"]
BLOCK = 38                      # windows per AdaBN refresh (about 5 min of 8-s windows)


# ------------------------------------------------------------------ network helpers
def _head(module):
    import torch.nn as nn
    h = getattr(module, "final_layer", None)
    if h is not None:
        return h
    lin = [m for m in module.modules() if isinstance(m, nn.Linear)]
    return lin[-1]


def embed(module, X, device, bs=128):
    """Input to the task head, flattened (or channel-averaged when wider than 2048)."""
    import torch
    module.eval()
    grabbed = []

    def hook(m, inp, out):
        z = inp[0].detach().float()
        if z.ndim > 2:
            z = z.reshape(z.shape[0], -1) if z[0].numel() <= 2048 else z.reshape(z.shape[0], z.shape[1], -1).mean(-1)
        grabbed.append(z.cpu().numpy())
    h = _head(module).register_forward_hook(hook)
    try:
        with torch.no_grad():
            for i in range(0, len(X), bs):
                module(torch.tensor(np.asarray(X[i:i + bs], np.float32), device=device))
    finally:
        h.remove()
    return np.concatenate(grabbed)


def _bns(module):
    import torch.nn as nn
    return [m for m in module.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


def _set_bn_stats(module, Xref, device, bs=64):
    import torch
    for m in _bns(module):
        m.reset_running_stats(); m.momentum = None
    module.eval()
    for m in _bns(module):
        m.train()
    with torch.no_grad():
        for i in range(0, len(Xref), bs):
            xb = np.asarray(Xref[i:i + bs], np.float32)
            if len(xb) < 2:
                continue
            module(torch.tensor(xb, device=device))
    module.eval()


def adabn_stream(module, X, Xs, device):
    from run_stream_nets import predict
    m = copy.deepcopy(module)
    _set_bn_stats(m, X, device)
    return predict(m, Xs, device)


def adabn_refreshed(module, X, t, cut, span_s, gap_s, device, min_ref=10):
    from run_stream_nets import predict
    m = copy.deepcopy(module)
    base = {k: v.clone() for k, v in m.state_dict().items()}
    out = np.empty(len(X) - cut, np.float32)
    lo = np.searchsorted(t, t - gap_s - span_s, side="left")
    hi = np.searchsorted(t, t - gap_s, side="right")
    for b0 in range(cut, len(X), BLOCK):
        b1 = min(b0 + BLOCK, len(X))
        r0, r1 = lo[b0], hi[b0]
        if r1 - r0 >= min_ref:
            _set_bn_stats(m, X[r0:r1], device)
        else:
            m.load_state_dict(base)
        out[b0 - cut:b1 - cut] = predict(m, X[b0:b1], device)
    return out


def tent(module, Xs, device, bs=32, lr=1e-4):
    import torch
    import torch.nn as nn
    m = copy.deepcopy(module)
    m.eval()
    params = []
    for mod in m.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            mod.train(); mod.track_running_stats = False; mod.running_mean = None; mod.running_var = None
            params += [p for p in (mod.weight, mod.bias) if p is not None]
        elif isinstance(mod, nn.LayerNorm):
            params += [p for p in (mod.weight, mod.bias) if p is not None]
    for p in m.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    if not params:
        return None
    opt = torch.optim.Adam(params, lr=lr)
    out = []
    for i in range(0, len(Xs), bs):
        xb = np.asarray(Xs[i:i + bs], np.float32)
        x = torch.tensor(xb, device=device)
        if len(xb) < 2:                                   # a lone window cannot form batch statistics
            with torch.no_grad():
                m.eval(); out.append(torch.softmax(m(x), 1)[:, 1].cpu().numpy()); continue
        logits = m(x)
        prob = torch.softmax(logits, 1)
        ent = -(prob * torch.log(prob + 1e-8)).sum(1).mean()
        opt.zero_grad(); ent.backward(); opt.step()
        out.append(prob[:, 1].detach().cpu().numpy())
    return np.concatenate(out)


# ------------------------------------------------------------------ representation analysis (E7)
def identity_label(Z, pat, y, seed=0):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    Z = np.nan_to_num(np.asarray(Z, float))
    k = min(20, Z.shape[1], len(Z) - 1)
    P = PCA(k, random_state=seed).fit_transform(StandardScaler().fit_transform(Z))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    pid = cross_val_predict(clf, P, pat, cv=cv)
    acc = float(np.mean(pid == pat))
    pl = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")),
                           P, y, cv=cv, method="predict_proba")[:, 1]
    # variance shares per component: patient (between-patient), label (between-class within patient)
    sp, sl = [], []
    for j in range(P.shape[1]):
        v = P[:, j]; tot = v.var() + 1e-12
        mp = np.array([v[pat == s].mean() for s in pat])
        sp.append(mp.var() / tot)
        r = v - mp
        ml = np.zeros_like(r)
        for s in np.unique(pat):
            for c in (0, 1):
                mm = (pat == s) & (y == c)
                if mm.any():
                    ml[mm] = r[mm].mean()
        sl.append(ml.var() / tot)
    return {"id_acc": round(acc, 4), "chance": round(1 / len(np.unique(pat)), 4),
            "label_auc_cv": round(float(roc_auc_score(y, pl)), 4),
            "patient_share": round(float(np.median(sp)), 4), "label_share": round(float(np.median(sl)), 4)}


def main() -> int:
    import torch
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from bench_data import load_task
    from bsf.models import zoo
    from bsf.models.auranet_train import AURANetClassifier
    from bsf.utils.compute import resolve_device
    from bsf.utils.io import save_json
    from run_baseline_placement import early_mask
    from run_final_stats import signflip
    from run_stream_nets import N_FOLDS, load_streams, predict
    from run_stream_replication import RULE, _rolling, _zr, events_at, smooth
    from run_streaming_monitor import _stream_features, _train_window_features

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default=",".join(NETS))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--micro-bs", type=int, default=16)
    ap.add_argument("--skip-auranet", action="store_true")
    ap.add_argument("--out", default="stream_tta")
    a = ap.parse_args()
    global OUT
    OUT = REPO / "results" / a.out
    dev = resolve_device("auto")
    U = OUT / "units"; U.mkdir(parents=True, exist_ok=True)
    fs, C_, T_ = RULE["fs"], RULE["channels"], 2000
    span, gap = RULE["rolling_min"] * 60, RULE["gap_min"] * 60

    W, y, g, ev, _ = load_task("detect", "dev")
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
    streams = load_streams(subjects, RULE["max_hours"])
    subj = sorted(streams)
    Y = {s: streams[s]["y"][streams[s]["cut"]:] for s in subj}
    T = {s: streams[s]["t"][streams[s]["cut"]:] for s in subj}

    def stream_X(s):
        st = streams[s]
        return np.load(st["file"])["X"][st["order"]][:, :C_, :T_]

    Fst = {}
    for s in subj:
        st = streams[s]
        F = _stream_features({"subject": s, "X": np.load(st["file"])["X"]}, "detect", fs, C_, T_)[st["order"]]
        Fst[s] = {"none": F[st["cut"]:], "refreshed": _rolling(F, st["t"], st["cut"], span, gap, RULE["min_ref_windows"])}
    print(f"=== stream TTA: {len(subj)} streams, {N_FOLDS} folds, device {dev} ===", flush=True)

    P = {}            # P[model][arm][stream] -> scores on the scored windows
    E7 = {}
    nets = [m.strip() for m in a.models.split(",") if m.strip()]
    for mname in nets:
        for f in range(N_FOLDS):
            te = [s for s in subj if fold_of[s] == f]
            uf = U / f"{mname}_f{f}.npz"
            if uf.exists():
                d = np.load(uf, allow_pickle=True)
                for key in d.files:
                    if key.startswith("E7|"):
                        continue
                    arm, s = key.split("|")
                    P.setdefault(mname, {}).setdefault(arm, {})[s] = d[key]
                E7[f"{mname}|f{f}"] = json.loads(str(d["E7|json"])) if "E7|json" in d.files else None
                print(f"  (resume) {mname} fold {f}", flush=True)
                continue
            trp = np.array([fold_of[s] != f for s in g])
            tr = keep & trp
            spec = zoo.build(mname, C_, T_, fs, device=dev, seed=0)
            wf = U / f"{mname}_f{f}.pt"
            if wf.exists():
                spec.module.load_state_dict(torch.load(wf, map_location=dev))
            else:
                zoo.fit_predict(spec, W[tr].reshape(-1, C_, T_), np.repeat(y[tr], k), W[tr][:1, 0], dev,
                                epochs=a.epochs, micro_bs=a.micro_bs, seed=0)
                torch.save(spec.module.state_dict(), wf)
            mod = spec.module.eval()
            has_bn = len(_bns(mod)) > 0
            out = {}
            # E2 training embeddings: training patients' clips (sub-windows) and their baseline clips
            Etr = embed(mod, W[tr].reshape(-1, C_, T_), dev)
            gtr = np.repeat(g[tr], k); ytr = np.repeat(y[tr], k)
            Etr_z = np.empty_like(Etr)
            for s in np.unique(g[tr]):
                Eb = embed(mod, W[(g == s) & base].reshape(-1, C_, T_), dev)
                Etr_z[gtr == s] = _zr(Etr[gtr == s], Eb)
            heads = {"lr": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced")),
                     "gbm": lambda: HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.1,
                                                                   random_state=0)}
            fitted = {(h, arm): mk().fit(Etr if arm == "none" else Etr_z, ytr)
                      for h, mk in heads.items() for arm in ("none", "refreshed")}
            del Etr, Etr_z
            for s in te:
                st = streams[s]
                X = stream_X(s)
                Xs = X[st["cut"]:]
                out[f"none|{s}"] = predict(mod, Xs, dev)
                if has_bn:
                    out[f"adabn_stream|{s}"] = adabn_stream(mod, X, Xs, dev)
                    out[f"adabn_refreshed|{s}"] = adabn_refreshed(mod, X, st["t"], st["cut"], span, gap, dev)
                tt = tent(mod, Xs, dev)
                if tt is not None:
                    out[f"tent|{s}"] = tt
                Es = embed(mod, X, dev)
                Ez = _rolling(Es, st["t"], st["cut"], span, gap, RULE["min_ref_windows"])
                for h in heads:
                    out[f"probe_{h}_none|{s}"] = fitted[(h, "none")].predict_proba(Es[st["cut"]:])[:, 1]
                    out[f"probe_{h}_refreshed|{s}"] = fitted[(h, "refreshed")].predict_proba(Ez)[:, 1]
                del X, Xs, Es, Ez
                if dev == "cuda":
                    torch.cuda.empty_cache()
            # E7: held-out patients' clips, embedding vs AURA features
            hm = keep & ~trp
            Ec = embed(mod, W[hm].reshape(-1, C_, T_), dev).reshape(hm.sum(), k, -1).mean(1)
            e7 = {"net": identity_label(Ec, g[hm], y[hm]), "aura": identity_label(Ftr[hm].mean(1), g[hm], y[hm]),
                  "n_patients": int(len(np.unique(g[hm]))), "embed_dim": int(Ec.shape[1])}
            E7[f"{mname}|f{f}"] = e7
            np.savez(uf, **out, **{"E7|json": json.dumps(e7)})
            for key, v in out.items():
                arm, s = key.split("|")
                P.setdefault(mname, {}).setdefault(arm, {})[s] = v
            print(f"  {mname} fold {f}: " + ", ".join(
                f"{arm} {np.median([roc_auc_score(Y[s], P[mname][arm][s]) for s in te]):.3f}"
                for arm in sorted({kk.split('|')[0] for kk in out})) +
                  f" | id acc net {e7['net']['id_acc']:.2f} aura {e7['aura']['id_acc']:.2f} (chance {e7['net']['chance']:.2f})",
                  flush=True)
            del spec, mod
            if dev == "cuda":
                torch.cuda.empty_cache()

    # ---- E3: AURA-Net hybrid with raw or (refreshed) baseline-normalized features ----
    if not a.skip_auranet:
        for arm, Ftrain in (("none", Ftr), ("refreshed", Fz)):
            for f in range(N_FOLDS):
                te = [s for s in subj if fold_of[s] == f]
                uf = U / f"auranet_hybrid_{arm}_f{f}.npz"
                if uf.exists():
                    d = np.load(uf)
                    for s in te:
                        P.setdefault("auranet_hybrid", {}).setdefault(arm, {})[s] = d[s]
                    print(f"  (resume) auranet_hybrid {arm} fold {f}", flush=True)
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
                for s, v in res.items():
                    P.setdefault("auranet_hybrid", {}).setdefault(arm, {})[s] = v
                print(f"  auranet_hybrid {arm} fold {f}: median AUC "
                      f"{np.median([roc_auc_score(Y[s], res[s]) for s in te]):.3f}", flush=True)
                del clf
                if dev == "cuda":
                    torch.cuda.empty_cache()

    # ---- the paired arms of run_stream_nets.py (AURA, input-level refreshed EA, mean ensemble) ----
    nets_res = REPO / "results" / "stream_nets" / "preds.npz"
    if nets_res.exists():
        with np.load(nets_res, allow_pickle=True) as z:
            G = z["_groups"].astype(str)
            for key in z.files:
                if key.startswith("_"):
                    continue
                m, arm = key.split("|")
                arm2 = {"refreshed": "ea_refreshed"}.get(arm, arm) if m in NETS else arm
                if m in NETS and arm == "none":
                    continue                               # this script's own none arm is the reference
                for s in subj:
                    P.setdefault(m, {}).setdefault(arm2, {})[s] = z[key][G == s]

    # ---- stacked ensemble: logistic on logits of the refreshed/adapted arms, fitted on other folds ----
    comp = [("feat_gbm", "refreshed")] + [(m, "ea_refreshed") for m in NETS] + [("auranet_hybrid", "refreshed")]
    comp = [c for c in comp if c[0] in P and c[1] in P[c[0]] and all(s in P[c[0]][c[1]] for s in subj)]
    if len(comp) >= 2:
        lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731
        Xs = {s: np.column_stack([lg(P[m][arm][s]) for m, arm in comp]) for s in subj}
        st_out = {}
        for f in range(N_FOLDS):
            tr_s = [s for s in subj if fold_of[s] != f]
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0).fit(
                np.concatenate([Xs[s] for s in tr_s]), np.concatenate([Y[s] for s in tr_s]))
            for s in subj:
                if fold_of[s] == f:
                    st_out[s] = clf.predict_proba(Xs[s])[:, 1]
        P.setdefault("stacked", {})["refreshed"] = st_out

    # ---- summary ----
    R = {"n_streams": len(subj), "stacked_components": [f"{m}|{arm}" for m, arm in comp], "models": {}, "E7": E7}
    Yc = np.concatenate([Y[s] for s in subj])
    ref = {s: roc_auc_score(Y[s], P["feat_gbm"]["refreshed"][s]) for s in subj} if "feat_gbm" in P else None
    for m in P:
        res = {}
        for arm, d in P[m].items():
            if not all(s in d for s in subj):
                continue
            per = {s: round(float(roc_auc_score(Y[s], d[s])), 4) for s in subj}
            pool = np.concatenate([d[s] for s in subj])
            cau = nev = fa = 0; hrs = 0.0
            for s in subj:
                oth = np.concatenate([smooth(d[o], T[o]) for o in subj if o != s])
                e = events_at(Y[s], smooth(d[s], T[s]), T[s], float(np.quantile(oth, 0.9)), RULE["window_s"])
                cau += e["caught"]; nev += e["n_events"]; fa += e["false_alarms"]; hrs += len(Y[s]) * RULE["window_s"] / 3600
            r = {"auc": round(float(roc_auc_score(Yc, pool)), 4), "median_stream_auc": round(float(np.median(list(per.values()))), 4),
                 "per_stream_auc": per, "caught": cau, "n_events": nev, "fa_per_24h": round(24 * fa / hrs, 2)}
            if "none" in P[m] and arm != "none" and all(s in P[m]["none"] for s in subj):
                dd = [per[s] - roc_auc_score(Y[s], P[m]["none"][s]) for s in subj]
                r["delta_vs_none"] = {"mean": round(float(np.mean(dd)), 4), "n_improved": int(sum(x > 0 for x in dd)),
                                      "p_signflip": signflip(dd)}
            if ref is not None and not (m == "feat_gbm" and arm == "refreshed"):
                dd = [per[s] - ref[s] for s in subj]
                r["delta_vs_aura_refreshed"] = {"mean": round(float(np.mean(dd)), 4),
                                                "n_better": int(sum(x > 0 for x in dd)), "p_signflip": signflip(dd)}
            res[arm] = r
        R["models"][m] = res
        print(f"[{m}] " + "  ".join(f"{arm} {v['auc']:.3f}" for arm, v in res.items()), flush=True)
    np.savez_compressed(OUT / "preds.npz", _y=Yc.astype(np.int8),
                        _groups=np.concatenate([[s] * len(Y[s]) for s in subj]),
                        _t=np.concatenate([T[s] - T[s][0] for s in subj]).astype(np.float32),
                        **{f"{m}|{arm}": np.concatenate([P[m][arm][s] for s in subj]).astype(np.float32)
                           for m in P for arm in P[m] if all(s in P[m][arm] for s in subj)})
    save_json(R, OUT / "results.json")
    print(f"-> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
