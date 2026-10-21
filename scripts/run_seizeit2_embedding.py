#!/usr/bin/env python
"""POST-HOC (not pre-registered): embedding-level refreshed normalization in the SeizeIT2 external cohort.

The pre-registered SeizeIT2 pass (run_seizeit2_external.py) tested the feature-level refreshed baseline,
input alignment, AdaBN and TENT. After it, the held-out scalp pass showed that normalizing a network's
learned embedding to the patient's preceding 30 min helped BIOT on held-out patients. This script asks the
same question in SeizeIT2, on the same split, training windows and held-out streams, and is reported as a
post-hoc analysis. Each network is trained as in the pass (development half, benchmark recipe, seed 0);
its penultimate embedding is read by a boosted head, either as is or robust-z-scored (training patients by
their earliest third of non-ictal windows; streams by their windows 35 to 5 min earlier).

-> results/seizeit2_embedding_<stamp>/results.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


def main() -> int:
    import torch
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    from bsf.models import zoo
    from bsf.utils.compute import resolve_device
    from bsf.utils.io import save_json
    from bsf.utils.provenance import new_results_dir
    from run_final_stats import signflip
    from run_seizeit2_external import FS, MANIFEST, NETS, WIN_S, stream, training_windows
    from run_stream_replication import RULE, _rolling, _zr, events_at, smooth
    from run_stream_tta import embed

    dev = resolve_device("auto")
    M = json.loads(MANIFEST.read_text())
    out = new_results_dir("seizeit2_embedding", seed=1234, config={"manifest": M["sha256"], "post_hoc": True})
    span, gap = RULE["rolling_min"] * 60, RULE["gap_min"] * 60
    n_base = int(round(RULE["fixed_baseline_min"] * 60 / RULE["stride_s"]))
    rng = np.random.default_rng(20261004)
    Xs, ys, gs, bs = [], [], [], []
    for s in M["development"]:
        r = training_windows(s, rng)               # cached by the pre-registered pass
        if r is None:
            continue
        X, y, order = r
        neg = np.where(y == 0)[0]
        b = np.zeros(len(y), bool)
        b[neg[np.argsort(order[neg], kind="stable")[:max(5, len(neg) // 3)]]] = True
        Xs.append(X); ys.append(y); gs.append(np.array([s] * len(y))); bs.append(b)
    X, y, g, base = np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs), np.concatenate(bs)
    del Xs
    tr = ~base
    streams = {}
    for s in M["held_out"]:
        r = stream(s)
        if r is None:
            continue
        Xst, yst, tst = r
        neg = np.where(yst == 0)[0]
        if len(neg) < n_base:
            continue
        cut = neg[n_base - 1] + 1
        if yst[cut:].sum() == 0:
            continue
        streams[s] = {"X": Xst, "y": yst, "t": tst, "cut": cut}
    subj = sorted(streams)
    Y = {s: streams[s]["y"][streams[s]["cut"]:] for s in subj}
    T = {s: streams[s]["t"][streams[s]["cut"]:] for s in subj}
    R = {"post_hoc": True, "manifest": M["sha256"], "n_streams": len(subj), "models": {}}
    for m in NETS:
        spec = zoo.build(m, 2, int(WIN_S * FS), FS, device=dev, seed=0)
        zoo.fit_predict(spec, X[tr], y[tr], X[:1], dev, epochs=M.get("epochs", 25), micro_bs=32, seed=0)
        mod = spec.module.eval()
        Etr = embed(mod, X[tr], dev)
        Ez = np.empty_like(Etr)
        gtr = g[tr]
        for s in np.unique(gtr):
            Ez[gtr == s] = _zr(Etr[gtr == s], embed(mod, X[(g == s) & base], dev))
        heads = {arm: HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.1, random_state=0).fit(
            Etr if arm == "none" else Ez, y[tr]) for arm in ("none", "refreshed")}
        P = {"none": {}, "refreshed": {}}
        for s in subj:
            st = streams[s]
            E = embed(mod, st["X"], dev)
            P["none"][s] = heads["none"].predict_proba(E[st["cut"]:])[:, 1]
            P["refreshed"][s] = heads["refreshed"].predict_proba(
                _rolling(E, st["t"], st["cut"], span, gap, RULE["min_ref_windows"]))[:, 1]
        res = {}
        for arm in P:
            per = {s: float(roc_auc_score(Y[s], P[arm][s])) for s in subj}
            cau = nev = fa = 0; hrs = 0.0
            for s in subj:
                oth = np.concatenate([smooth(P[arm][o], T[o]) for o in subj if o != s])
                e = events_at(Y[s], smooth(P[arm][s], T[s]), T[s], float(np.quantile(oth, 0.9)), WIN_S)
                cau += e["caught"]; nev += e["n_events"]; fa += e["false_alarms"]; hrs += len(Y[s]) * WIN_S / 3600
            res[arm] = {"auc": round(float(roc_auc_score(np.concatenate([Y[s] for s in subj]),
                                                         np.concatenate([P[arm][s] for s in subj]))), 4),
                        "median_stream_auc": round(float(np.median(list(per.values()))), 4),
                        "per_stream_auc": {s: round(v, 4) for s, v in per.items()},
                        "caught": cau, "n_events": nev, "fa_per_24h": round(24 * fa / hrs, 2)}
        dd = [res["refreshed"]["per_stream_auc"][s] - res["none"]["per_stream_auc"][s] for s in subj]
        res["refreshed"]["delta_vs_none"] = {"mean": round(float(np.mean(dd)), 4), "n_improved": int(sum(x > 0 for x in dd)),
                                             "n": len(dd), "p_signflip": signflip(dd)}
        R["models"][m] = res
        print(f"[{m}] embedding none {res['none']['auc']:.3f} refreshed {res['refreshed']['auc']:.3f} "
              f"per-stream {np.mean(dd):+.3f} ({sum(x > 0 for x in dd)}/{len(dd)}) p={res['refreshed']['delta_vs_none']['p_signflip']}",
              flush=True)
        del spec, mod
        if dev == "cuda":
            torch.cuda.empty_cache()
        save_json(R, out / "results.json")
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
