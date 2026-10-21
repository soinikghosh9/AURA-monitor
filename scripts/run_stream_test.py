#!/usr/bin/env python
"""ONE pre-registered evaluation of continuous seizure detection on the held-out test patients.

The refreshed baseline, the networks with and without test-time adaptation, AURA-Net, the ensembles
and first-seizure personalization were all developed on development patients (run_stream_replication,
run_stream_nets, run_stream_tta, run_stream_personalize). This pass scores them once on the continuous
EEG of the held-out test patients (CHB-MIT and Siena), with everything that could be tuned fixed
beforehand from development data and recorded in ``results/_stream_test_manifest.json``:

  * models: every model is fitted once on ALL development detection clips (baseline clips removed, each
    training patient normalized by its earliest third of negatives for the normalized arms);
  * the stream builder, the 30-min fixed-baseline cut, the refreshed rule (35 to 5 min before each
    window) and the inclusion criterion, identical to the development replication;
  * the alarm threshold of every model/arm: the 90th percentile of its 2-min-averaged scores on the
    development streams, transferred unchanged; the stacked ensemble's weights fitted on development
    streams;
  * primary endpoint: per-stream AUC of boosted AURA with the refreshed baseline minus without, over
    the test streams, two-sided sign-flip test at alpha 0.05; everything else is secondary.

  python scripts/run_stream_test.py --write-manifest   # after the development results exist
  python scripts/run_stream_test.py --i-am-ready       # the single pass; a journal refuses a second

-> results/stream_test_<stamp>/{results.json, preds.npz}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

MANIFEST = REPO / "results" / "_stream_test_manifest.json"
JOURNAL = REPO / "results" / "_stream_test_journal.json"
NETS = ["EEGNet", "SCCNet", "BIOT", "deep_cnn_raw"]


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _smoothed_q90(p, G, T):
    from run_stream_replication import smooth
    sm = np.concatenate([smooth(p[G == s], T[G == s]) for s in dict.fromkeys(G)])
    return float(np.quantile(sm, 0.9))


def dev_thresholds_and_stack():
    """Every arm's alarm threshold from its development stream scores, and the stacked ensemble's
    weights fitted on the development streams."""
    from make_paper_numbers import latest
    from sklearn.linear_model import LogisticRegression
    th = {}
    rep = latest("stream_replication_*")
    with np.load(rep / "preds.npz", allow_pickle=True) as z:
        G, T = z["_groups"].astype(str), z["_t"].astype(float)
        for arm in ("none", "refreshed"):
            for m in ("feat_gbm", "feat_logistic"):
                th[f"{m}|{arm}"] = _smoothed_q90(z[f"{m}|{arm}"].astype(float), G, T)
    tta = REPO / "results" / "stream_tta" / "preds.npz"
    stack = None
    with np.load(tta, allow_pickle=True) as z:
        G, T, Y = z["_groups"].astype(str), z["_t"].astype(float), z["_y"].astype(int)
        for key in z.files:
            if key.startswith("_") or key.startswith("feat_"):
                continue
            th[key] = _smoothed_q90(z[key].astype(float), G, T)
        R = json.loads((REPO / "results" / "stream_tta" / "results.json").read_text())
        comp = R.get("stacked_components", [])
        if comp:
            lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731
            X = np.column_stack([lg(z[c].astype(float)) for c in comp])
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0).fit(X, Y)
            stack = {"components": comp, "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0])}
    return th, stack


def arms_plan():
    """Every (model, arm) scored on the test streams; the first is the primary endpoint's pair."""
    plan = [("feat_gbm", "none"), ("feat_gbm", "refreshed"), ("feat_logistic", "none"), ("feat_logistic", "refreshed")]
    for m in NETS:
        plan += [(m, "none"), (m, "ea_refreshed"), (m, "tent"), (m, "probe_gbm_none"), (m, "probe_gbm_refreshed")]
        if m != "BIOT":
            plan += [(m, "adabn_refreshed")]
    plan += [("auranet_hybrid", "none"), ("auranet_hybrid", "refreshed"), ("ensemble", "none"),
             ("ensemble", "refreshed"), ("stacked", "refreshed"), ("personalized", "refreshed")]
    return plan


def main() -> int:
    from run_stream_replication import RULE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-manifest", action="store_true")
    ap.add_argument("--i-am-ready", action="store_true")
    ap.add_argument("--epochs", type=int, default=25)
    a = ap.parse_args()
    seal = json.loads((REPO / "results" / "_blind_split.json").read_text())
    test = [("chbmit", s) for s in seal["cohorts"]["scalp_chbmit"]["test"]] + \
           [("siena", s) for s in seal["cohorts"]["scalp_siena"]["test"]]
    if a.write_manifest:
        if MANIFEST.exists():
            raise SystemExit("manifest exists -- refusing to overwrite")
        th, stack = dev_thresholds_and_stack()
        body = {"rule": RULE, "test_patients": sorted(f"{c}_{s}" for c, s in test), "arms": arms_plan(),
                "thresholds_dev_q90_smoothed": th, "stacked_dev": stack,
                "primary": "feat_gbm per-stream AUC(refreshed) - AUC(none) over the test streams; two-sided "
                           "sign-flip test, alpha 0.05",
                "training": "all development detection clips (earliest-third baseline clips removed); "
                            "normalized arms: each training patient z-scored by its baseline",
                "personalization": "mean of refreshed boosted AURA and an L2 logistic (C=0.1) fitted on the "
                                   "first seizure's windows and the non-ictal windows before it, refreshed features"}
        MANIFEST.write_text(json.dumps({**body, "sha256": _sha(body),
                                        "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=2))
        print("manifest", _sha(body))
        return 0
    if not a.i_am_ready:
        print("dry run: pass --write-manifest, commit it, then --i-am-ready")
        return 0
    if JOURNAL.exists():
        raise SystemExit(f"the test-stream pass already ran ({JOURNAL}) -- refusing a second pass")
    M = json.loads(MANIFEST.read_text())
    body = {k: v for k, v in M.items() if k not in ("sha256", "created")}
    if _sha(body) != M["sha256"]:
        raise SystemExit("manifest altered after it was recorded -- refusing")
    JOURNAL.write_text(json.dumps({"started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                   "manifest": M["sha256"]}, indent=2))
    return run_pass(M, test, a.epochs)


def run_pass(M, test, epochs) -> int:
    import torch
    from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from bench_data import euclidean_align, load_task
    from bsf.models import zoo
    from bsf.models.auranet_train import AURANetClassifier
    from bsf.utils.compute import resolve_device
    from bsf.utils.io import save_json
    from bsf.utils.provenance import new_results_dir
    from run_baseline_placement import early_mask
    from run_final_stats import signflip
    from run_stream_nets import predict, refreshed_ea
    from run_stream_replication import RULE, _rolling, _zr, build_stream, events_at, smooth
    from run_stream_tta import _bns, adabn_refreshed, embed, tent
    from run_streaming_monitor import _stream_features, _train_window_features

    dev = resolve_device("auto")
    out = new_results_dir("stream_test", seed=1234, config={"manifest": M["sha256"]})
    fs, C_, T_ = RULE["fs"], RULE["channels"], 2000
    span, gap = RULE["rolling_min"] * 60, RULE["gap_min"] * 60
    n_base = int(round(RULE["fixed_baseline_min"] * 60 / RULE["stride_s"]))

    # ---- training data: all development detection clips ----
    W, y, g, ev, _ = load_task("detect", "dev")
    W = W[:, :, :C_, :T_]; k = W.shape[1]
    base = early_mask(y, g, ev)
    no_ref = [s for s in np.unique(g) if not ((g == s) & base).any()]
    keep = ~base & ~np.isin(g, no_ref)
    Ftr = _train_window_features(W, fs)
    Fz = np.empty_like(Ftr)
    for s in np.unique(g):
        m = g == s
        r = Ftr[m & base].reshape(-1, Ftr.shape[-1])
        Fz[m] = _zr(Ftr[m], r) if len(r) else np.nan

    # ---- test streams (same builder and inclusion rule as development) ----
    class _A:
        max_hours = RULE["max_hours"]
    streams = {}
    for coh, s in sorted(test):
        st = build_stream(coh, s, _A)
        if st is None:
            print(f"  {coh}_{s}: no scoreable stream", flush=True); continue
        o = np.argsort(st["t0s"], kind="stable")
        yy, tt = st["y"][o], st["t0s"][o]
        neg = np.where(yy == 0)[0]
        if len(neg) < n_base:
            continue
        cut = neg[n_base - 1] + 1
        if yy[cut:].sum() == 0:
            print(f"  {coh}_{s}: no seizure after the baseline", flush=True); continue
        F = _stream_features(st, "detect", fs, C_, T_)[o]
        streams[st["subject"]] = {"order": o, "cut": cut, "y": yy, "t": tt, "X": st["X"][o][:, :C_, :T_],
                                  "F": F, "Z": _rolling(F, tt, cut, span, gap, RULE["min_ref_windows"])}
        print(f"  {st['subject']}: {len(yy) - cut} scored windows, {int(np.sum(np.diff(np.r_[0, yy[cut:]]) == 1))} seizures",
              flush=True)
    subj = sorted(streams)
    Y = {s: streams[s]["y"][streams[s]["cut"]:] for s in subj}
    T = {s: streams[s]["t"][streams[s]["cut"]:] for s in subj}
    P = {}

    def put(m, arm, s, v):
        P.setdefault(m, {}).setdefault(arm, {})[s] = np.asarray(v, float)

    # ---- AURA ----
    tr = keep
    for m, mk in (("feat_gbm", lambda: GradientBoostingClassifier(n_estimators=200, random_state=0)),
                  ("feat_logistic", lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")))):
        c0 = mk().fit(Ftr[tr].reshape(-1, Ftr.shape[-1]), np.repeat(y[tr], k))
        c1 = mk().fit(np.nan_to_num(Fz[tr].reshape(-1, Ftr.shape[-1])), np.repeat(y[tr], k))
        for s in subj:
            st = streams[s]
            put(m, "none", s, c0.predict_proba(st["F"][st["cut"]:])[:, 1])
            put(m, "refreshed", s, c1.predict_proba(st["Z"])[:, 1])
    print("  AURA scored", flush=True)

    # ---- networks: none (+ TTA + probes) and input-level refreshed EA ----
    ref_W = {s: W[(g == s) & base] for s in np.unique(g) if s not in no_ref}
    W_ea = euclidean_align(W[keep], g[keep], ref_W)
    for m in NETS:
        spec = zoo.build(m, C_, T_, fs, device=dev, seed=0)
        zoo.fit_predict(spec, W[keep].reshape(-1, C_, T_), np.repeat(y[keep], k), W[keep][:1, 0], dev,
                        epochs=epochs, micro_bs=16, seed=0)
        mod = spec.module.eval()
        Etr = embed(mod, W[keep].reshape(-1, C_, T_), dev)
        gtr = np.repeat(g[keep], k); ytr = np.repeat(y[keep], k)
        Etr_z = np.empty_like(Etr)
        for s in np.unique(g[keep]):
            Etr_z[gtr == s] = _zr(Etr[gtr == s], embed(mod, W[(g == s) & base].reshape(-1, C_, T_), dev))
        hg = {arm: HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.1, random_state=0).fit(
            Etr if arm == "none" else Etr_z, ytr) for arm in ("none", "refreshed")}
        del Etr, Etr_z
        for s in subj:
            st = streams[s]; X = st["X"]; Xs = X[st["cut"]:]
            put(m, "none", s, predict(mod, Xs, dev))
            if _bns(mod):
                put(m, "adabn_refreshed", s, adabn_refreshed(mod, X, st["t"], st["cut"], span, gap, dev))
            tt = tent(mod, Xs, dev)
            if tt is not None:
                put(m, "tent", s, tt)
            Es = embed(mod, X, dev)
            put(m, "probe_gbm_none", s, hg["none"].predict_proba(Es[st["cut"]:])[:, 1])
            put(m, "probe_gbm_refreshed", s, hg["refreshed"].predict_proba(
                _rolling(Es, st["t"], st["cut"], span, gap, RULE["min_ref_windows"]))[:, 1])
        del spec, mod
        spec = zoo.build(m, C_, T_, fs, device=dev, seed=0)
        zoo.fit_predict(spec, W_ea.reshape(-1, C_, T_), np.repeat(y[keep], k), W_ea[:1, 0], dev,
                        epochs=epochs, micro_bs=16, seed=0)
        for s in subj:
            st = streams[s]
            put(m, "ea_refreshed", s, predict(spec.module, refreshed_ea(st["X"], st["t"], st["cut"], span, gap,
                                                                        RULE["min_ref_windows"]), dev))
        del spec
        if dev == "cuda":
            torch.cuda.empty_cache()
        print(f"  {m} scored", flush=True)

    # ---- AURA-Net hybrid ----
    for arm, Ft in (("none", Ftr), ("refreshed", Fz)):
        clf = AURANetClassifier(fs=fs, epochs=epochs, device=dev, seed=0, d_feat=Ftr.shape[-1])
        clf.fit(W[keep], y[keep], g[keep], feats=np.nan_to_num(Ft[keep].mean(1)))
        for s in subj:
            st = streams[s]
            Xs = st["X"][st["cut"]:][:, None].astype(np.float32)
            put("auranet_hybrid", arm, s, clf.predict_proba(Xs, np.array([s] * len(Xs)),
                                                            feats=np.nan_to_num(st["F"][st["cut"]:] if arm == "none" else st["Z"])))
        del clf
    print("  AURA-Net scored", flush=True)

    # ---- ensembles and personalization (as specified on development data) ----
    for arm, narm in (("none", "none"), ("refreshed", "ea_refreshed")):
        for s in subj:
            put("ensemble", arm, s, 0.5 * P["feat_gbm"][arm][s] + 0.5 * np.mean([P[m][narm][s] for m in NETS], 0))
    stack = M.get("stacked_dev")
    if stack:
        lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731
        for s in subj:
            X = np.column_stack([lg(P[c.split("|")[0]][c.split("|")[1]][s]) for c in stack["components"]])
            put("stacked", "refreshed", s, 1 / (1 + np.exp(-(X @ np.array(stack["coef"]) + stack["intercept"]))))
    pers_streams = []
    for s in subj:
        st = streams[s]; Z = st["Z"]; yy = Y[s]; tt = T[s]
        idx = np.flatnonzero(yy)
        runs = np.split(idx, np.where((np.diff(idx) > 1) | (np.diff(tt[idx]) > 1.5 * RULE["window_s"]))[0] + 1)
        if len(runs) < 2:
            continue
        first = runs[0]; neg = np.arange(0, first[0]); neg = neg[yy[neg] == 0]
        if len(neg) < 10:
            continue
        sc = StandardScaler().fit(Z[:first[-1] + 1])
        lr = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000).fit(
            sc.transform(np.concatenate([Z[first], Z[neg]])), np.r_[np.ones(len(first)), np.zeros(len(neg))])
        after = np.arange(first[-1] + 1, len(yy))
        if yy[after].sum() == 0:
            continue
        pers_streams.append(s)
        P.setdefault("personalized_after", {}).setdefault("population", {})[s] = P["feat_gbm"]["refreshed"][s][after]
        P["personalized_after"].setdefault("combined", {})[s] = 0.5 * P["feat_gbm"]["refreshed"][s][after] + \
            0.5 * lr.predict_proba(sc.transform(Z[after]))[:, 1]
        P["personalized_after"].setdefault("_y", {})[s] = yy[after]
        P["personalized_after"].setdefault("_t", {})[s] = tt[after]

    # ---- evaluation ----
    th = M["thresholds_dev_q90_smoothed"]
    Yc = np.concatenate([Y[s] for s in subj])
    R = {"manifest": M["sha256"], "n_streams": len(subj), "streams": {s: {"n_scored": int(len(Y[s])),
         "hours": round(len(Y[s]) * RULE["window_s"] / 3600, 2), "n_seizures": int(np.sum(np.diff(np.r_[0, Y[s]]) == 1))}
         for s in subj}, "models": {}}
    for m in P:
        if m == "personalized_after":
            continue
        for arm, d in P[m].items():
            per = {s: float(roc_auc_score(Y[s], d[s])) for s in subj}
            key = f"{m}|{arm}"
            r = {"auc": round(float(roc_auc_score(Yc, np.concatenate([d[s] for s in subj]))), 4),
                 "median_stream_auc": round(float(np.median(list(per.values()))), 4),
                 "per_stream_auc": {s: round(v, 4) for s, v in per.items()}}
            if key in th:
                cau = nev = fa = 0; hrs = 0.0
                for s in subj:
                    e = events_at(Y[s], smooth(d[s], T[s]), T[s], th[key], RULE["window_s"])
                    cau += e["caught"]; nev += e["n_events"]; fa += e["false_alarms"]; hrs += len(Y[s]) * RULE["window_s"] / 3600
                r.update({"caught": cau, "n_events": nev, "fa_per_24h": round(24 * fa / hrs, 2),
                          "time_in_warning": round(float(np.mean(np.concatenate([smooth(d[s], T[s]) >= th[key] for s in subj]))), 4)})
            R["models"][key] = r
    pf = R["models"]
    for m in ("feat_gbm", "feat_logistic", *NETS, "auranet_hybrid", "ensemble"):
        for arm in ("refreshed", "ea_refreshed", "tent", "adabn_refreshed", "probe_gbm_refreshed"):
            if f"{m}|{arm}" in pf and f"{m}|none" in pf:
                dd = [pf[f"{m}|{arm}"]["per_stream_auc"][s] - pf[f"{m}|none"]["per_stream_auc"][s] for s in subj]
                pf[f"{m}|{arm}"]["delta_vs_none"] = {"mean": round(float(np.mean(dd)), 4),
                                                     "n_improved": int(sum(x > 0 for x in dd)), "n": len(dd),
                                                     "p_signflip": signflip(dd)}
    ref = pf["feat_gbm|refreshed"]["per_stream_auc"]
    for key, r in pf.items():
        if key != "feat_gbm|refreshed":
            dd = [r["per_stream_auc"][s] - ref[s] for s in subj]
            r["delta_vs_aura_refreshed"] = {"mean": round(float(np.mean(dd)), 4), "n_better": int(sum(x > 0 for x in dd)),
                                            "p_signflip": signflip(dd)}
    R["primary"] = pf["feat_gbm|refreshed"]["delta_vs_none"]
    if pers_streams:
        pa = P["personalized_after"]
        per = {s: {a: float(roc_auc_score(pa["_y"][s], pa[a][s])) for a in ("population", "combined")} for s in pers_streams}
        dd = [per[s]["combined"] - per[s]["population"] for s in pers_streams]
        R["personalization"] = {"n_streams": len(pers_streams), "per_stream": per,
                                "median_population": round(float(np.median([per[s]["population"] for s in pers_streams])), 4),
                                "median_combined": round(float(np.median([per[s]["combined"] for s in pers_streams])), 4),
                                "mean_delta": round(float(np.mean(dd)), 4), "n_improved": int(sum(x > 0 for x in dd)),
                                "p_signflip": signflip(dd)}
    np.savez_compressed(out / "preds.npz", _y=Yc.astype(np.int8), _groups=np.concatenate([[s] * len(Y[s]) for s in subj]),
                        _t=np.concatenate([T[s] - T[s][0] for s in subj]).astype(np.float32),
                        **{f"{m}|{arm}": np.concatenate([P[m][arm][s] for s in subj]).astype(np.float32)
                           for m in P if m != "personalized_after" for arm in P[m]})
    save_json(R, out / "results.json")
    j = json.loads(JOURNAL.read_text())
    j.update({"finished": datetime.now(timezone.utc).isoformat(timespec="seconds"), "results": str(out)})
    JOURNAL.write_text(json.dumps(j, indent=2))
    pr = R["primary"]
    print(f"PRIMARY: refreshed - none per-stream AUC {pr['mean']:+.3f} ({pr['n_improved']}/{pr['n']}), "
          f"sign-flip p={pr['p_signflip']}", flush=True)
    for key in ("feat_gbm|none", "feat_gbm|refreshed", "ensemble|refreshed", "stacked|refreshed"):
        if key in pf:
            print(f"  {key}: AUC {pf[key]['auc']:.3f} events {pf[key].get('caught')}/{pf[key].get('n_events')} "
                  f"FA/24h {pf[key].get('fa_per_24h')}", flush=True)
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
