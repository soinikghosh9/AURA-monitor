#!/usr/bin/env python
"""AURA anchoring in continuous monitoring, with the baseline recorded before anything it scores.

The clinical use of AURA's anchor is: record a short seizure-free baseline at the start of a
monitoring session, then score every later window relative to it. This tests exactly that on the
development streams of the streaming monitor. For each stream, the first ``--baseline-min`` minutes of
non-ictal windows form the patient's reference and are removed from scoring; every later window is
z-scored by that reference. The training patients are anchored the same way, by their earliest third
of negative clips (recording order). The unanchored model is scored on the identical windows, so the
comparison is paired. The streamed patient is never in the training set. CPU only; reuses the
streaming monitor's cached window features.

A second, exploratory arm refreshes the reference: each scored window is z-scored by the median and
median absolute deviation of the stream's windows from ``--rolling-min`` minutes up to ``--gap-min``
minutes before it, without labels (a seizure is short relative to the reference, and the robust
statistics discount it). Training patients are anchored with the same robust statistics of their
earliest negatives. This is the design element proposed for prospective monitoring.

-> results/stream_anchor_<stamp>/results.json
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


def _z(F, ref):
    mu, sd = ref.mean(0), ref.std(0) + 1e-6
    return np.clip((F - mu) / sd, -20, 20)


def _zr(F, ref):
    med = np.median(ref, 0)
    mad = np.median(np.abs(ref - med), 0) * 1.4826 + 1e-6
    return np.clip((F - med) / mad, -20, 20)


def _rolling(F, t, start, span_s, gap_s):
    """Robust z-score of F[i] (i >= start) by the windows in [t_i - gap - span, t_i - gap]."""
    out = np.empty((len(F) - start, F.shape[1]))
    lo = np.searchsorted(t, t - gap_s - span_s, side="left")
    hi = np.searchsorted(t, t - gap_s, side="right")
    for k, i in enumerate(range(start, len(F))):
        ref = F[lo[i]:hi[i]]
        out[k] = _zr(F[i:i + 1], ref)[0] if len(ref) >= 10 else 0.0
    return out


def main() -> int:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from bench_data import load_task
    from run_baseline_placement import early_mask
    from run_final_stats import signflip
    from run_streaming_monitor import (CHB, SIENA, SEAL, _labels, _stream_features,
                                       _stream_record, _train_window_features)
    from bsf.eval import clinical as C
    from bsf.utils.io import save_json
    from bsf.utils.provenance import new_results_dir

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-min", type=float, default=30.0)
    ap.add_argument("--max-records", type=int, default=6)
    ap.add_argument("--max-hours", type=float, default=12.0)
    ap.add_argument("--rolling-min", type=float, default=30.0)
    ap.add_argument("--gap-min", type=float, default=5.0)
    a = ap.parse_args()
    win_s = stride_s = 8.0
    fs = 250.0
    out = new_results_dir("stream_anchor", seed=1234, config=vars(a))
    seal = json.loads((REPO / SEAL).read_text())

    # training pool: the detection clips, as sub-window features (cached by the streaming monitor)
    W, y, g, ev, _ = load_task("detect", "dev")
    n, k, c, t = W.shape
    Ftr = _train_window_features(W[:, :, :22, :2000], fs)
    del W
    ref_tr = early_mask(y, g, ev)
    anch_tr = np.empty_like(Ftr); anch_tr_r = np.empty_like(Ftr)
    for s in np.unique(g):
        m = g == s
        r = Ftr[m & ref_tr].reshape(-1, Ftr.shape[-1])
        anch_tr[m] = _z(Ftr[m], r) if len(r) else np.nan
        anch_tr_r[m] = _zr(Ftr[m], r) if len(r) else np.nan
    has_ref = np.array([(ref_tr & (g == s)).any() for s in g])

    # the development streams, built exactly as the streaming monitor builds them
    dev_subj = {("chbmit", s) for s in seal["cohorts"]["scalp_chbmit"]["dev"]}
    dev_subj |= {("siena", s) for s in seal["cohorts"]["scalp_siena"]["dev"]}
    streams = []
    for cohort, s in sorted(dev_subj):
        if len(streams) >= a.max_records:
            break
        pdir = (CHB if cohort == "chbmit" else SIENA) / s
        if not pdir.is_dir():
            continue
        rec = _stream_record(pdir, cohort, win_s, fs, stride_s, a.max_hours)
        if rec is None or not rec[2]:
            continue
        X, t0s, spans, _ = rec
        yy, ee, usable = _labels(t0s, spans, win_s, "detect", 5.0, 30.0, 30.0, 120.0)
        X, t0s, yy = X[usable], t0s[usable], yy[usable]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        st = {"subject": f"{cohort}_{s}", "X": X, "y": yy, "t0s": t0s, "spans": spans}
        st["F"] = _stream_features(st, "detect", fs, 22, 2000)
        del st["X"]
        streams.append(st)
        print(f"  stream {st['subject']}: {len(yy)} windows, {len(spans)} seizures", flush=True)

    heads = {"feat_gbm": lambda: GradientBoostingClassifier(n_estimators=200, random_state=0),
             "feat_logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(
                 max_iter=2000, class_weight="balanced"))}
    n_base = int(round(a.baseline_min * 60 / stride_s))
    R = {"baseline_min": a.baseline_min, "rolling_min": a.rolling_min, "gap_min": a.gap_min,
         "streams": {}, "models": {}}
    pooled = {m: {"none": [], "anchored": [], "rolling": []} for m in heads}
    Ys, Gs, Ts = [], [], []
    for st in streams:
        order = np.argsort(st["t0s"], kind="stable")
        yy, F, tt = st["y"][order], st["F"][order], st["t0s"][order]
        neg = np.where(yy == 0)[0]
        if len(neg) < n_base:
            continue
        cut = neg[n_base - 1] + 1                    # the baseline ends here; score only what follows
        ref, ys, Fs = F[neg[:n_base]], yy[cut:], F[cut:]
        n_ev_lost = int(np.sum(np.diff(np.r_[0, yy[:cut]]) == 1))
        if ys.sum() == 0:
            continue
        Zs = _z(Fs, ref)
        Zr = _rolling(F, tt, cut, a.rolling_min * 60, a.gap_min * 60)
        keep = (g != st["subject"]) & ~ref_tr & has_ref      # leave the streamed patient out
        Xn = Ftr[keep].reshape(-1, Ftr.shape[-1]); Xa = anch_tr[keep].reshape(-1, Ftr.shape[-1])
        Xr = anch_tr_r[keep].reshape(-1, Ftr.shape[-1])
        yt = np.repeat(y[keep], k)
        for m, mk in heads.items():
            pooled[m]["none"].append(mk().fit(Xn, yt).predict_proba(Fs)[:, 1])
            pooled[m]["anchored"].append(mk().fit(Xa, yt).predict_proba(Zs)[:, 1])
            pooled[m]["rolling"].append(mk().fit(Xr, yt).predict_proba(Zr)[:, 1])
        Ys.append(ys); Gs.append(np.array([st["subject"]] * len(ys))); Ts.append(tt[cut:] - tt[0])
        R["streams"][st["subject"]] = {"n_scored": int(len(ys)), "prevalence": round(float(ys.mean()), 5),
                                       "baseline_windows": n_base, "seizures_before_cut": n_ev_lost,
                                       "hours_scored": round(len(ys) * stride_s / 3600, 2)}
    Y, G = np.concatenate(Ys), np.concatenate(Gs)
    for m in heads:
        P = {arm: np.concatenate(v) for arm, v in pooled[m].items()}
        res = {}
        for arm, p in P.items():
            pl = C.full_panel(Y, p, groups=G, hours_per_window=stride_s / 3600, budgets=(0.05, 0.10, 0.25),
                              n_boot=400)
            b10 = pl["clinical"].get("budget_10pct", {})
            res[arm] = {"auc": pl["discrimination"]["auc"], "auprc": pl["discrimination"]["auprc"],
                        "sens10": b10.get("event_sensitivity"), "chance10": b10.get("chance_sensitivity")}
        per = {}
        for s in np.unique(G):
            mm = G == s
            from sklearn.metrics import roc_auc_score
            per[s] = {arm: round(float(roc_auc_score(Y[mm], P[arm][mm])), 4) for arm in P}
        d = [per[s]["anchored"] - per[s]["none"] for s in per]
        res["per_stream_auc"] = per
        res["n_streams_improved"] = int(sum(x > 0 for x in d))
        res["mean_stream_delta"] = round(float(np.mean(d)), 4)
        res["p_signflip"] = signflip(d)
        dr = [per[s]["rolling"] - per[s]["none"] for s in per]
        res["rolling_n_improved"] = int(sum(x > 0 for x in dr))
        res["rolling_mean_delta"] = round(float(np.mean(dr)), 4)
        res["rolling_p_signflip"] = signflip(dr)
        R["models"][m] = res
        print(f"[{m}] none AUC {res['none']['auc']:.3f} sens10 {res['none']['sens10']} | anchored AUC "
              f"{res['anchored']['auc']:.3f} sens10 {res['anchored']['sens10']} | streams improved "
              f"{res['n_streams_improved']}/{len(per)} mean dAUC {res['mean_stream_delta']:+.3f} "
              f"sf p={res['p_signflip']} | rolling AUC {res['rolling']['auc']:.3f} sens10 "
              f"{res['rolling']['sens10']} improved {res['rolling_n_improved']}/{len(per)} "
              f"mean dAUC {res['rolling_mean_delta']:+.3f} sf p={res['rolling_p_signflip']}", flush=True)
    # per-window predictions, for the stream timelines (time in s from the stream's first window)
    np.savez_compressed(out / "preds.npz", _y=np.concatenate(Ys).astype(np.int8), _groups=np.concatenate(Gs),
                        _t=np.concatenate(Ts).astype(np.float32),
                        **{f"{m}|{arm}": np.concatenate(v).astype(np.float32)
                           for m in heads for arm, v in pooled[m].items()})
    R["n_streams"] = len(Ys); R["n_events"] = int(sum(np.sum(np.diff(np.r_[0, y_]) == 1) for y_ in Ys))
    save_json(R, out / "results.json")
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
