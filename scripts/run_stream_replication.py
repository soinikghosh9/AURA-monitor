#!/usr/bin/env python
"""Pre-registered replication of the refreshed baseline in continuous monitoring (development patients).

The refreshed baseline (each window robust-z-scored by the stream's own windows from 35 to 5 min before
it, no labels) was devised on six development streams in ``run_stream_anchor.py`` and is exploratory
there. This script applies the same rule, unchanged, to every development patient with a scoreable
stream. The rule, the endpoints, the candidate patients and the stream-inclusion criterion are written to
``results/_stream_replication_manifest.json`` and its SHA-256 printed BEFORE any stream is built; a second
call refuses to run against a changed manifest. Test patients are never touched.

Primary endpoint: per-stream AUC of boosted AURA with the refreshed baseline minus without, over the
streams NOT among the original six, exact/Monte-Carlo sign-flip test (two-sided, alpha 0.05).
Secondary: the same for logistic AURA and over all streams; pooled AUC; event sensitivity, false alarms
per 24 h and detection latency at a threshold fixed on the OTHER streams' scores (10% time in warning),
for raw scores and for a causal 2-min moving average.

Training is unchanged from the original analysis: leave-the-streamed-patient-out, on the detection clips
of the other development patients, each normalized by the robust statistics of its earliest third of
negative clips. Each built stream (standardized windows, labels, times) is also cached for
``run_stream_nets.py``.

-> results/stream_replication_<stamp>/{results.json, preds.npz}
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

MANIFEST = REPO / "results" / "_stream_replication_manifest.json"
CACHE = REPO / "results" / "_clip_cache"
ORIGINAL = ["chbmit_chb01", "chbmit_chb03", "chbmit_chb04", "chbmit_chb06", "chbmit_chb08",
            "chbmit_chb09"]
RULE = {"window_s": 8.0, "stride_s": 8.0, "fs": 250.0, "channels": 22, "max_hours": 12.0,
        "fixed_baseline_min": 30.0, "rolling_min": 30.0, "gap_min": 5.0, "min_ref_windows": 10,
        "statistic": "median and 1.4826*MAD of the reference windows, clip at +-20",
        "training": "leave-streamed-patient-out on dev detection clips; each training patient normalized "
                    "by its earliest third of negative clips (same statistic)",
        "heads": {"feat_gbm": "GradientBoostingClassifier(n_estimators=200, random_state=0)",
                  "feat_logistic": "StandardScaler + LogisticRegression(class_weight=balanced)"},
        "inclusion": "a development patient's stream is scored if, after its first 30 min of non-ictal "
                     "windows (removed as the fixed baseline), it contains >= 1 ictal window",
        "event_threshold": "per stream, the 90th percentile of the other scored streams' scores in the same "
                           "arm (10% time in warning)",
        "smoothing": "causal moving average over windows starting in (t-120 s, t]"}
ENDPOINTS = {"primary": "feat_gbm: per-stream AUC(refreshed) - AUC(none) over streams not in ORIGINAL; "
                        "two-sided sign-flip test, alpha 0.05",
             "secondary": ["feat_logistic, same contrast", "all streams, both heads",
                           "pooled AUC per arm", "event sensitivity, false alarms per 24 h, median latency "
                           "at the transferred threshold, raw and smoothed"]}


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def write_or_check_manifest(candidates: list[str]) -> str:
    body = {"rule": RULE, "endpoints": ENDPOINTS, "original_streams": ORIGINAL,
            "candidates": sorted(candidates), "split": "results/_blind_split.json (dev only)"}
    digest = _sha(body)
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text())
        if old["sha256"] != digest:
            raise SystemExit(f"manifest changed: recorded {old['sha256'][:12]}, now {digest[:12]} -- refusing")
        print(f"manifest {digest[:12]} (recorded {old['created']})", flush=True)
    else:
        MANIFEST.write_text(json.dumps({**body, "sha256": digest,
                                        "created": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                                       indent=2))
        print(f"manifest written: {digest}", flush=True)
    return digest


def _z(F, ref):
    mu, sd = ref.mean(0), ref.std(0) + 1e-6
    return np.clip((F - mu) / sd, -20, 20)


def _zr(F, ref):
    med = np.median(ref, 0)
    mad = np.median(np.abs(ref - med), 0) * 1.4826 + 1e-6
    return np.clip((F - med) / mad, -20, 20)


def _rolling(F, t, start, span_s, gap_s, min_ref):
    out = np.empty((len(F) - start, F.shape[1]))
    lo = np.searchsorted(t, t - gap_s - span_s, side="left")
    hi = np.searchsorted(t, t - gap_s, side="right")
    for k, i in enumerate(range(start, len(F))):
        ref = F[lo[i]:hi[i]]
        out[k] = _zr(F[i:i + 1], ref)[0] if len(ref) >= min_ref else 0.0
    return out


def smooth(p, t, span_s=120.0):
    """Causal moving average over windows starting in (t - span, t]."""
    c = np.r_[0.0, np.cumsum(p)]
    lo = np.searchsorted(t, t - span_s, side="right")
    hi = np.arange(1, len(p) + 1)
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)


def events_at(y, p, t, thr, win_s):
    """Seizures caught, false-alarm runs and latency (s) at one threshold; runs end at recording gaps."""
    warn = p >= thr
    brk = np.r_[True, np.diff(t) > 1.5 * win_s]          # a gap starts a new run
    def runs(mask):
        out, cur = [], []
        for i in range(len(mask)):
            if mask[i] and cur and not brk[i] and cur[-1] == i - 1:
                cur.append(i)
            elif mask[i]:
                if cur:
                    out.append(cur)
                cur = [i]
            else:
                if cur:
                    out.append(cur)
                cur = []
        if cur:
            out.append(cur)
        return out
    evs = runs(y.astype(bool))
    caught, lat = 0, []
    for e in evs:
        hit = np.flatnonzero(warn[e])
        if hit.size:
            caught += 1
            lat.append(float(t[e[hit[0]]] - t[e[0]]))
    fa = sum(1 for r in runs(warn) if not y[r].any())
    return {"n_events": len(evs), "caught": caught, "false_alarms": fa, "latency_s": lat,
            "time_in_warning": float(warn.mean())}


def build_stream(cohort, s, a, montage="native"):
    """The stream exactly as the original analysis builds it; cached for the network script."""
    from run_streaming_monitor import CHB, SIENA, _labels, _stream_record
    subj = f"{cohort}_{s}"
    f = CACHE / f"streamrep_{subj}_h{int(a.max_hours)}{'' if montage == 'native' else '_' + montage}.npz"
    if f.exists():
        d = np.load(f, allow_pickle=False)
        return {"subject": subj, "X": d["X"], "y": d["y"], "t0s": d["t0s"],
                "spans": [tuple(x) for x in d["spans"]]}
    pdir = (CHB if cohort == "chbmit" else SIENA) / s
    if not pdir.is_dir():
        return None
    rec = _stream_record(pdir, cohort, RULE["window_s"], RULE["fs"], RULE["stride_s"], a.max_hours, montage)
    if rec is None or not rec[2]:
        return None
    X, t0s, spans, _ = rec
    yy, _ee, usable = _labels(t0s, spans, RULE["window_s"], "detect", 5.0, 30.0, 30.0, 120.0)
    X, t0s, yy = X[usable], t0s[usable], yy[usable]
    if yy.sum() == 0 or yy.sum() == len(yy):
        return None
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(f, X=X, y=yy, t0s=t0s, spans=np.asarray(spans, float).reshape(-1, 2))
    return {"subject": subj, "X": X, "y": yy, "t0s": t0s, "spans": spans}


def main() -> int:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from bench_data import load_task
    from run_baseline_placement import early_mask
    from run_final_stats import signflip
    from run_streaming_monitor import SEAL, _stream_features, _train_window_features
    from bsf.eval import clinical as C
    from bsf.utils.io import save_json
    from bsf.utils.provenance import new_results_dir

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-hours", type=float, default=RULE["max_hours"])
    ap.add_argument("--n-jobs", type=int, default=12)
    ap.add_argument("--manifest-only", action="store_true")
    a = ap.parse_args()
    seal = json.loads((REPO / SEAL).read_text())
    cands = [("chbmit", s) for s in seal["cohorts"]["scalp_chbmit"]["dev"]]
    cands += [("siena", s) for s in seal["cohorts"]["scalp_siena"]["dev"]]
    digest = write_or_check_manifest([f"{c}_{s}" for c, s in cands])
    if a.manifest_only:
        return 0
    out = new_results_dir("stream_replication", seed=1234, config={**vars(a), "manifest": digest})
    win_s, fs = RULE["window_s"], RULE["fs"]

    W, y, g, ev, _ = load_task("detect", "dev")
    n, k, c, t = W.shape
    Ftr = _train_window_features(W[:, :, :RULE["channels"], :2000], fs)
    del W
    ref_tr = early_mask(y, g, ev)
    an_tr = np.empty_like(Ftr); an_tr_r = np.empty_like(Ftr)
    for s in np.unique(g):
        m = g == s
        r = Ftr[m & ref_tr].reshape(-1, Ftr.shape[-1])
        an_tr[m] = _z(Ftr[m], r) if len(r) else np.nan
        an_tr_r[m] = _zr(Ftr[m], r) if len(r) else np.nan
    has_ref = np.array([(ref_tr & (g == s)).any() for s in g])

    heads = {"feat_gbm": lambda: GradientBoostingClassifier(n_estimators=200, random_state=0),
             "feat_logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(
                 max_iter=2000, class_weight="balanced"))}
    arms = ("none", "fixed", "refreshed")
    n_base = int(round(RULE["fixed_baseline_min"] * 60 / RULE["stride_s"]))
    P = {m: {arm: [] for arm in arms} for m in heads}
    Ys, Gs, Ts, info = [], [], [], {}
    for cohort, s in sorted(cands):
        st = build_stream(cohort, s, a)
        if st is None:
            print(f"  {cohort}_{s}: no scoreable stream", flush=True)
            continue
        st["F"] = _stream_features(st, "detect", fs, RULE["channels"], 2000, n_jobs=a.n_jobs)
        order = np.argsort(st["t0s"], kind="stable")
        yy, F, tt = st["y"][order], st["F"][order], st["t0s"][order]
        neg = np.where(yy == 0)[0]
        if len(neg) < n_base:
            continue
        cut = neg[n_base - 1] + 1
        ys, Fs = yy[cut:], F[cut:]
        if ys.sum() == 0:
            print(f"  {st['subject']}: no seizure after the baseline", flush=True)
            continue
        ref = F[neg[:n_base]]
        Z = {"none": Fs, "fixed": _z(Fs, ref),
             "refreshed": _rolling(F, tt, cut, RULE["rolling_min"] * 60, RULE["gap_min"] * 60,
                                   RULE["min_ref_windows"])}
        keep = (g != st["subject"]) & ~ref_tr & has_ref
        Xtr = {"none": Ftr[keep], "fixed": an_tr[keep], "refreshed": an_tr_r[keep]}
        yt = np.repeat(y[keep], k)
        for m, mk in heads.items():
            for arm in arms:
                P[m][arm].append(mk().fit(Xtr[arm].reshape(-1, Ftr.shape[-1]), yt).predict_proba(Z[arm])[:, 1])
        Ys.append(ys); Gs.append(np.array([st["subject"]] * len(ys))); Ts.append(tt[cut:])
        info[st["subject"]] = {"n_scored": int(len(ys)), "hours_scored": round(len(ys) * win_s / 3600, 2),
                               "prevalence": round(float(ys.mean()), 5), "original": st["subject"] in ORIGINAL}
        print(f"  {st['subject']}: {len(ys)} windows, {info[st['subject']]['hours_scored']} h, "
              f"gbm AUC none {roc_auc_score(ys, P['feat_gbm']['none'][-1]):.3f} refreshed "
              f"{roc_auc_score(ys, P['feat_gbm']['refreshed'][-1]):.3f}", flush=True)
        del st

    subj = [gg[0] for gg in Gs]
    R = {"manifest_sha256": digest, "streams": info, "n_streams": len(subj),
         "n_replication": int(sum(s not in ORIGINAL for s in subj)), "models": {}}
    for m in heads:
        per = {s: {arm: float(roc_auc_score(Ys[i], P[m][arm][i])) for arm in arms} for i, s in enumerate(subj)}
        res = {"per_stream_auc": per}
        for setname, keep_s in (("replication", [s for s in subj if s not in ORIGINAL]),
                                ("original", [s for s in subj if s in ORIGINAL]), ("all", subj)):
            idx = [subj.index(s) for s in keep_s]
            if not idx:
                continue
            Y = np.concatenate([Ys[i] for i in idx]); G = np.concatenate([Gs[i] for i in idx])
            blk = {"n_streams": len(idx), "n_events": 0}
            for arm in arms:
                p = np.concatenate([P[m][arm][i] for i in idx])
                pl = C.full_panel(Y, p, groups=G, hours_per_window=win_s / 3600, budgets=(0.10,), n_boot=400)
                blk[arm] = {"auc": pl["discrimination"]["auc"], "auprc": pl["discrimination"]["auprc"],
                            "sens10_pooled": pl["clinical"]["budget_10pct"]["event_sensitivity"]}
                for sm in (False, True):
                    tot = {"n_events": 0, "caught": 0, "false_alarms": 0, "hours": 0.0, "latency_s": [],
                           "tiw": []}
                    for i in idx:
                        others = np.concatenate([P[m][arm][j] for j in range(len(subj)) if j != i])
                        if sm:
                            parts = [smooth(P[m][arm][j], Ts[j]) for j in range(len(subj)) if j != i]
                            others = np.concatenate(parts)
                        thr = float(np.quantile(others, 0.90))
                        pi = smooth(P[m][arm][i], Ts[i]) if sm else P[m][arm][i]
                        e = events_at(Ys[i], pi, Ts[i], thr, win_s)
                        tot["n_events"] += e["n_events"]; tot["caught"] += e["caught"]
                        tot["false_alarms"] += e["false_alarms"]; tot["latency_s"] += e["latency_s"]
                        tot["hours"] += len(Ys[i]) * win_s / 3600; tot["tiw"].append(e["time_in_warning"])
                        del others
                    key = "smoothed" if sm else "raw"
                    blk[arm][f"events_{key}"] = {
                        "sensitivity": round(tot["caught"] / max(tot["n_events"], 1), 4),
                        "caught": tot["caught"], "n_events": tot["n_events"],
                        "fa_per_24h": round(24 * tot["false_alarms"] / max(tot["hours"], 1e-9), 2),
                        "median_latency_s": (round(float(np.median(tot["latency_s"])), 1)
                                             if tot["latency_s"] else None),
                        "median_time_in_warning": round(float(np.median(tot["tiw"])), 4)}
                    blk["n_events"] = tot["n_events"]
            for arm in ("fixed", "refreshed"):
                d = [per[s][arm] - per[s]["none"] for s in keep_s]
                blk[f"delta_{arm}"] = {"mean": round(float(np.mean(d)), 4), "median": round(float(np.median(d)), 4),
                                       "n_improved": int(sum(x > 0 for x in d)), "n": len(d),
                                       "p_signflip": signflip(d)}
            res[setname] = blk
        R["models"][m] = res
        rb = res.get("replication", {})
        if rb:
            print(f"[{m}] replication n={rb['n_streams']} events={rb['n_events']}: AUC none {rb['none']['auc']:.3f} "
                  f"fixed {rb['fixed']['auc']:.3f} refreshed {rb['refreshed']['auc']:.3f}; per-stream refreshed "
                  f"delta {rb['delta_refreshed']['mean']:+.3f} ({rb['delta_refreshed']['n_improved']}/"
                  f"{rb['delta_refreshed']['n']}) p={rb['delta_refreshed']['p_signflip']}", flush=True)
    np.savez_compressed(out / "preds.npz", _y=np.concatenate(Ys).astype(np.int8), _groups=np.concatenate(Gs),
                        _t=np.concatenate([tt - tt[0] for tt in Ts]).astype(np.float32),
                        **{f"{m}|{arm}": np.concatenate(P[m][arm]).astype(np.float32)
                           for m in heads for arm in arms})
    save_json(R, out / "results.json")
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
