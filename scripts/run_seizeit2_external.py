#!/usr/bin/env python
"""External cohort: SeizeIT2 wearable behind-the-ear EEG (OpenNeuro ds005873, 125 patients, 5 centres).

A different device (two behind-the-ear channels), different centres and a different population from
the CHB-MIT and Siena development cohorts. The question is whether the refreshed baseline -- devised
and replicated on scalp EEG -- carries over unchanged, and how the networks and their test-time
adaptations fare. Everything is fixed in ``results/_seizeit2_manifest.json`` (``--write-manifest``)
before any EEG is read:

  * split: the 125 patients are permuted with seed 20261004; the first half (development) trains the
    models, the second half (held-out) is scored once;
  * training windows: per development patient, up to 360 ictal and 360 non-ictal 8-s windows
    (non-ictal >= 120 s from any seizure) from all its runs; for the normalized arms each patient is
    z-scored (median/MAD) by its earliest third of non-ictal windows;
  * held-out streams: the 12 h of the run that contains the patient's first seizure, positioned with
    about two thirds before it; consecutive 8-s windows at 250 Hz, standardized as in training;
    non-ictal windows within 120 s of a seizure excluded; the first 30 min of non-ictal windows removed
    as the fixed baseline; a stream is scored if a seizure follows that baseline;
  * the refreshed rule unchanged (each window robust-z-scored by the stream's windows 35 to 5 min
    earlier); models: AURA (boosted, logistic), EEGNet, SCCNet, BIOT and the repository CNN with the
    benchmark recipe; arms as on the development streams;
  * primary endpoint: per-stream AUC of boosted AURA with the refreshed baseline minus without, over the
    held-out streams, two-sided sign-flip test, alpha 0.05.

  python scripts/run_seizeit2_external.py --write-manifest
  python scripts/run_seizeit2_external.py --run          # the single pass; a journal refuses a second

-> results/seizeit2_external_<stamp>/{results.json, preds.npz}
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

ROOT = Path(__file__).resolve().parents[1] / "datasets" / "seizeit2"
MANIFEST = REPO / "results" / "_seizeit2_manifest.json"
JOURNAL = REPO / "results" / "_seizeit2_journal.json"
CACHE = REPO / "results" / "_clip_cache" / "seizeit2"
NETS = ["EEGNet", "SCCNet", "BIOT", "deep_cnn_raw"]
FS, WIN_S, MAX_H, BUF_S, PER_CLASS = 250.0, 8.0, 12.0, 120.0, 360


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def runs_of(subj):
    """[(edf, events DataFrame-like list of (onset, duration, type))] in run order."""
    out = []
    for edf in sorted((ROOT / subj / "ses-01" / "eeg").glob("*_eeg.edf")):
        tsv = Path(str(edf).replace("_eeg.edf", "_events.tsv"))
        ev = []
        if tsv.exists():
            lines = tsv.read_text().splitlines()
            hdr = lines[0].split("\t")
            for ln in lines[1:]:
                f = dict(zip(hdr, ln.split("\t")))
                try:
                    ev.append((float(f["onset"]), float(f["duration"]), f["eventType"]))
                except (KeyError, ValueError):
                    continue
        out.append((edf, ev))
    return out


def _seizures(ev):
    return [(o, o + d) for o, d, t in ev if t.startswith("sz")]


def _read(edf):
    import mne
    from math import gcd

    from scipy.signal import resample_poly
    r = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")
    x = r.get_data()[:2].astype(np.float32)
    fs = r.info["sfreq"]
    up, dn = int(FS), int(round(fs)); gg = gcd(up, dn)
    return resample_poly(x, up // gg, dn // gg, axis=1).astype(np.float32)


def _win(x, t0):
    from run_streaming_monitor import _standardise
    n = int(WIN_S * FS)
    return _standardise(x[:, int(t0 * FS):int(t0 * FS) + n])


def training_windows(subj, rng):
    """Up to PER_CLASS ictal and non-ictal 8-s windows from all of a patient's runs, cached."""
    f = CACHE / f"train_{subj}.npz"
    if f.exists():
        d = np.load(f)
        return d["X"], d["y"], d["order"]
    X, y, order = [], [], []
    runs = runs_of(subj)
    ict, non = [], []
    for ri, (edf, ev) in enumerate(runs):
        sz = _seizures(ev)
        import mne
        dur = mne.io.read_raw_edf(edf, preload=False, verbose="ERROR")
        dur = dur.n_times / dur.info["sfreq"]
        for a, b in sz:                                   # ictal: 8-s windows every 4 s inside the seizure
            t = a
            while t + WIN_S <= min(b, dur):
                ict.append((ri, t)); t += 4.0
        t = 0.0
        while t + WIN_S <= dur:                           # non-ictal candidates every 8 s
            if all(t + WIN_S <= a - BUF_S or t >= b + BUF_S for a, b in sz):
                non.append((ri, t))
            t += WIN_S
    if not ict:
        return None
    ict = [ict[i] for i in sorted(rng.choice(len(ict), min(PER_CLASS, len(ict)), replace=False))]
    non = [non[i] for i in sorted(rng.choice(len(non), min(PER_CLASS, len(non)), replace=False))]
    for ri, (edf, _ev) in enumerate(runs):
        need = [(t, 1) for r_, t in ict if r_ == ri] + [(t, 0) for r_, t in non if r_ == ri]
        if not need:
            continue
        x = _read(edf)
        for t, lab in need:
            w = _win(x, t)
            if w.shape[1] == int(WIN_S * FS):
                X.append(w); y.append(lab); order.append(ri * 1e6 + t)
        del x
    X, y, order = np.stack(X), np.asarray(y, np.int64), np.asarray(order)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(f, X=X, y=y, order=order)
    return X, y, order


def stream(subj):
    """The 12 h of the first-seizure run, positioned as on the development cohorts; cached."""
    from run_stream_replication import RULE
    f = CACHE / f"stream_{subj}.npz"
    if f.exists():
        d = np.load(f)
        return d["X"], d["y"], d["t"]
    runs = runs_of(subj)
    first = next(((edf, ev) for edf, ev in runs if _seizures(ev)), None)
    if first is None:
        return None
    edf, ev = first
    sz = _seizures(ev)
    x = _read(edf)
    dur = x.shape[1] / FS
    begin = max(0.0, sz[0][0] - 0.65 * MAX_H * 3600)
    end = min(dur, begin + MAX_H * 3600)
    X, y, t = [], [], []
    tt = begin
    while tt + WIN_S <= end:
        ictal = any(tt + WIN_S > a and tt < b for a, b in sz)
        near = any(tt + WIN_S > a - BUF_S and tt < b + BUF_S for a, b in sz)
        if ictal or not near:
            X.append(_win(x, tt)); y.append(int(ictal)); t.append(tt)
        tt += RULE["stride_s"]
    X, y, t = np.stack(X), np.asarray(y, np.int64), np.asarray(t)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(f, X=X, y=y, t=t)
    return X, y, t


def feats(X, key, n_jobs=12):
    from joblib import Parallel, delayed

    from bsf.data.features import rich_plus
    f = CACHE / f"feat_{key}.npy"
    if f.exists():
        return np.load(f)
    F = np.nan_to_num(np.asarray(Parallel(n_jobs=n_jobs, batch_size=64)(
        delayed(rich_plus)(np.asarray(w, np.float32).T, FS) for w in X), np.float32))
    np.save(f, F)
    return F


def main() -> int:
    from run_stream_replication import RULE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-manifest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--epochs", type=int, default=25)
    a = ap.parse_args()
    subjects = sorted(p.name for p in ROOT.glob("sub-*") if p.is_dir())
    if a.write_manifest:
        if MANIFEST.exists():
            raise SystemExit("manifest exists -- refusing to overwrite")
        if len(subjects) != 125:
            raise SystemExit(f"{len(subjects)} subjects on disk; finish the download first")
        perm = list(np.random.RandomState(20261004).permutation(subjects))
        body = {"dataset": "OpenNeuro ds005873 v1.1.0 (SeizeIT2), behind-the-ear EEG, 2 channels",
                "development": sorted(perm[:63]), "held_out": sorted(perm[63:]),
                "rule": {**RULE, "channels": 2, "fs": FS, "train_per_class": PER_CLASS, "buffer_s": BUF_S,
                         "stream": "12 h of the first-seizure run, ~2/3 before the first seizure"},
                "models": ["feat_gbm", "feat_logistic", *NETS], "epochs": a.epochs,
                "arms": ["none", "fixed", "refreshed", "ea_refreshed", "adabn_refreshed", "tent", "ensemble",
                         "personalized"],
                "primary": "feat_gbm per-stream AUC(refreshed) - AUC(none) over held-out streams; two-sided "
                           "sign-flip, alpha 0.05",
                "events": "2-min causal average; threshold = 90th percentile of the other held-out streams' "
                          "scores (10% time in warning)"}
        MANIFEST.write_text(json.dumps({**body, "sha256": _sha(body),
                                        "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=2))
        print("manifest", _sha(body))
        return 0
    if not a.run:
        print("pass --write-manifest (after the download), commit it, then --run")
        return 0
    if JOURNAL.exists():
        raise SystemExit("the SeizeIT2 pass already ran -- refusing a second")
    M = json.loads(MANIFEST.read_text())
    if _sha({k: v for k, v in M.items() if k not in ("sha256", "created")}) != M["sha256"]:
        raise SystemExit("manifest altered -- refusing")
    JOURNAL.write_text(json.dumps({"started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                   "manifest": M["sha256"]}, indent=2))
    return run_pass(M, a.epochs)


def run_pass(M, epochs) -> int:
    import torch
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from bench_data import _inv_sqrt
    from bsf.models import zoo
    from bsf.utils.compute import resolve_device
    from bsf.utils.io import save_json
    from bsf.utils.provenance import new_results_dir
    from run_final_stats import signflip
    from run_stream_nets import predict, refreshed_ea
    from run_stream_replication import RULE, _rolling, _z, _zr, events_at, smooth
    from run_stream_tta import _bns, adabn_refreshed, tent

    dev = resolve_device("auto")
    out = new_results_dir("seizeit2_external", seed=1234, config={"manifest": M["sha256"]})
    span, gap = RULE["rolling_min"] * 60, RULE["gap_min"] * 60
    n_base = int(round(RULE["fixed_baseline_min"] * 60 / RULE["stride_s"]))
    T_ = int(WIN_S * FS)

    # ---- training windows (development half) ----
    rng = np.random.default_rng(20261004)
    Xs, ys, gs, Fs, base = [], [], [], [], []
    for s in M["development"]:
        r = training_windows(s, rng)
        if r is None:
            continue
        X, y, order = r
        F = feats(X, f"train_{s}")
        neg = np.where(y == 0)[0]
        b = np.zeros(len(y), bool)
        b[neg[np.argsort(order[neg], kind="stable")[:max(5, len(neg) // 3)]]] = True
        Xs.append(X); ys.append(y); gs.append(np.array([s] * len(y))); Fs.append(F); base.append(b)
        print(f"  train {s}: {int(y.sum())} ictal / {int((y == 0).sum())} non-ictal windows", flush=True)
    X, y, g, F, base = (np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs), np.concatenate(Fs),
                        np.concatenate(base))
    del Xs
    Fz = np.empty_like(F); Xea = np.empty_like(X)
    for s in np.unique(g):
        m = g == s
        Fz[m] = _zr(F[m], F[m & base])
        R_ = X[m & base].astype(np.float64)
        Cm = np.einsum("nct,ndt->cd", R_, R_) / (R_.shape[0] * R_.shape[-1])
        Xea[m] = np.einsum("cd,ndt->nct", _inv_sqrt(Cm).astype(np.float32), X[m].astype(np.float32))
    tr = ~base

    # ---- held-out streams ----
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
        Fst = feats(Xst, f"stream_{s}")
        streams[s] = {"X": Xst, "y": yst, "t": tst, "cut": cut, "F": Fst,
                      "Z": _rolling(Fst, tst, cut, span, gap, RULE["min_ref_windows"]),
                      "Zf": _z(Fst[cut:], Fst[neg[:n_base]])}
        print(f"  stream {s}: {len(yst) - cut} scored windows, {int(np.sum(np.diff(np.r_[0, yst[cut:]]) == 1))} seizures",
              flush=True)
    subj = sorted(streams)
    Y = {s: streams[s]["y"][streams[s]["cut"]:] for s in subj}
    T = {s: streams[s]["t"][streams[s]["cut"]:] for s in subj}
    P = {}

    def put(m, arm, s, v):
        P.setdefault(m, {}).setdefault(arm, {})[s] = np.asarray(v, float)

    for m, mk in (("feat_gbm", lambda: GradientBoostingClassifier(n_estimators=200, random_state=0)),
                  ("feat_logistic", lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")))):
        c0 = mk().fit(F[tr], y[tr]); c1 = mk().fit(np.nan_to_num(Fz[tr]), y[tr])
        for s in subj:
            st = streams[s]
            put(m, "none", s, c0.predict_proba(st["F"][st["cut"]:])[:, 1])
            put(m, "fixed", s, c1.predict_proba(st["Zf"])[:, 1])
            put(m, "refreshed", s, c1.predict_proba(st["Z"])[:, 1])
    print("  AURA scored", flush=True)
    for m in NETS:
        for arm, Xtr in (("none", X), ("ea_refreshed", Xea)):
            spec = zoo.build(m, 2, T_, FS, device=dev, seed=0)
            zoo.fit_predict(spec, Xtr[tr], y[tr], Xtr[:1], dev, epochs=epochs, micro_bs=32, seed=0)
            mod = spec.module.eval()
            for s in subj:
                st = streams[s]
                if arm == "none":
                    Xsc = st["X"][st["cut"]:]
                    put(m, "none", s, predict(mod, Xsc, dev))
                    if _bns(mod):
                        put(m, "adabn_refreshed", s, adabn_refreshed(mod, st["X"], st["t"], st["cut"], span, gap, dev))
                    tt = tent(mod, Xsc, dev)
                    if tt is not None:
                        put(m, "tent", s, tt)
                else:
                    put(m, "ea_refreshed", s, predict(mod, refreshed_ea(st["X"], st["t"], st["cut"], span, gap,
                                                                        RULE["min_ref_windows"]), dev))
            del spec, mod
            if dev == "cuda":
                torch.cuda.empty_cache()
        print(f"  {m} scored", flush=True)
    for arm, narm in (("none", "none"), ("refreshed", "ea_refreshed")):
        for s in subj:
            put("ensemble", arm, s, 0.5 * P["feat_gbm"][arm][s] + 0.5 * np.mean([P[m][narm][s] for m in NETS], 0))

    # ---- evaluation ----
    Yc = np.concatenate([Y[s] for s in subj])
    R = {"manifest": M["sha256"], "n_streams": len(subj), "n_seizures": int(sum(np.sum(np.diff(np.r_[0, Y[s]]) == 1) for s in subj)),
         "hours": round(float(sum(len(Y[s]) for s in subj) * WIN_S / 3600), 1), "models": {}}
    for m in P:
        for arm, d in P[m].items():
            per = {s: float(roc_auc_score(Y[s], d[s])) for s in subj}
            cau = nev = fa = 0; hrs = 0.0
            for s in subj:
                oth = np.concatenate([smooth(d[o], T[o]) for o in subj if o != s])
                e = events_at(Y[s], smooth(d[s], T[s]), T[s], float(np.quantile(oth, 0.9)), WIN_S)
                cau += e["caught"]; nev += e["n_events"]; fa += e["false_alarms"]; hrs += len(Y[s]) * WIN_S / 3600
            R["models"][f"{m}|{arm}"] = {"auc": round(float(roc_auc_score(Yc, np.concatenate([d[s] for s in subj]))), 4),
                                         "median_stream_auc": round(float(np.median(list(per.values()))), 4),
                                         "per_stream_auc": {s: round(v, 4) for s, v in per.items()},
                                         "caught": cau, "n_events": nev, "fa_per_24h": round(24 * fa / hrs, 2)}
    pf = R["models"]
    for key, r in pf.items():
        m, arm = key.split("|")
        if arm != "none" and f"{m}|none" in pf:
            dd = [r["per_stream_auc"][s] - pf[f"{m}|none"]["per_stream_auc"][s] for s in subj]
            r["delta_vs_none"] = {"mean": round(float(np.mean(dd)), 4), "median": round(float(np.median(dd)), 4),
                                  "n_improved": int(sum(x > 0 for x in dd)), "n": len(dd), "p_signflip": signflip(dd)}
        if key != "feat_gbm|refreshed":
            dd = [r["per_stream_auc"][s] - pf["feat_gbm|refreshed"]["per_stream_auc"][s] for s in subj]
            r["delta_vs_aura_refreshed"] = {"mean": round(float(np.mean(dd)), 4), "n_better": int(sum(x > 0 for x in dd)),
                                            "p_signflip": signflip(dd)}
    R["primary"] = pf["feat_gbm|refreshed"]["delta_vs_none"]
    # first-seizure personalization on streams with later seizures
    per = {}
    for s in subj:
        st = streams[s]; Z = st["Z"]; yy = Y[s]; tt = T[s]
        idx = np.flatnonzero(yy)
        runs = np.split(idx, np.where((np.diff(idx) > 1) | (np.diff(tt[idx]) > 1.5 * WIN_S))[0] + 1)
        if len(runs) < 2:
            continue
        first = runs[0]; neg = np.arange(0, first[0]); neg = neg[yy[neg] == 0]
        after = np.arange(first[-1] + 1, len(yy))
        if len(neg) < 10 or yy[after].sum() == 0:
            continue
        sc = StandardScaler().fit(Z[:first[-1] + 1])
        lr = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000).fit(
            sc.transform(np.concatenate([Z[first], Z[neg]])), np.r_[np.ones(len(first)), np.zeros(len(neg))])
        pop = P["feat_gbm"]["refreshed"][s][after]
        comb = 0.5 * pop + 0.5 * lr.predict_proba(sc.transform(Z[after]))[:, 1]
        per[s] = {"population": float(roc_auc_score(yy[after], pop)), "combined": float(roc_auc_score(yy[after], comb))}
    if per:
        dd = [v["combined"] - v["population"] for v in per.values()]
        R["personalization"] = {"n_streams": len(per), "median_population": round(float(np.median([v["population"] for v in per.values()])), 4),
                                "median_combined": round(float(np.median([v["combined"] for v in per.values()])), 4),
                                "mean_delta": round(float(np.mean(dd)), 4), "n_improved": int(sum(x > 0 for x in dd)),
                                "p_signflip": signflip(dd)}
    np.savez_compressed(out / "preds.npz", _y=Yc.astype(np.int8), _groups=np.concatenate([[s] * len(Y[s]) for s in subj]),
                        _t=np.concatenate([T[s] - T[s][0] for s in subj]).astype(np.float32),
                        **{f"{m}|{arm}": np.concatenate([P[m][arm][s] for s in subj]).astype(np.float32) for m in P for arm in P[m]})
    save_json(R, out / "results.json")
    j = json.loads(JOURNAL.read_text()); j.update({"finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                                   "results": str(out)})
    JOURNAL.write_text(json.dumps(j, indent=2))
    pr = R["primary"]
    print(f"PRIMARY (SeizeIT2, {len(subj)} held-out streams): refreshed - none {pr['mean']:+.3f} "
          f"({pr['n_improved']}/{pr['n']}), p={pr['p_signflip']}", flush=True)
    for key in ("feat_gbm|none", "feat_gbm|refreshed", "ensemble|refreshed"):
        print(f"  {key}: AUC {pf[key]['auc']:.3f} events {pf[key]['caught']}/{pf[key]['n_events']} FA/24h {pf[key]['fa_per_24h']}",
              flush=True)
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
