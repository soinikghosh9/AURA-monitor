#!/usr/bin/env python
"""T2: continuous monitoring. What is the risk right now, and how often am I in warning?

The other tasks score a model on curated clips drawn from a record. That is the right way to compare
architectures and the wrong way to describe deployment, because a deployed monitor does not receive a
balanced sample -- it receives every window, most of them uninteresting, and its errors are counted in
false alarms per day rather than in AUC.

So this scores CONTIGUOUS windows across whole scalp recordings, pseudo-prospectively:

  * The model is trained on OTHER subjects and never on the record it scores, so nothing about that
    patient's own future is available at scoring time.
  * Windows are consecutive, in chronological order, and spaced by the patient's real clock, so the
    elapsed time behind "false alarms per day" is elapsed time. Nothing is resampled or class
    balanced; prevalence comes out at roughly 5-17% on these stretches rather than the ~50% a curated
    clip set has.
  * Two honest departures from "everything the monitor saw". The streamed stretch is POSITIONED to
    contain a seizure, because a stretch without one can only accrue false alarms and leaves
    sensitivity undefined -- so this is a seizure-enriched slice of monitoring time, not a
    representative day. And windows whose label is genuinely undefined (the seizure itself, the SPH
    gap, a short post-ictal period) are excluded rather than forced into the negative class.
  * The metrics are the clinical currency rather than the ML one: sensitivity at a fixed
    time-in-warning budget, FALSE ALARMS PER DAY, time-in-warning actually incurred, and detection
    latency from electrographic onset.
  * The chance model is the Winterhalder/Snyder analytic one with Benjamini-Hochberg control, and its
    Bernoulli N is the number of independent seizure EVENTS, never the number of windows -- the single
    most common way this literature reports significance it does not have.

Two modes, because they are different clinical claims and conflating them is how a detector gets sold
as a forecaster:
  --mode detect    is a seizure happening now?     (positive = ictal window)
  --mode forecast  is one coming within the horizon? (positive = within `--horizon-min` before onset,
                   with the same SPH gap the forecasting task uses)

-> results/streaming_monitor_<stamp>/results.json + streaming.csv
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

from extract_scalp_raw import (  # noqa: E402
    CHB, SIENA, chb_seizures, siena_seizures, siena_spans)
from run_unified_benchmark import _load_detect, _load_scalp  # noqa: E402

SEAL = Path("results/_blind_split.json")


def _standardise(w: np.ndarray) -> np.ndarray:
    """One streamed window, normalised EXACTLY as every training clip was.

    Training sub-windows are z-scored per channel when cached (``subwindows``) and then robustly
    re-standardised per channel by the benchmark pool (``_pool``). An earlier version of this script
    streamed the raw EDF samples, in volts (~1e-5), into models trained on unit-variance input: the
    models saw near-zero signals, one of them emitted a constant, and the resulting AUCs, warning
    times and false-alarm rates described that mismatch rather than the models. Stored as float16.
    """
    w = np.nan_to_num(np.asarray(w, np.float32))
    w = (w - w.mean(1, keepdims=True)) / (w.std(1, keepdims=True) + 1e-6)
    med = np.median(w, axis=1, keepdims=True)
    mad = np.median(np.abs(w - med), axis=1, keepdims=True) * 1.4826 + 1e-6
    return np.clip((w - med) / mad, -20, 20).astype(np.float16)


def _stream_record(pdir, cohort, win_s, target_fs, stride_s, max_hours, montage="native"):
    """Decode a CONTIGUOUS stretch of one patient's monitoring into consecutive windows.

    Returns (X, t0s, spans, fs) with ``t0s`` on the patient's real clock, so the gaps between
    recordings are real gaps and the elapsed time behind "false alarms per day" is elapsed time.

    Two things this deliberately does NOT do. It does not order files by how many seizures they
    contain: a first version did, and the result was a re-ordered concatenation of a patient's most
    eventful hours, which is not a recording anyone is ever monitored on -- prevalence came out at
    38-67% and every rate derived from it was fiction. And it does not start at the beginning and
    hope: a stretch with no seizure in it can only accrue false alarms, leaving sensitivity undefined.

    Instead the window is a real, chronologically ordered stretch positioned to contain the patient's
    first seizure, with roughly two thirds of the budget spent before it. That is a biased sample of
    monitoring time and is stated as such, but every window inside it is in the order and at the
    spacing the patient actually experienced.
    """
    import mne
    from scipy.signal import resample_poly

    from extract_scalp_raw import timeline

    edfs = sorted(pdir.glob("*.edf"))
    if not edfs:
        return None
    tl = timeline("chbmit" if cohort == "chbmit" else "siena", pdir, edfs)
    starts = tl["starts"]
    ordered = [e for e in sorted(edfs, key=lambda e: starts.get(e.name, float("inf")))
               if e.name in starts]
    if not ordered or not tl["spans"]:
        return None
    first_sz = min(on for on, _off in tl["spans"])
    rec_start = min(starts[e.name] for e in ordered)
    # Clamp to the record start BEFORE fixing the end, so the full budget is always spent. Subtracting
    # blind gave chb03 a 1.7 h stream out of a 10 h budget -- its first seizure is minutes into the
    # recording, so most of the requested lead-in was time that does not exist, and the surviving
    # stretch was 77% pre-ictal.
    begin = max(rec_start, first_sz - 0.65 * max_hours * 3600.0)
    end = begin + max_hours * 3600.0

    sz = chb_seizures(pdir) if cohort == "chbmit" else None
    X, t0s, spans, fs_out = [], [], [], None
    for edf in ordered:
        t0 = starts[edf.name]
        try:
            r = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")
        except Exception:  # noqa: BLE001
            continue
        fs = r.info["sfreq"]; dur = r.n_times / fs
        if t0 + dur < begin or t0 > end:
            del r
            continue                                       # outside the streamed stretch
        raw = r.get_data()
        if montage == "db18":                             # common double banana (scripts/montage.py)
            from montage import db18
            raw = db18(raw, r.ch_names)
            if raw is None:
                del r
                continue
        # one implementation of the clock->offset mapping, shared with the extractor, so a fix to the
        # file-name matching or the midnight unwrap cannot land in one place and not the other
        sp = ([(a, b) for a, b in sz.get(edf.name, []) if 0 <= a <= dur] if cohort == "chbmit"
              else siena_spans(pdir, edf, r))
        up, dn = int(round(target_fs)), int(round(fs))
        from math import gcd
        gg = gcd(up, dn)
        t = max(0.0, begin - t0)
        while t + win_s <= dur and t0 + t < end:
            seg = raw[:, int(t * fs):int((t + win_s) * fs)]
            if seg.shape[1] >= int(win_s * fs) - 2:
                s = resample_poly(seg, up // gg, dn // gg, axis=1)
                need = int(round(win_s * target_fs))
                if s.shape[1] >= need:
                    X.append(_standardise(s[:, :need]))
                    t0s.append(t0 + t)                     # real clock, not a running concatenation
            t += stride_s
        spans += [(t0 + a, t0 + b) for a, b in sp]
        fs_out = target_fs
        del r, raw
    if not X:
        return None
    n_ch = min(x.shape[0] for x in X)
    return (np.stack([x[:n_ch] for x in X]), np.asarray(t0s), spans, fs_out)


def _labels(t0s, spans, win_s, mode, sph_min, horizon_min, postictal_min=30.0,
            detect_buffer_s=120.0):
    """Per-window label, the event each positive belongs to, and which windows are USABLE.

    The third return value is the point. In forecast mode a window that is itself ictal, or sits in
    the SPH gap, or is postictal, is not a negative -- "no seizure is coming" is false during a
    seizure and meaningless just after one. Left in, they inflate the negative class with exactly the
    high-amplitude activity that makes the task look easy, and the false-alarm rate is then computed
    against a denominator that includes seizures. They are excluded rather than relabelled, because
    there is no honest label for them under this question.
    """
    t0s = np.asarray(t0s, float)
    y = np.zeros(len(t0s), int)
    ev = np.full(len(t0s), -1)
    usable = np.ones(len(t0s), bool)
    # Neither mode uses the curated arm's four-hour interictal guard. That guard exists to select
    # CLEAN negatives for training, and importing it into a streaming evaluation destroys the very
    # thing the evaluation is for. Applied to detection it left no scoreable stream at all; applied to
    # forecasting it deleted 563 of 900 windows and drove prevalence to 67%, when the entire point of
    # scoring a continuous record is to face the low prevalence a monitor actually meets and to make
    # false-alarms-per-day mean something. Here the exclusion covers only the interval whose label is
    # genuinely undefined: the seizure itself, the SPH gap, and a short post-ictal settling period.
    guard = float(detect_buffer_s) if mode == "detect" else postictal_min * 60.0
    for k, (on, off) in enumerate(spans):
        ictal = (t0s + win_s > on) & (t0s < off)
        if mode == "detect":
            y[ictal] = 1
            ev[ictal] = k
            # a non-ictal window must be genuinely far from every seizure, matching the guard the
            # curated detection clips use, so the two tables describe the same negative class
            near = (t0s + win_s > on - guard) & (t0s < off + guard) & ~ictal
            usable &= ~near
        else:
            lo, hi = on - sph_min * 60 - horizon_min * 60, on - sph_min * 60
            pre = (t0s >= lo) & (t0s + win_s <= hi)
            y[pre] = 1
            ev[pre] = k
            excl = (t0s + win_s > hi) & (t0s < off + guard)   # SPH gap, ictal, postictal
            usable &= ~(excl & ~pre)
    return y, ev, usable


def _feat_logistic():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))


def _feat_gbm():
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(n_estimators=200, random_state=0)


FEATURE_ARMS = {"feat_logistic": _feat_logistic, "feat_gbm": _feat_gbm}
# The analytic arms are fitted and scored on the SAME input the networks get: the 18-D vector of each
# standardised 8-s training sub-window (with its clip's label, as the networks are trained) and of each
# standardised streamed window, on the same channels and samples. The first v2 run fitted them on the
# cache's clip-level vector (one raw 60-s clip) and scored them on standardised 8-s windows, so their
# streaming numbers measured that mismatch; results without this marker are recomputed on resume.
FEATURE_INPUT = "standardised_8s_window"


def _stream_features(st, mode, fs, n_ch, n_t, n_jobs=6):
    """The 18-D vector for every streamed window (standardised, first ``n_ch`` channels and ``n_t``
    samples, as the networks see it). Cached per stream: ~5,400 windows per 12 h, PAC dominates."""
    import hashlib

    from joblib import Parallel, delayed

    from bsf.data.features import rich_plus
    key = hashlib.sha1(np.ascontiguousarray(st["X"][:, 0, :64]).tobytes()).hexdigest()[:12]
    f = Path(f"results/_clip_cache/streamfeat3_{mode}_{st['subject']}_c{n_ch}t{n_t}_{key}.npy")
    if f.exists():
        return np.load(f)
    full = Path(f"results/_clip_cache/streamfeat_{mode}_{st['subject']}_{key}.npy")
    if full.exists() and st["X"].shape[1:] == (n_ch, n_t):
        return np.load(full)         # the earlier cache is the same computation on the whole window
    F = Parallel(n_jobs=n_jobs, batch_size=64)(
        delayed(rich_plus)(np.asarray(w[:n_ch, :n_t], np.float32).T, fs) for w in st["X"])
    F = np.nan_to_num(np.asarray(F, np.float32))
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, F)
    return F


def _train_window_features(W, fs, n_jobs=6):
    """The 18-D vector of every standardised training sub-window, (n_clips, n_subwin, 18); cached."""
    import hashlib

    from joblib import Parallel, delayed

    from bsf.data.features import rich_plus
    n, k, c, t = W.shape
    key = hashlib.sha1(np.ascontiguousarray(W[:, :, 0, :64]).tobytes()).hexdigest()[:12]
    f = Path(f"results/_clip_cache/featwin_{n}x{k}_c{c}t{t}_{key}.npy")
    if f.exists():
        return np.load(f)
    flat = Parallel(n_jobs=n_jobs, batch_size=32)(
        delayed(rich_plus)(np.asarray(W[i, j], np.float32).T, fs) for i in range(n) for j in range(k))
    F = np.nan_to_num(np.asarray(flat, np.float32)).reshape(n, k, -1)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, F)
    return F


def _recalibrate(Y, P, G, prior_tr):
    """Two corrections that never see the scored patient's labels.

    PRIOR SHIFT: every model is trained on class-balanced clips, so its probabilities are calibrated
    to a ~50% prevalence a monitor never meets. Shifting the logit by log-odds(deployment prevalence)
    minus log-odds(training prevalence) is the textbook correction (Saerens et al., 2002); the
    deployment prevalence is estimated from the OTHER patients' streams.
    LOSO PLATT: a two-parameter map fitted on the other patients' streamed predictions, applied to the
    held-out patient -- the post-pooling, held-out-subject recalibration the benchmark recommends.
    """
    from sklearn.linear_model import LogisticRegression

    from bsf.eval import clinical as C
    pc = np.clip(P, 1e-6, 1 - 1e-6)
    z = np.log(pc / (1 - pc))
    q_prior = np.empty_like(pc); q_platt = np.empty_like(pc)
    for s in np.unique(G):
        m = G == s; o = ~m
        pi = float(np.clip(Y[o].mean(), 1e-4, 1 - 1e-4)); pt = prior_tr.get(s, 0.5)
        q_prior[m] = 1 / (1 + np.exp(-(z[m] + np.log(pi / (1 - pi)) - np.log(pt / (1 - pt)))))
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(z[o][:, None], Y[o])
        a = float(lr.coef_[0][0])
        q_platt[m] = (lr.predict_proba(z[m][:, None])[:, 1] if a > 0 else np.full(m.sum(), pi))
    out = {}
    for name, q in (("raw", pc), ("prior_shift", q_prior), ("loso_platt", q_platt)):
        out[name] = {"brier_skill": round(C.brier_skill(Y, q), 4),
                     "ece": round(C.calibration_error(Y, q), 4),
                     "mean_predicted": round(float(np.mean(q)), 4),
                     "prevalence": round(float(np.mean(Y)), 4)}
    return out


def main() -> int:
    import torch

    from bsf.eval import clinical as C
    from bsf.models import zoo
    from bsf.utils import resume
    from bsf.utils.compute import resolve_device
    from bsf.utils.io import save_json
    from bsf.utils.provenance import new_results_dir
    from bsf.utils.seeds import set_global_seeds

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="detect", choices=["detect", "forecast"])
    ap.add_argument("--models", default="feat_logistic,feat_gbm,BIOT,ContraWR,SCCNet,"
                                        "deep_cnn_raw,EEGNet,ShallowFBCSPNet")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--win-s", type=float, default=8.0)
    ap.add_argument("--stride-s", type=float, default=8.0)
    ap.add_argument("--target-fs", type=float, default=250.0)
    ap.add_argument("--max-hours", type=float, default=12.0,
                    help="per scored record. Long on purpose: a few hours around a "
                         "seizure cannot contain a realistic interictal stretch, and "
                         "false alarms per day computed over such a slice is fiction.")
    ap.add_argument("--max-records", type=int, default=6, help="DEV subjects to stream")
    ap.add_argument("--sph-min", type=float, default=5.0)
    ap.add_argument("--horizon-min", type=float, default=30.0)
    ap.add_argument("--postictal-min", type=float, default=30.0,
                    help="forecast mode: minutes after a seizure whose label is undefined. NOT the "
                         "curated arm's 4 h interictal guard -- see _labels.")
    ap.add_argument("--detect-buffer-s", type=float, default=120.0,
                    help="detect mode: a non-ictal window need only clear the ictal boundary by this")
    ap.add_argument("--max-per-class", type=int, default=60)
    ap.add_argument("--n-subwin", type=int, default=6)
    ap.add_argument("--micro-bs", type=int, default=16)
    ap.add_argument("--budgets", default="0.05,0.10,0.25")
    args = ap.parse_args()
    set_global_seeds(1234)
    dev = resolve_device("auto")
    seal = json.loads(SEAL.read_text())
    out = new_results_dir("streaming_monitor", seed=1234,
                          config={**vars(args), "seal": seal["sha256"]})
    budgets = tuple(float(b) for b in args.budgets.split(",") if b.strip())
    names = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"=== T2 streaming monitor [{args.mode}] (DEV only, device={dev}) ===", flush=True)

    # Training pool: the SAME curated clips the other tasks use, so the only thing that changes
    # between this table and the benchmark is how the model is SCORED, not how it was fitted.
    train = (_load_detect if args.mode == "detect" else _load_scalp)(args.max_per_class,
                                                                     args.n_subwin)
    if train is None:
        print("  no DEV training data (run extract_scalp_raw.py first) -- nothing to do", flush=True)
        save_json({"skipped": "no training data", "mode": args.mode}, out / "results.json")
        return 0
    Wtr, ytr, gtr, _ = train
    n_sub, n_ch_tr, n_t_tr = Wtr.shape[1], Wtr.shape[2], Wtr.shape[3]
    print(f"  train pool: {len(ytr)} clips / {len(np.unique(gtr))} DEV subjects, "
          f"{n_ch_tr}ch x {n_t_tr}", flush=True)

    dev_subj = {("chbmit", s) for s in seal["cohorts"].get("scalp_chbmit", {}).get("dev", [])}
    dev_subj |= {("siena", s) for s in seal["cohorts"].get("scalp_siena", {}).get("dev", [])}
    streams, used = [], []
    for cohort, s in sorted(dev_subj):
        if len(streams) >= args.max_records:
            break
        pdir = (CHB if cohort == "chbmit" else SIENA) / s
        if not pdir.is_dir():
            continue
        rec = _stream_record(pdir, cohort, args.win_s, args.target_fs, args.stride_s,
                             args.max_hours)
        if rec is None or not rec[2]:
            continue                       # no annotated event in the streamed span: nothing to catch
        X, t0s, spans, _ = rec
        y, ev, usable = _labels(t0s, spans, args.win_s, args.mode, args.sph_min,
                                args.horizon_min, args.postictal_min, args.detect_buffer_s)
        X, t0s, y, ev = X[usable], t0s[usable], y[usable], ev[usable]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        streams.append({"subject": f"{cohort}_{s}", "X": X, "y": y, "ev": ev, "spans": spans,
                        "t0s": t0s, "n_excluded": int((~usable).sum())})
        used.append(f"{cohort}_{s}")
        print(f"  stream {cohort}_{s}: {len(y)} windows "
              f"({100*y.mean():.2f}% positive, {len(spans)} seizures, "
              f"{len(y)*args.stride_s/3600:.1f} h, "
              f"{int((~usable).sum())} unlabelable windows excluded)", flush=True)
    if not streams:
        print("  no scoreable DEV streams found", flush=True)
        save_json({"skipped": "no streams", "mode": args.mode}, out / "results.json")
        return 0

    hours_per_window = args.stride_s / 3600.0
    # v2: windows standardised like the training clips; the v1 checkpoints scored raw volts
    unit = f"stream_v2_{args.mode}_e{args.epochs}_r{len(streams)}"
    results = dict(resume.done_units(unit))

    for mname in names:
        if mname in results and not (mname in FEATURE_ARMS
                                     and results[mname].get("feature_input") != FEATURE_INPUT):
            print(f"  (resume) {mname}", flush=True); continue
        try:
            per_subject, pooled_y, pooled_p, pooled_g = {}, [], [], []
            prior_tr = {}
            for st in streams:
                # leave-this-subject-out: never train on the record being scored
                keep = gtr != st["subject"]
                prior_tr[st["subject"]] = float(ytr[keep].mean())
                n_ch = min(n_ch_tr, st["X"].shape[1])
                n_t = min(n_t_tr, st["X"].shape[2])
                spec = None
                if mname in FEATURE_ARMS:
                    Fs = _stream_features(st, args.mode, args.target_fs, n_ch, n_t)
                    Ftr = _train_window_features(Wtr[:, :, :n_ch, :n_t], args.target_fs)
                    clf = FEATURE_ARMS[mname]().fit(Ftr[keep].reshape(-1, Ftr.shape[-1]),
                                                    np.repeat(ytr[keep], n_sub))
                    p = clf.predict_proba(Fs)[:, 1]
                else:
                    spec = zoo.build(mname, n_ch, n_t, args.target_fs, device=dev, seed=0)
                    p = zoo.fit_predict(spec,
                                        Wtr[keep][:, :, :n_ch, :n_t].reshape(-1, n_ch, n_t),
                                        np.repeat(ytr[keep], n_sub),
                                        st["X"][:, :n_ch, :n_t], dev,
                                        epochs=args.epochs, micro_bs=args.micro_bs)
                y = st["y"]
                panel = C.full_panel(y, p, hours_per_window=hours_per_window, budgets=budgets,
                                     n_boot=200)
                ev_hit = {}
                for k in sorted(set(st["ev"][st["ev"] >= 0])):
                    m = st["ev"] == k
                    ev_hit[int(k)] = float(p[m].max()) if m.any() else None
                per_subject[st["subject"]] = {
                    "panel": panel, "n_windows": int(len(y)), "n_events": len(st["spans"]),
                    "hours": round(len(y) * hours_per_window, 2),
                    "prevalence": round(float(y.mean()), 5),
                    "event_peak_scores": ev_hit}
                pooled_y.append(y); pooled_p.append(p)
                pooled_g.append(np.array([st["subject"]] * len(y)))
                del spec
                if dev == "cuda":
                    torch.cuda.empty_cache()

            Y = np.concatenate(pooled_y); P = np.concatenate(pooled_p)
            G = np.concatenate(pooled_g)
            pooled = C.full_panel(Y, P, groups=G, hours_per_window=hours_per_window,
                                  budgets=budgets, n_boot=400)
            pooled["recalibrated"] = _recalibrate(Y, P, G, prior_tr)
            n_events = sum(len(s["spans"]) for s in streams)
            results[mname] = {"pooled": pooled, "per_subject": per_subject,
                              "n_events": n_events,
                              "n_params": (None if mname in FEATURE_ARMS else zoo.probe(
                                  mname, n_ch_tr, n_t_tr, args.target_fs, dev).get("n_params")),
                              "family": "features" if mname in FEATURE_ARMS
                              else zoo.family_of(mname),
                              "feature_input": FEATURE_INPUT if mname in FEATURE_ARMS else None}
            resume.update_unit(unit, mname, results[mname])
            b10 = pooled["clinical"].get("budget_10pct", {})
            ay = pooled["clinical"].get("at_youden", {})
            print(f"  {mname:20s} AUC {pooled['discrimination']['auc']:.3f}  "
                  f"sens@10%TiW {b10.get('event_sensitivity')} "
                  f"(chance {b10.get('chance_sensitivity')}, "
                  f"beats={b10.get('beats_chance')})  "
                  f"FA/day {ay.get('false_alarms_per_day')}  "
                  f"n_events {n_events}", flush=True)
        except Exception as e:  # noqa: BLE001
            results[mname] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
            print(f"  {mname:20s} FAILED {str(e)[:80]}", flush=True)
            if dev == "cuda":
                torch.cuda.empty_cache()

    R = {"protocol": f"T2 continuous monitoring [{args.mode}]: consecutive windows in record order, "
                     f"model trained on OTHER DEV subjects, unbalanced natural prevalence. "
                     f"Event-level N = seizures, not windows.",
         "blind_seal": seal["sha256"], "streams": used,
         "hours_per_window": hours_per_window, "models": results}
    save_json(R, out / "results.json")

    rows = ["mode,model,family,n_params,auc,auprc,prevalence,n_events,"
            + ",".join(f"sens_at_{int(b*100)}pct_tiw,beats_chance_{int(b*100)}pct" for b in budgets)
            + ",false_alarms_per_day,time_in_warning,median_latency_h,brier_skill,ece"]
    for m, r in results.items():
        if not isinstance(r, dict) or "pooled" not in r:
            continue
        pl = r["pooled"]
        sens = []
        for b in budgets:
            d = pl["clinical"].get(f"budget_{int(b*100)}pct", {})
            sens += [str(d.get("event_sensitivity", "")), str(d.get("beats_chance", ""))]
        ay = pl["clinical"].get("at_youden", {})
        rows.append(f"{args.mode},{m},{r.get('family','')},{r.get('n_params','')},"
                    f"{pl['discrimination']['auc']},{pl['discrimination']['auprc']},"
                    f"{pl['prevalence']},{r['n_events']}," + ",".join(sens)
                    + f",{ay.get('false_alarms_per_day','')},{ay.get('time_in_warning','')},"
                    f"{ay.get('median_detection_latency_h','')},"
                    f"{pl['calibration']['brier_skill']},{pl['calibration']['ece']}")
    (out / "streaming.csv").write_text("\n".join(rows))
    print(f"\n-> {out}  ({len(rows) - 1} rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
