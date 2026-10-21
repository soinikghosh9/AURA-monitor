#!/usr/bin/env python
"""Clinically oriented post-hoc analyses of the locked continuous-EEG predictions (audit Phase 3, CPU only).
No model is retrained; every input is a preds.npz written by a locked pass. All analyses here are post hoc.

  A1 threshold portability  each recording scored at the 6-per-24-h threshold set on the other recordings:
                            the false-alarm rate a new patient actually gets, with and without normalization
  A2 recalibration          intercept and slope on the logit, fitted on the other recordings (cross-fitted) and,
                            for boosted AURA, on the development streams and applied to the held-out streams
  A3 subgroups              children (CHB-MIT) against adults (Siena); seizure duration strata
  A4 review workload        alarms per 24 h, share of time under alarm, alarms reviewed per detected seizure,
                            detection latency
  A5 ensemble agreement     alarm precision by agreement among AURA-Ens members (development, double banana)

  python scripts/run_phase3_posthoc.py   -> results/phase3_posthoc_<stamp>/results.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from bsf.eval import szcore  # noqa: E402
from run_phase2_rescore import SETS, WIN_S, calibration, smooth, streams_of  # noqa: E402

RES = ROOT / "results"
BUDGET = 6.0
MODELS = {
    "heldout": ["feat_gbm|none", "feat_gbm|refreshed", "BIOT|none", "BIOT|probe_gbm_refreshed", "SCCNet|none",
                "SCCNet|probe_gbm_refreshed", "EEGNet|none", "EEGNet|probe_gbm_refreshed", "deep_cnn_raw|none",
                "deep_cnn_raw|probe_gbm_refreshed", "ensemble|refreshed"],
    "seizeit2": ["feat_gbm|none", "feat_gbm|refreshed", "BIOT|none", "EEGNet|none", "SCCNet|none",
                 "deep_cnn_raw|none", "ensemble|none"],
    "dev_db18": ["feat_gbm|refreshed", "aura_ens|post", "ens_mean|post"] + [
        f"{n}|{a}" for n in ("BIOT", "SCCNet", "EEGNet", "ATCNet", "ContraWR", "deep_cnn_raw")
        for a in ("none", "probe_gbm_refreshed")],
}
MEMBERS = ["feat_gbm|refreshed", "auranet_hybrid|refreshed"] + [
    f"{n}|probe_gbm_refreshed" for n in ("BIOT", "SCCNet", "EEGNet", "ATCNet", "ContraWR", "deep_cnn_raw")]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def crossfit_thresholds(sm, budget=BUDGET):
    names = sorted(sm)
    return {s: szcore.threshold_for_budget([sm[o] for o in names if o != s], budget, WIN_S) for s in names}


def seizure_spans(y, t):
    """Annotated seizures as (onset, offset) from the per-window labels."""
    out, start = [], None
    for i in range(len(y)):
        if y[i] and start is None:
            start = t[i]
        if start is not None and (i == len(y) - 1 or not y[i + 1] or t[i + 1] - t[i] > WIN_S + 0.5):
            out.append((start, t[i] + WIN_S)); start = None
    return out


def detected(spans, hyp_mask, t, pre=30.0, post=60.0):
    """Per seizure: detected (alarm overlapping the SzCORE tolerance) and latency from onset."""
    res = []
    for a, b in spans:
        m = hyp_mask & (t + WIN_S > a - pre) & (t < b + post)
        res.append((bool(m.any()), float(max(0.0, t[m][0] - a)) if m.any() else None, b - a))
    return res


def recal_fit(lp, y):
    """Logistic recalibration a + b * logit(p) by Newton's method."""
    X = np.c_[np.ones(len(lp)), lp]
    w = np.array([0.0, 1.0])
    for _ in range(50):
        mu = 1 / (1 + np.exp(-(X @ w)))
        H = X.T @ (X * (mu * (1 - mu))[:, None]) + 1e-9 * np.eye(2)
        w = w + np.linalg.solve(H, X.T @ (y - mu))
    return w


def main() -> int:
    out_dir = RES / f"phase3_posthoc_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir.mkdir(parents=True)
    D = {k: np.load(v / "preds.npz", allow_pickle=True) for k, v in SETS.items() if (v / "preds.npz").exists()}
    R: dict = {"note": "post hoc; locked predictions; peri-ictal windows (120 s) absent", "budget_per_24h": BUDGET,
               "A1_portability": {}, "A2_recalibration": {}, "A3_subgroups": {}, "A4_workload": {},
               "A5_agreement": {}}

    for k, keys in MODELS.items():
        for key in keys:
            if key not in D[k].files:
                continue
            st = streams_of(D[k], key)
            sm = {s: (y, smooth(p, t), t) for s, (y, p, t) in st.items()}
            thr = crossfit_thresholds(sm)
            # A1: false alarms each new recording actually receives
            fa, sens_rec = {}, {}
            per_seiz = []
            alarm_s, hours, n_alarm, tp = 0.0, 0.0, 0, 0
            for s, (y, p, t) in sm.items():
                e = szcore.events(y, p, t, thr[s], WIN_S)
                fa[s] = 24 * e["fp"] / e["hours"]
                hours += e["hours"]; n_alarm += e["n_hyp"]; tp += e["tp_hyp"]
                alarm_s += float((p >= thr[s]).sum() * WIN_S)
                for det, lat, dur in detected(seizure_spans(y, t), p >= thr[s], t):
                    per_seiz.append({"rec": s, "det": det, "lat": lat, "dur": dur})
            v = np.array(list(fa.values()))
            R["A1_portability"].setdefault(k, {})[key] = {
                "median_fa": round(float(np.median(v)), 2), "iqr_fa": [round(float(x), 2) for x in np.percentile(v, [25, 75])],
                "max_fa": round(float(v.max()), 1), "share_over_2x_budget": round(float(np.mean(v > 2 * BUDGET)), 3),
                "pooled_fa": round(24 * (n_alarm - tp) / hours, 2)}
            # A4: review workload at the 6-per-24-h operating point
            nd = sum(x["det"] for x in per_seiz)
            lats = [x["lat"] for x in per_seiz if x["det"]]
            R["A4_workload"].setdefault(k, {})[key] = {
                "alarms_per_24h": round(24 * n_alarm / hours, 2), "share_time_in_alarm": round(alarm_s / 3600 / hours, 4),
                "alarms_per_detected_seizure": round(n_alarm / nd, 1) if nd else None,
                "seizures_detected": f"{nd}/{len(per_seiz)}",
                "median_latency_s": round(float(np.median(lats)), 1) if lats else None}
            # A3: children vs adults, and seizure duration strata
            sub = {}
            for name, pre in (("CHB-MIT (children)", "chbmit_"), ("Siena (adults)", "siena_")):
                recs = [s for s in st if s.startswith(pre)]
                if not recs:
                    continue
                aucs = [roc_auc_score(st[s][0], st[s][1]) for s in recs]
                ps = [x for x in per_seiz if x["rec"].startswith(pre)]
                sub[name] = {"n_rec": len(recs), "median_rec_auc": round(float(np.median(aucs)), 3),
                             "detected": f"{sum(x['det'] for x in ps)}/{len(ps)}"}
            for lo, hi in ((0, 30), (30, 60), (60, 1e9)):
                ps = [x for x in per_seiz if lo <= x["dur"] < hi]
                if ps:
                    sub[f"duration {lo}-{'inf' if hi > 1e8 else int(hi)} s"] = f"{sum(x['det'] for x in ps)}/{len(ps)}"
            R["A3_subgroups"].setdefault(k, {})[key] = sub
            # A2: cross-fitted logistic recalibration
            names = sorted(st)
            ys, ps_ = [], []
            for s in names:
                Ltr = np.concatenate([logit(st[o][1]) for o in names if o != s])
                Ytr = np.concatenate([st[o][0] for o in names if o != s])
                w = recal_fit(Ltr, Ytr)
                ps_.append(1 / (1 + np.exp(-(w[0] + w[1] * logit(st[s][1]))))); ys.append(st[s][0])
            rc = {s: (ys[i], ps_[i], st[s][2]) for i, s in enumerate(names)}
            R["A2_recalibration"].setdefault(k, {})[key] = {"raw": calibration(st), "crossfit_recalibrated": calibration(rc)}
            print(k, key, R["A1_portability"][k][key], R["A4_workload"][k][key], flush=True)

    # A2b: boosted AURA recalibrated on the development streams, applied unchanged to the held-out streams
    dev = D["dev"]
    for arm in ("none", "refreshed"):
        key = f"feat_gbm|{arm}"
        w = recal_fit(logit(np.asarray(dev[key], float)), dev["_y"].astype(float))
        st = streams_of(D["heldout"], key)
        rc = {s: (y, 1 / (1 + np.exp(-(w[0] + w[1] * logit(p)))), t) for s, (y, p, t) in st.items()}
        R["A2_recalibration"].setdefault("dev_to_heldout", {})[key] = {
            "coef": [round(float(x), 3) for x in w], "heldout_recalibrated": calibration(rc)}

    # A5: alarm precision by agreement among AURA-Ens members (development, double banana)
    d = D["dev_db18"]
    members = [m for m in MEMBERS if m in d.files]
    st = streams_of(d, "aura_ens|post")
    Zm = {m: streams_of(d, m) for m in members}
    sm = {s: (y, smooth(p, t), t) for s, (y, p, t) in st.items()}
    thr = crossfit_thresholds(sm)
    rows = []
    for s, (y, p, t) in sm.items():
        # per-window member agreement: share of members whose own score is above that member's median on this stream
        votes = np.mean([Zm[m][s][1] > np.median(Zm[m][s][1]) for m in members], 0)
        hyp = szcore._merge_split(szcore._runs(p >= thr[s], t, WIN_S), szcore.SZCORE["merge_s"], szcore.SZCORE["max_s"])
        ref = [(a - 30.0, b + 60.0) for a, b in seizure_spans(y, t)]
        for a, b in hyp:
            m = (t >= a) & (t < b)
            rows.append((float(votes[m].mean()) if m.any() else np.nan, any(a < rb and ra < b for ra, rb in ref)))
    rows = np.array([r for r in rows if np.isfinite(r[0])], float)
    cuts = np.quantile(rows[:, 0], [1 / 3, 2 / 3])
    for name, m in (("low", rows[:, 0] <= cuts[0]), ("mid", (rows[:, 0] > cuts[0]) & (rows[:, 0] <= cuts[1])),
                    ("high", rows[:, 0] > cuts[1])):
        R["A5_agreement"][name] = {"n_alarms": int(m.sum()), "precision": round(float(rows[m, 1].mean()), 3),
                                   "agreement_range": [round(float(rows[m, 0].min()), 2), round(float(rows[m, 0].max()), 2)]}
    try:
        R["A5_agreement"]["auc_agreement_vs_true_alarm"] = round(float(roc_auc_score(rows[:, 1], rows[:, 0])), 3)
    except ValueError:
        pass
    R["A5_agreement"]["members"] = members
    print("A5", R["A5_agreement"], flush=True)

    (out_dir / "results.json").write_text(json.dumps(R, indent=1, default=float))
    print("->", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
