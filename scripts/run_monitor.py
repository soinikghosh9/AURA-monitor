#!/usr/bin/env python
"""AURA Monitor, end to end: one patient's recordings in, one clinician report out.

  python scripts/run_monitor.py --cohort chbmit --patient chb01 [--hours 24] [--budget 6]

Steps: common montage and signal quality per 8-s window; AURA features and per-channel features with the
patient's refreshed reference; seizure probability from the banked AURA-Ens fold that held this patient
out; alarms at the false-alarm budget (threshold from the other development patients); a replayed
clinician review (alarms adjudicated against the annotations) driving the personalization loop; IED
detection, type and channel maps on 4-s epochs; seizure fingerprints, types and similarity; next-hour
risk from the seizures the monitor has confirmed; an evidence card per alarm; and the report.

-> reports/<cohort>_<patient>/{report.html, report.json}
"""
from __future__ import annotations

import argparse
import glob
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


def _latest(pattern):
    d = sorted(glob.glob(str(REPO / "results" / pattern)))
    return json.loads(Path(d[-1], "results.json").read_text()) if d else None


def window_features(raw: np.ndarray, fs: float):
    """One raw 8-s window (18, 2000) in volts -> standardised window, AURA features, channel features,
    quality, and its two 4-s IED epochs' features."""
    from bsf.data.features import rich_plus
    from bsf.monitor import ied as I
    from bsf.monitor import signals as S
    q = S.quality(raw, fs)
    z = S.standardise(raw)
    out = {"W": z.astype(np.float16), "F": np.nan_to_num(rich_plus(z.T, fs)).astype(np.float32),
           "C": S.channel_features(z), "q": q["score"],
           "flag": [S.DB18_NAMES[i] for i in np.flatnonzero(q["flat"] | q["clipped"] | q["artefact"] | q["line"])]}
    n = raw.shape[1] // 2
    ep = []
    for k in (0, 1):
        r = raw[:, k * n:(k + 1) * n] * 1e6
        ze = S.standardise(r)
        ep.append((S.channel_features(ze), I.amplitude_features(r[None])[0], np.nan_to_num(rich_plus(ze.T, fs))))
    out["ied"] = ep
    return out


def main() -> int:
    from joblib import Parallel, delayed

    from bsf.eval import szcore
    from bsf.models import pbn as PBN
    from bsf.monitor import explain, ingest, patterns, report
    from bsf.monitor import ied as I
    from bsf.monitor import risk as RK
    from bsf.monitor import signals as S
    from bsf.monitor.detector import SeizureDetector
    from bsf.monitor.personalize import Personalizer
    from bsf.utils.compute import resolve_device
    from run_stream_replication import smooth

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", required=True, choices=["chbmit", "siena"])
    ap.add_argument("--patient", required=True)
    ap.add_argument("--hours", type=float, default=0.0, help="0 = every recording")
    ap.add_argument("--budget", type=float, default=6.0, help="false alarms per 24 h")
    ap.add_argument("--ens", default="results/aura_ens_db18")
    ap.add_argument("--stack", default="post")
    ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args()
    dev = resolve_device("auto")
    key = f"{a.cohort}_{a.patient}"
    outdir = REPO / "reports" / key; outdir.mkdir(parents=True, exist_ok=True)
    pdir = REPO / "datasets" / ("chbmit" if a.cohort == "chbmit" else "siena-scalp-eeg") / a.patient
    recs, spans = ingest.patient(a.cohort, pdir)
    t_start = recs[0].t0
    t_end = t_start + a.hours * 3600 if a.hours else max(r.t0 + r.dur for r in recs)
    recs = [r for r in recs if r.t0 < t_end]
    spans = [(x, y) for x, y in spans if x < t_end]
    print(f"[{key}] {len(recs)} recordings, {len(spans)} annotated seizures", flush=True)

    # ---- 1. windows and features
    rows, T = [], []
    for r in recs:
        got = ingest.read(r, None, min(r.t0 + r.dur, t_end))
        if got is None:
            continue
        Wr, tr = ingest.windows(*got)
        rows += Parallel(n_jobs=a.n_jobs, batch_size=32)(delayed(window_features)(w, S.FS) for w in Wr)
        T.append(tr)
        print(f"  {r.path.name}: {len(tr)} windows", flush=True)
    t = np.concatenate(T)
    W = np.stack([x["W"] for x in rows]).astype(np.float32)
    F = np.stack([x["F"] for x in rows]); Cf = np.stack([x["C"] for x in rows]); Q = np.array([x["q"] for x in rows])
    y = np.array([any(a_ <= ti + 4 < b_ for a_, b_ in spans) for ti in t], int)
    span_s, gap_s = 1800.0, 300.0

    # ---- 2. seizure probability, threshold, alarms
    det = SeizureDetector(REPO / a.ens, a.stack, dev)
    sc = det.score(W, t, F, key)
    p, Fz = sc["ensemble"], sc["features_z"]
    ps = smooth(p, t, 120.0)
    pr = np.load(REPO / a.ens / "preds.npz", allow_pickle=True)
    G = pr["_groups"].astype(str)
    kk = f"aura_ens|{a.stack}"
    others = [(pr["_y"][G == s], smooth(pr[kk][G == s], pr["_t"][G == s]), pr["_t"][G == s]) for s in np.unique(G) if s != key]
    thr = szcore.threshold_for_budget(others, a.budget)
    Zc = PBN.rolling_reference(Cf.reshape(len(Cf), -1), t, 0, span_s, gap_s, 10).reshape(Cf.shape)

    # ---- 3. replayed review and personalization
    pers = Personalizer(thr, a.budget)
    rep = pers.replay(y, p, Fz, t)
    first_conf = next((e["t_on"] for e in rep["events"] if e["confirmed"]), None)
    pers_summary = None
    if first_conf is not None:
        late = t > first_conf + 600
        base_c = szcore.events(y[late], ps[late], t[late], thr)
        online = smooth(rep["scores"], t, 120.0) - rep["thr_t"]
        ad_c = szcore.events(y[late], online[late], t[late], 0.0)
        hrs = late.sum() * 8 / 3600
        pers_summary = {"n_confirmed": sum(e["confirmed"] for e in rep["events"]), "n_rejected": sum(not e["confirmed"] for e in rep["events"]),
                        "threshold_start": thr, "threshold_end": rep["final_threshold"], "n_after": base_c["n_ref"],
                        "detected_before": base_c["tp_ref"], "detected_after": ad_c["tp_ref"],
                        "fa_before": 24 * base_c["fp"] / max(hrs, 1e-6), "fa_after": 24 * ad_c["fp"] / max(hrs, 1e-6)}

    # ---- 4. IEDs on 4-s epochs, patient-referenced by the preceding 35 to 5 min
    with open(REPO / "results/_monitor_models/ied/ied_models.pkl", "rb") as fh:
        IM = pickle.load(fh)
    te = np.repeat(t, 2) + np.tile([0.0, 4.0], len(t))
    ech = np.stack([e[0] for x in rows for e in x["ied"]]); eam = np.stack([e[1] for x in rows for e in x["ied"]])
    egl = np.stack([e[2] for x in rows for e in x["ied"]])
    Zch_e = PBN.rolling_reference(ech.reshape(len(ech), -1), te, 0, span_s, gap_s, 20).reshape(ech.shape)
    Zam_e = PBN.rolling_reference(eam.reshape(len(eam), -1), te, 0, span_s, gap_s, 20).reshape(eam.shape)
    Zgl_e = PBN.rolling_reference(egl, te, 0, span_s, gap_s, 20)
    Dz, Dr = I.design(Zch_e, Zam_e, Zgl_e), I.design(ech, eam, egl)
    D = {"patz": Dz, "raw": Dr, "both": np.concatenate([Dr, Dz], 1)}[IM["input"]]
    pi = IM["detect"].predict_proba(D)[:, 1]
    ied = (pi >= IM["thr_1fpm"]) & (np.repeat(Q, 2) >= 0.8) & (np.repeat(y, 2) == 0)
    idx = np.flatnonzero(ied)
    Dg = I.design(I.rolling_gated_z(ech, te, pi), I.rolling_gated_z(eam, te, pi), I.rolling_gated_z(egl, te, pi))
    gen = IM["type_gen_vs_focal"].predict_proba(Dg[idx])[:, 1] >= 0.5 if len(idx) else np.zeros(0, bool)
    cm = I.channel_map(Zch_e[idx]) if len(idx) else np.zeros((0, 18))
    # lobe classification was at chance in validation: report only where the evidence is strongest (descriptive)
    region = [max(I.region_scores(c).items(), key=lambda kv: kv[1])[0] for c in cm]
    sides = [I.lateralize(c)["side"] for c, g_ in zip(cm, gen) if not g_]
    bins = np.arange(np.floor(t[0] / 600) * 600, t[-1] + 8, 600)
    nb = np.histogram(te[idx], bins)[0]
    covered = np.histogram(te, bins)[0] > 0
    per_h = nb[covered] * 6

    # ---- 5. seizure fingerprints, types, similarity
    seizures = []
    for k, (on, off) in enumerate(spans):
        clear = np.array([all(ti + 8 <= s0 - 300 or ti >= s1 + 300 for s0, s1 in spans) for ti in t])
        rm = (t >= on - span_s - gap_s) & (t < on - gap_s) & clear
        pm = (t >= on - 32) & (t < on + 64)
        if rm.sum() < 10 or pm.sum() < 6:
            continue
        raw_ref = np.stack([W[i] for i in np.flatnonzero(rm)])
        fp = patterns.fingerprint(raw_ref, W[pm], t[pm] - on, float(off - on))
        fp.pop("evidence")
        sw = (t >= on - 30) & (t < off + 30)
        hit = any(e["t_on"] < off + 60 and e["t_off"] > on - 30 for e in rep["events"])
        snip = None
        rec = next((r for r in recs if r.t0 <= on < r.t0 + r.dur), None)
        if rec is not None:
            g2 = ingest.read(rec, on - 10, on + 20)
            if g2 is not None:
                x = g2[0]; k_ = max(1, x.shape[1] // 600)
                snip = {"x": x[:, ::k_].tolist(), "t_rel": (g2[1] - on + np.arange(0, x.shape[1], k_) / S.FS).tolist()}
        seizures.append({"source": "annotated", "t_on": on, "duration_s": float(off - on), "detected": bool(hit),
                         "fingerprint": fp, "features": explain.top_features(Fz[sw]) if sw.any() else [], "eeg": snip})
    labels = patterns.cluster([s["fingerprint"] for s in seizures]) if seizures else []
    for i, s in enumerate(seizures):
        s["type"] = int(labels[i]) if labels else 0
        s["similar"] = sorted([{"when": explain_clock(seizures[j]["t_on"]), "similarity": patterns.similarity(s["fingerprint"], seizures[j]["fingerprint"])}
                               for j in range(i)], key=lambda d: -d["similarity"])

    # ---- 6. risk from the seizures the monitor has confirmed
    with open(REPO / "results/_monitor_models/risk/risk_models.pkl", "rb") as fh:
        RM = pickle.load(fh)[60]
    tb = RK.coverage_bins(recs)
    tb = tb[tb < t_end]
    conf_on = np.array([e["t_on"] for e in rep["events"] if e["confirmed"]])
    Fr = RK.features(tb, conf_on, t_start, RM["population_rate_per_h"], observed_h=np.arange(1, len(tb) + 1) * RK.BIN_S / 3600)
    risk = RM["model"].predict(Fr)

    # ---- 7. alarm cards
    cards = []
    onset_maps = [(s["t_on"], np.clip(np.asarray(s["fingerprint"]["onset_map"]), 0, None)) for s in seizures]
    for e in rep["events"]:
        m = (t >= e["t_on"]) & (t < e["t_off"])
        if not m.any():
            continue
        mem = {k.split("|f")[0]: float(np.mean(v[m])) for k, v in sc["members"].items()}
        cmap = patterns.evidence(Zc[m]).mean(0)
        sim = sorted([{"when": explain_clock(on), "similarity": float(cmap @ om / (np.linalg.norm(cmap) * np.linalg.norm(om) + 1e-9))}
                      for on, om in onset_maps if on < e["t_on"] - 60], key=lambda d: -d["similarity"])
        recent = ((te[idx] >= e["t_on"] - 1800) & (te[idx] < e["t_on"])).sum()
        c = explain.alarm_card(e["t_on"], e["t_off"], float(ps[m].max()), mem, Fz[m], Zc[m], float(Q[m].mean()),
                               sorted({f for i in np.flatnonzero(m) for f in rows[i]["flag"]}), float(recent),
                               float(np.median(nb[covered]) * 3) if covered.any() else 0.0, sim, thr)
        c["confirmed"] = e["confirmed"]
        cards.append(c)

    # ---- 8. validation and report
    ens = json.loads((REPO / a.ens / "results.json").read_text())
    em = ens["models"]["aura_ens"][a.stack]["szcore"][str(int(a.budget))] if str(int(a.budget)) in ens["models"]["aura_ens"][a.stack]["szcore"] else None
    iedv = _latest("ied_monitor_*"); patv = _latest("seizure_patterns_*"); rkv = _latest("risk_monitor_*")
    best_ied = IM["arm"]
    rows_v = [
        ("Seizure detection (AURA-Ens)", f"25 continuous development streams: pooled AUC {ens['models']['aura_ens'][a.stack]['auc']:.3f}; "
         + (f"at {a.budget:g} false alarms/24 h budget: sensitivity {em['sensitivity']:.2f}, {em['fa_per_24h']:.1f} false alarms/24 h, "
            f"median latency {em['median_latency_s']} s" if em else "")),
        ("IED detection", f"VEPISET, {iedv['n_patients']} patients, 5-fold: AUC {iedv['detection'][best_ied]['auc']:.3f}, "
         f"sensitivity {iedv['detection'][best_ied]['sens_at_1fp_per_min']:.2f} at 1 false detection/min; generalized vs focal AUC "
         f"{iedv['typing_gated']['generalized_vs_focal_auc']:.2f}; lobe not classifiable (balanced accuracy {iedv['typing_gated']['lobe_balanced_acc']:.2f}, "
         f"chance 0.25), so regions are descriptive") if iedv else ("IED detection", "pending"),
        ("Seizure fingerprints", f"{patv['n_seizures']} seizures, {patv['n_patients']} patients: similarity within patient "
         f"{patv['stereotypy']['within']:.2f} vs between {patv['stereotypy']['between']:.2f} (p {patv['stereotypy']['p_perm']:.4f}); "
         f"automated lateralization not reliable ({patv['lateralization']['correct']}/{patv['lateralization']['n_lateralized_calls']} confident calls, "
         f"abstains on {100 * patv['lateralization']['abstain_rate']:.0f}%)") if patv else ("Seizure fingerprints", "pending"),
        ("Risk, next hour", f"{rkv['n_patients']} patients, leave-one-out: Brier skill {rkv['horizons']['60']['aura_risk']['brier_skill']:+.3f}; "
         f"sensitivity {rkv['horizons']['60']['aura_risk']['sens_at_20pct_tiw']:.2f} at 20% time in warning (chance 0.20)") if rkv else ("Risk", "pending"),
    ]
    alarms_eval = szcore.events(y, ps, t, thr)
    R = {"patient": key, "cohort_label": "CHB-MIT (public, de-identified)" if a.cohort == "chbmit" else "Siena Scalp EEG (public, de-identified)",
         "t_start": float(t[0]), "t_end": float(t[-1] + 8), "hours": float(len(t) * 8 / 3600), "version": "0.1 (2026-10-05)",
         "data_note": "PhysioNet open-access recordings; models trained without this patient.",
         "detection": {"threshold": float(thr), "budget_per_24h": a.budget},
         "summary": {"alarms": len(rep["events"]), "confirmed": sum(e["confirmed"] for e in rep["events"]), "annotated": len(spans),
                     "detected": sum(s["detected"] for s in seizures), "fa_per_24h": round(24 * alarms_eval["fp"] / max(len(t) * 8 / 3600, 1e-6), 1),
                     "ied_per_h": int(np.median(per_h)) if len(per_h) else 0,
                     "ied_side": max(("left", "right", "bilateral"), key=sides.count) if sides else "n/a",
                     "n_types": len(set(labels)) if labels else 0, "risk_now": f"{100 * risk[-1]:.0f}%" if len(risk) else "n/a"},
         "annotated_seizures": [list(s) for s in spans],
         "timeline": {"t": t.tolist(), "prob": ps.round(4).tolist(), "quality": Q.round(2).tolist(),
                      "ied_bins_t": bins[:-1].tolist(), "ied_bins_n": nb.tolist(), "risk_t": (tb + RK.BIN_S).tolist(), "risk_p": risk.round(4).tolist()},
         "alarms": cards, "seizures": seizures, "personalization": pers_summary,
         "ied": {"n": int(len(idx)), "per_h_median": int(np.median(per_h)) if len(per_h) else 0, "per_h_min": int(per_h.min()) if len(per_h) else 0,
                 "per_h_max": int(per_h.max()) if len(per_h) else 0, "generalized": int(gen.sum()), "focal": int((~gen).sum()),
                 "lobes": {r: int(sum(rg == r and not g_ for rg, g_ in zip(region, gen))) for r in ("frontal", "temporal", "centro_parietal", "occipital")},
                 "side": {s: sides.count(s) for s in ("left", "right", "bilateral")},
                 "map": cm.mean(0).round(3).tolist() if len(cm) else [0.0] * 18},
         "validation": {"rows": rows_v, "risk": {"brier_skill": rkv["horizons"]["60"]["aura_risk"]["brier_skill"] if rkv else 0.0,
                                                 "sens": rkv["horizons"]["60"]["aura_risk"]["sens_at_20pct_tiw"] if rkv else 0.0}}}
    (outdir / "report.json").write_text(json.dumps(R, default=float), encoding="utf-8")
    (outdir / "report.html").write_text(report.render(R), encoding="utf-8")
    print(f"-> {outdir / 'report.html'} | alarms {R['summary']['alarms']} (confirmed {R['summary']['confirmed']}), "
          f"seizures detected {R['summary']['detected']}/{len(spans)}, IEDs {len(idx)}", flush=True)
    return 0


def explain_clock(t: float) -> str:
    d, s = divmod(int(t), 86400)
    return f"day {d + 1} {s // 3600:02d}:{(s % 3600) // 60:02d}"


if __name__ == "__main__":
    raise SystemExit(main())
