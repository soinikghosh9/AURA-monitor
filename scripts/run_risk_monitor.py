#!/usr/bin/env python
"""Monitor module M5 on the development patients: is the next-hour seizure likelihood informative and
calibrated?

Every 10-min bin covered by recordings of the development patients (CHB-MIT, Siena); bins inside a
seizure or its 30-min post-ictal period are not scored. Target: a seizure starts within the next 60 min
(also 6 h). Leave-one-patient-out, three models of increasing knowledge:

  population   the training patients' pooled event rate (a constant)
  clock        time of day only
  aura_risk    history (clustering) + own-rate shrinkage + time of day (bsf.monitor.risk)

Reported per horizon: Brier score and skill against population, AUC, reliability bins, and sensitivity
at 20% time in warning with the analytic chance level (a random predictor with the same time in warning
catches each seizure with probability 0.2; binomial over seizures). Caveat stated in the output: these
recordings were collected around seizures, so absolute rates are inflated relative to routine monitoring.

-> results/risk_monitor_<stamp>/results.json ; results/_monitor_models/risk/
"""
from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


def main() -> int:
    from scipy.stats import binomtest
    from sklearn.metrics import brier_score_loss, roc_auc_score

    from bsf.monitor import ingest
    from bsf.monitor import risk as RK
    from bsf.utils.io import save_json
    from bsf.utils.provenance import new_results_dir
    from run_unified_benchmark import dev_subjects

    pats = [("chbmit", s) for s in sorted(dev_subjects("scalp_chbmit", "dev"))]
    pats += [("siena", s) for s in sorted(dev_subjects("scalp_siena", "dev"))]
    out = new_results_dir("risk_monitor", seed=0, config={"bin_s": RK.BIN_S})
    data = {}
    for cohort, s in pats:
        pdir = REPO / "datasets" / ("chbmit" if cohort == "chbmit" else "siena-scalp-eeg") / s
        recs, spans = ingest.patient(cohort, pdir)
        if not recs:
            continue
        tb = RK.coverage_bins(recs)
        data[s] = {"tb": tb, "onsets": np.array([a for a, _ in spans]), "spans": spans,
                   "t_start": min(r.t0 for r in recs), "hours": float(len(tb) * RK.BIN_S / 3600)}
        print(f"{s}: {data[s]['hours']:.1f} h monitored, {len(spans)} seizures", flush=True)
    R = {"n_patients": len(data), "hours": round(sum(d["hours"] for d in data.values()), 1),
         "n_seizures": int(sum(len(d["onsets"]) for d in data.values())),
         "caveat": "recordings were selected around seizures; absolute rates exceed routine monitoring", "horizons": {}}
    final = {}
    for H in (60, 360):
        rows = {}
        for s, d in data.items():
            obs_h = np.arange(1, len(d["tb"]) + 1) * RK.BIN_S / 3600          # monitored hours so far
            keep = ~RK.ictal_mask(d["tb"], d["spans"])
            y = RK.labels(d["tb"], d["onsets"], H * 60)
            rows[s] = {"tb": d["tb"][keep], "y": y[keep], "obs_h": obs_h[keep]}
        preds = {k: {} for k in ("population", "clock", "aura_risk")}
        for s in rows:
            tr = [o for o in rows if o != s]
            rate = sum(len(data[o]["onsets"]) for o in tr) / sum(data[o]["hours"] for o in tr)
            Ftr = np.concatenate([RK.features(rows[o]["tb"], data[o]["onsets"], data[o]["t_start"], rate,
                                              observed_h=rows[o]["obs_h"]) for o in tr])
            ytr = np.concatenate([rows[o]["y"] for o in tr])
            Fte = RK.features(rows[s]["tb"], data[s]["onsets"], data[s]["t_start"], rate, observed_h=rows[s]["obs_h"])
            preds["population"][s] = np.full(len(Fte), ytr.mean())
            preds["clock"][s] = RK.RiskModel().fit(Ftr[:, 5:7], ytr).predict(Fte[:, 5:7])
            preds["aura_risk"][s] = RK.RiskModel().fit(Ftr, ytr).predict(Fte)
        Y = np.concatenate([rows[s]["y"] for s in rows])
        res = {"n_bins": int(len(Y)), "positive_rate": round(float(Y.mean()), 4)}
        base = brier_score_loss(Y, np.concatenate([preds["population"][s] for s in rows]))
        for k, P in preds.items():
            p = np.concatenate([P[s] for s in rows])
            caught = n_sz = 0
            for s in rows:
                # warnings (Snyder): a bin above threshold raises a warning over the next horizon; threshold
                # set per patient so warnings cover 20% of its monitored time; chance sensitivity = 0.2
                tb = rows[s]["tb"]; ps = P[s]
                grid = np.linspace(0, len(tb) * RK.BIN_S, 1)  # noqa: F841
                best = None
                for thr in np.unique(ps)[::-1]:
                    st = tb[ps >= thr] + RK.BIN_S
                    iv = sorted((a, a + H * 60) for a in st)
                    tot, cur = 0.0, None
                    for a, b in iv:
                        if cur and a <= cur[1]:
                            cur[1] = max(cur[1], b)
                        else:
                            tot += (cur[1] - cur[0]) if cur else 0.0; cur = [a, b]
                    tot += (cur[1] - cur[0]) if cur else 0.0
                    if tot > 0.2 * len(tb) * RK.BIN_S:
                        break
                    best = st
                for on in data[s]["onsets"]:
                    if not ((tb + RK.BIN_S < on) & (tb + RK.BIN_S + H * 60 >= on)).any():
                        continue                      # onset not preceded by a scored bin within the horizon
                    n_sz += 1
                    if k != "population" and best is not None and np.any((best < on) & (best + H * 60 >= on)):
                        caught += 1
            per = [roc_auc_score(rows[s]["y"], P[s]) for s in rows if 0 < rows[s]["y"].sum() < len(rows[s]["y"]) and np.ptp(P[s]) > 0]
            b = brier_score_loss(Y, p)
            res[k] = {"brier": round(float(b), 5), "brier_skill": round(float(1 - b / base), 4),
                      "auc": round(float(roc_auc_score(Y, p)), 4) if k != "population" else None,
                      "median_patient_auc": round(float(np.median(per)), 4) if per else None,
                      "reliability": RK.reliability(p, Y), "sens_at_20pct_tiw": round(caught / n_sz, 4) if n_sz else None,
                      "n_seizures_scored": n_sz,
                      "p_vs_chance": float(binomtest(caught, n_sz, 0.2, alternative="greater").pvalue) if n_sz and k != "population" else None}
            print(f"[H={H} min] {k}: Brier skill {res[k]['brier_skill']:+.3f}, AUC {res[k]['auc']}, patient-median {res[k]['median_patient_auc']}, "
                  f"sens@20%TiW {res[k]['sens_at_20pct_tiw']} (p {res[k]['p_vs_chance']})", flush=True)
        R["horizons"][str(H)] = res
        rate = sum(len(d["onsets"]) for d in data.values()) / sum(d["hours"] for d in data.values())
        Fall = np.concatenate([RK.features(rows[s]["tb"], data[s]["onsets"], data[s]["t_start"], rate,
                                           observed_h=rows[s]["obs_h"]) for s in rows])
        final[H] = {"model": RK.RiskModel().fit(Fall, np.concatenate([rows[s]["y"] for s in rows])), "population_rate_per_h": rate}
    md = REPO / "results" / "_monitor_models" / "risk"; md.mkdir(parents=True, exist_ok=True)
    with open(md / "risk_models.pkl", "wb") as fh:
        pickle.dump(final, fh)
    save_json(R, out / "results.json")
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
