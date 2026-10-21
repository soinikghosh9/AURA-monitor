#!/usr/bin/env python
"""One-shot personalization from a patient's first recorded seizure (exploratory, development patients).

Seizures of the same patient depart from the preceding 30 min in similar directions, those of different
patients do not (run_stream_onset_profile.py). This asks whether the first seizure of a monitoring
session can personalize AURA for the rest of it. For every replication stream with at least two
seizures, the windows of the FIRST seizure define the patient's signature in refreshed-baseline feature
space, and every window that starts after that seizure is scored by

  population   boosted AURA with the refreshed baseline (the replication's predictions, unchanged)
  template     similarity of the window's refreshed features to the first-seizure signature
               (cosine, mapped to [0, 1])
  patient      an L2 logistic fitted on the first seizure's windows (positives) and the non-ictal
               windows before it (negatives), refreshed features
  combined     mean of population and patient

Nothing after the first seizure is used for fitting; the comparison is paired per stream on identical
windows (per-stream AUC, sign-flip test over streams). Exploratory: devised after the onset profiles.

-> results/stream_personalize_<stamp>/results.json
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
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    from bsf.utils.io import save_json
    from bsf.utils.provenance import new_results_dir
    from make_paper_numbers import latest
    from run_final_stats import signflip
    from run_stream_nets import load_streams
    from run_stream_replication import RULE, _rolling, events_at, smooth
    from run_streaming_monitor import _stream_features

    rep = latest("stream_replication_*")
    R = json.loads((rep / "results.json").read_text())
    with np.load(rep / "preds.npz", allow_pickle=True) as z:
        G = z["_groups"].astype(str)
        Ppop = z["feat_gbm|refreshed"].astype(float)
    streams = load_streams(sorted(R["streams"]), RULE["max_hours"])
    out = new_results_dir("stream_personalize", seed=0, config={"replication": rep.name})
    arms = ("population", "template", "patient", "combined")
    per, pooled = {}, {a: [] for a in arms}
    Yp, Tp, Gp = [], [], []
    for s, st in streams.items():
        X = np.load(st["file"])["X"]
        F = _stream_features({"subject": s, "X": X}, "detect", RULE["fs"], RULE["channels"], 2000)[st["order"]]
        del X
        Z = _rolling(F, st["t"], st["cut"], RULE["rolling_min"] * 60, RULE["gap_min"] * 60, RULE["min_ref_windows"])
        y, t = st["y"][st["cut"]:], st["t"][st["cut"]:]
        pop = Ppop[G == s]
        assert len(pop) == len(y), s
        idx = np.flatnonzero(y)
        runs = np.split(idx, np.where((np.diff(idx) > 1) | (np.diff(t[idx]) > 1.5 * RULE["window_s"]))[0] + 1)
        if len(runs) < 2:
            continue
        first = runs[0]
        after = np.arange(first[-1] + 1, len(y))
        if y[after].sum() == 0 or y[after].sum() == len(after):
            continue
        sig = Z[first].mean(0)
        sc = StandardScaler().fit(Z[:first[-1] + 1])
        tmpl = sc.transform(Z) @ (sc.transform(sig[None])[0])
        tmpl /= (np.linalg.norm(sc.transform(Z), axis=1) * np.linalg.norm(sc.transform(sig[None])[0]) + 1e-9)
        tmpl = (tmpl + 1) / 2
        neg = np.arange(0, first[0])
        neg = neg[y[neg] == 0]
        Xfit = np.concatenate([Z[first], Z[neg]]); yfit = np.r_[np.ones(len(first)), np.zeros(len(neg))]
        lr = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000).fit(sc.transform(Xfit), yfit)
        pat = lr.predict_proba(sc.transform(Z))[:, 1]
        S = {"population": pop, "template": tmpl, "patient": pat, "combined": 0.5 * pop + 0.5 * pat}
        ya = y[after]
        per[s] = {a: round(float(roc_auc_score(ya, S[a][after])), 4) for a in arms}
        per[s].update({"n_seizures_after": len(runs) - 1, "first_seizure_windows": int(len(first)),
                       "hours_after": round(len(after) * RULE["window_s"] / 3600, 2)})
        for a in arms:
            pooled[a].append(S[a][after])
        Yp.append(ya); Tp.append(t[after]); Gp.append(np.array([s] * len(after)))
        print(f"  {s}: {len(runs) - 1} later seizures, " + ", ".join(f"{a} {per[s][a]:.3f}" for a in arms), flush=True)
    subj = list(per)
    res = {"n_streams": len(subj), "n_seizures_after": int(sum(per[s]["n_seizures_after"] for s in subj)),
           "per_stream": per, "arms": {}}
    Y = np.concatenate(Yp)
    for a in arms:
        p = np.concatenate(pooled[a])
        d = [per[s][a] - per[s]["population"] for s in subj]
        # events at a threshold transferred from the other streams (10% time in warning), 2-min average
        cau = nev = fa = 0; hrs = 0.0
        for i, s in enumerate(subj):
            oth = np.concatenate([smooth(pooled[a][j], Tp[j]) for j in range(len(subj)) if j != i])
            e = events_at(Yp[i], smooth(pooled[a][i], Tp[i]), Tp[i], float(np.quantile(oth, 0.9)), RULE["window_s"])
            cau += e["caught"]; nev += e["n_events"]; fa += e["false_alarms"]; hrs += len(Yp[i]) * RULE["window_s"] / 3600
        res["arms"][a] = {"pooled_auc": round(float(roc_auc_score(Y, p)), 4),
                          "median_stream_auc": round(float(np.median([per[s][a] for s in subj])), 4),
                          "delta_vs_population": round(float(np.mean(d)), 4),
                          "n_improved": int(sum(x > 0 for x in d)), "p_signflip": signflip(d) if a != "population" else None,
                          "caught": cau, "n_events": nev, "fa_per_24h": round(24 * fa / hrs, 2)}
        print(f"[{a}] median stream AUC {res['arms'][a]['median_stream_auc']:.3f} delta {res['arms'][a]['delta_vs_population']:+.3f}"
              f" ({res['arms'][a]['n_improved']}/{len(subj)}) p={res['arms'][a]['p_signflip']}  events {cau}/{nev}"
              f" FA/24h {res['arms'][a]['fa_per_24h']}", flush=True)
    save_json(res, out / "results.json")
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
