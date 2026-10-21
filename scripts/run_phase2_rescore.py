#!/usr/bin/env python
"""Post-hoc re-scoring of the locked continuous-EEG predictions (audit 2026-10-21, Phase 2). No model is retrained
or re-run; every number here comes from preds.npz / results.json files already written by the locked passes.

  E2  SzCORE event scoring at fixed false-alarm budgets (2, 6, 24 per 24 h) for the development, held-out and
      SeizeIT2 streams, so that every continuous analysis reports the same clinical metric. The locked 2-min causal
      average is kept; each recording's threshold is set on the OTHER recordings of the same set (cross-fitted), so
      no recording's own labels set its threshold. Post hoc for the held-out and SeizeIT2 passes.
  E7  Precision of the primary endpoints: observed mean and SD of the per-recording AUC change, the smallest mean
      change detectable with 80% power by the two-sided sign-flip test at alpha 0.05, and the power at a 0.05 margin.
  E8  Calibration of the window probabilities (calibration-in-the-large, slope, Brier skill, ECE).
  E10 BIOT's held-out embedding-normalization gain restricted to recordings outside BIOT's pretraining corpus
      (CHB-MIT chb01-chb20), with SCCNet as an unpretrained comparison.

Peri-ictal windows (within 120 s of a seizure) were dropped before the locked passes saved their predictions, so
scoring without that exclusion needs new inference and is not done here.

  python scripts/run_phase2_rescore.py      -> results/phase2_rescore_<stamp>/results.json
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
from bsf.eval import szcore  # noqa: E402

RES = ROOT / "results"
SETS = {
    "dev": RES / "stream_replication_20261004T124204Z",
    "heldout": RES / "stream_test_20261004T200300Z",
    "seizeit2": RES / "seizeit2_external_20261004T171522Z",
    "dev_db18": RES / "aura_ens_db18",
}
BUDGETS = (2.0, 6.0, 24.0)
WIN_S = 8.0
SMOOTH_S = 120.0
BIOT_SEEN = {f"chbmit_chb{i:02d}" for i in range(1, 21)}
RNG = np.random.default_rng(20261021)


def smooth(p, t, span_s=SMOOTH_S):
    """Causal moving average over windows starting in (t - span, t] (the locked rule)."""
    c = np.r_[0.0, np.cumsum(p)]
    lo = np.searchsorted(t, t - span_s, side="right")
    hi = np.arange(1, len(p) + 1)
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)


def streams_of(d, key):
    y, g, t = d["_y"], d["_groups"], d["_t"]
    out = {}
    for s in np.unique(g):
        m = np.where(g == s)[0]
        o = m[np.argsort(t[m])]
        out[str(s)] = (y[o].astype(int), np.asarray(d[key], float)[o], t[o].astype(float))
    return out


def szcore_crossfit(st):
    """Per-recording threshold from the other recordings at each budget; pooled counts and a recording bootstrap."""
    names = sorted(st)
    sm = {s: (y, smooth(p, t), t) for s, (y, p, t) in st.items()}
    out = {}
    for b in BUDGETS:
        counts = {}
        for s in names:
            thr = szcore.threshold_for_budget([sm[o] for o in names if o != s], b, WIN_S)
            counts[s] = szcore.events(*sm[s], thr, WIN_S)
        summ = szcore.summarize(list(counts.values()))
        boots = []
        for _ in range(2000):
            pick = RNG.choice(names, len(names), replace=True)
            c = [counts[s] for s in pick]
            nref = sum(x["n_ref"] for x in c)
            hrs = sum(x["hours"] for x in c)
            boots.append((sum(x["tp_ref"] for x in c) / nref if nref else np.nan, 24 * sum(x["fp"] for x in c) / hrs))
        bt = np.array(boots)
        summ["sensitivity_ci95"] = [round(float(v), 3) for v in np.nanpercentile(bt[:, 0], [2.5, 97.5])]
        summ["fa_per_24h_ci95"] = [round(float(v), 2) for v in np.percentile(bt[:, 1], [2.5, 97.5])]
        out[str(int(b))] = summ
    return out


def per_rec_auc(st):
    return {s: float(roc_auc_score(y, p)) for s, (y, p, _t) in st.items() if 0 < y.sum() < len(y)}


def signflip_p(d, n_mc=None):
    d = np.asarray(d, float)
    n = len(d)
    obs = abs(d.mean())
    if n <= 16:
        signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    else:
        signs = RNG.choice([-1, 1], size=(n_mc or 20000, n))
    return float(np.mean(np.abs((signs * d).mean(1)) >= obs - 1e-12))


def power_signflip(sd, n, mu, n_sim=400, n_mc=1000, alpha=0.05):
    rej = 0
    signs = RNG.choice([-1, 1], size=(n_mc, n))
    for _ in range(n_sim):
        d = RNG.normal(mu, sd, n)
        rej += np.mean(np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-12) <= alpha
    return rej / n_sim


def mde(sd, n):
    lo, hi = 0.0, 0.5
    for _ in range(12):
        mid = (lo + hi) / 2
        if power_signflip(sd, n, mid) >= 0.8:
            hi = mid
        else:
            lo = mid
    return hi


def calibration(st):
    y = np.concatenate([v[0] for v in st.values()])
    p = np.clip(np.concatenate([v[1] for v in st.values()]), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))

    def newton(X, off):
        w = np.zeros(X.shape[1])
        for _ in range(50):
            eta = X @ w + off
            mu = 1 / (1 + np.exp(-eta))
            H = X.T @ (X * (mu * (1 - mu))[:, None]) + 1e-9 * np.eye(X.shape[1])
            w = w + np.linalg.solve(H, X.T @ (y - mu))
        return w
    citl = float(newton(np.ones((len(y), 1)), lp)[0])
    slope = float(newton(np.c_[np.ones(len(y)), lp], 0.0)[1])
    prev = y.mean()
    brier = float(np.mean((p - y) ** 2))
    edges = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    ece = float(sum(abs(p[idx == k].mean() - y[idx == k].mean()) * (idx == k).mean() for k in np.unique(idx)))
    return {"prevalence": round(float(prev), 5), "mean_predicted": round(float(p.mean()), 4),
            "calibration_in_the_large": round(citl, 3), "calibration_slope": round(slope, 3),
            "brier": round(brier, 5), "brier_skill": round(1 - brier / (prev * (1 - prev)), 3), "ece": round(ece, 4)}


def main() -> int:
    out_dir = RES / f"phase2_rescore_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir.mkdir(parents=True)
    R: dict = {"note": "post hoc re-scoring of locked predictions; peri-ictal windows (120 s) absent in all sets",
               "smoothing_s": SMOOTH_S, "budgets_per_24h": list(BUDGETS), "szcore": {}, "calibration": {},
               "precision": {}, "biot_uncontaminated": {}}
    D = {k: np.load(v / "preds.npz", allow_pickle=True) for k, v in SETS.items() if (v / "preds.npz").exists()}

    # E2 + E8: models scored in each set
    want = {
        "dev": ["feat_gbm|none", "feat_gbm|refreshed"],
        "heldout": ["feat_gbm|none", "feat_gbm|refreshed", "BIOT|none", "BIOT|probe_gbm_refreshed",
                    "SCCNet|none", "SCCNet|probe_gbm_refreshed", "auranet_hybrid|refreshed",
                    "ensemble|refreshed", "stacked|refreshed"],
        "seizeit2": ["feat_gbm|none", "feat_gbm|refreshed", "BIOT|none", "deep_cnn_raw|none", "ensemble|none"],
        "dev_db18": ["feat_gbm|refreshed", "BIOT|probe_gbm_refreshed", "ens_mean|post"],
    }
    for k, keys in want.items():
        if k not in D:
            continue
        R["szcore"][k], R["calibration"][k] = {}, {}
        for key in keys:
            if key not in D[k].files:
                continue
            st = streams_of(D[k], key)
            R["szcore"][k][key] = szcore_crossfit(st)
            R["calibration"][k][key] = calibration(st)
            print(k, key, {b: (v["detected"], v["n_ref"], v["fa_per_24h"]) for b, v in R["szcore"][k][key].items()},
                  flush=True)

    # E7: precision of each primary endpoint (boosted AURA, rolling reference minus none, per recording)
    rep = json.loads((SETS["dev"] / "results.json").read_text())
    orig = {s for s, v in rep["streams"].items() if v.get("original")}
    pa = rep["models"]["feat_gbm"]["per_stream_auc"]
    deltas = {"dev_replication": [v["refreshed"] - v["none"] for s, v in pa.items() if s not in orig]}
    for k in ("heldout", "seizeit2"):
        a0 = per_rec_auc(streams_of(D[k], "feat_gbm|none"))
        a1 = per_rec_auc(streams_of(D[k], "feat_gbm|refreshed"))
        deltas[k] = [a1[s] - a0[s] for s in sorted(a0) if s in a1]
    for k, d in deltas.items():
        d = np.array(d)
        sd = float(d.std(ddof=1))
        R["precision"][k] = {"n": len(d), "mean": round(float(d.mean()), 4), "sd": round(sd, 4),
                             "p_signflip": round(signflip_p(d), 4), "mde_80": round(mde(sd, len(d)), 3),
                             "power_at_0.05": round(power_signflip(sd, len(d), 0.05), 2)}
        print("E7", k, R["precision"][k], flush=True)

    # E10: BIOT and SCCNet held-out gains, all recordings and outside BIOT's pretraining corpus
    for net in ("BIOT", "SCCNet"):
        a0 = per_rec_auc(streams_of(D["heldout"], f"{net}|none"))
        a1 = per_rec_auc(streams_of(D["heldout"], f"{net}|probe_gbm_refreshed"))
        for sub, keep in (("all", lambda s: True), ("outside_biot_pretraining", lambda s: s not in BIOT_SEEN)):
            names = [s for s in sorted(a0) if keep(s) and s in a1]
            d = np.array([a1[s] - a0[s] for s in names])
            R["biot_uncontaminated"][f"{net}|{sub}"] = {
                "recordings": names, "n": len(d), "mean": round(float(d.mean()), 4),
                "n_improved": int((d > 0).sum()), "p_signflip": round(signflip_p(d), 4),
                "min_attainable_p": round(2 / 2 ** len(d), 4)}
            print("E10", net, sub, R["biot_uncontaminated"][f"{net}|{sub}"], flush=True)

    (out_dir / "results.json").write_text(json.dumps(R, indent=1))
    print("->", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
