#!/usr/bin/env python
"""Post-hoc recalibration on held-out SUBJECTS, scored only on subjects the map never saw.

Replaces run_calibration.py for the manuscript. That script fitted a Platt map on 25% of subjects and
then scored Brier skill on ALL subjects, including the ones the map was fitted to, and it dropped any
cell whose fitted slope came out non-positive without reporting it (six of 100 cells vanished that
way, which is why its denominator was 94).

Here, for every model-task cell and each of ``--repeats`` random subject splits:
  1. a calibration set of subjects (``--cal-frac``) is drawn;
  2. ONE map is fitted on their pooled out-of-fold predictions -- a single monotone map, so the
     ranking of the remaining subjects' predictions, and therefore their AUC, is exactly unchanged;
  3. Brier skill (against the evaluation subjects' own base rate) and ECE are computed on the
     evaluation subjects only, before and after.
Three maps: temperature (slope only), Platt (slope + intercept), and intercept-only. A map whose
fitted slope is <= 0 would invert the ranking; such cells are counted as ANTI-CALIBRATED, not skipped.

Reported per cell: median over repeats of BSS/ECE before and after, and the fraction of repeats in
which the calibrated probability beats the base rate (BSS > 0).

-> results/calibration_xfit_<stamp>/results.json + cells.csv
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_map(z, y, kind):
    """Return (a, b) for q = sigmoid(a*z + b), or None if the map would invert the ranking."""
    from scipy.optimize import minimize

    def nll(th):
        a, b = (th[0], th[1]) if kind == "platt" else ((th[0], 0.0) if kind == "temp" else (1.0, th[0]))
        s = a * z + b
        return float(np.mean(np.logaddexp(0, s) - y * s))
    x0 = {"platt": [1.0, 0.0], "temp": [1.0], "intercept": [0.0]}[kind]
    r = minimize(nll, x0, method="L-BFGS-B")
    th = r.x
    a, b = (th[0], th[1]) if kind == "platt" else ((th[0], 0.0) if kind == "temp" else (1.0, th[0]))
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0:
        return None
    return a, b


def bss(y, p):
    base = y.mean()
    ref = np.mean((base - y) ** 2)
    return float(1 - np.mean((p - y) ** 2) / ref) if ref > 0 else np.nan


def ece(y, p, bins=10):
    e = np.clip((p * bins).astype(int), 0, bins - 1)
    out = 0.0
    for b in range(bins):
        m = e == b
        if m.any():
            out += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(out)


def sources():
    """(task, model, y, groups, p) for every stored benchmark cell, FMs included when present."""
    for f in sorted(glob.glob(str(REPO / "results/_oof/unified_*.npz"))):
        task = Path(f).stem.replace("unified_", "").split("_e")[0]
        with np.load(f, allow_pickle=False) as z:
            y, g = z["_y"].astype(int), z["_groups"].astype(str)
            for k in z.files:
                if not k.startswith("_") and k != "auranet_anchor":
                    yield task, k, y, g, z[k].astype(float)
    for f in sorted(glob.glob(str(REPO / "results/_oof_variants/variant_*_subject_none_e25_*_fm.npz"))):
        task = Path(f).stem.split("_")[1]
        with np.load(f, allow_pickle=False) as z:
            y, g = z["_y"].astype(int), z["_groups"].astype(str)
            for k in z.files:
                if k.endswith("@s0"):
                    yield task, k.split("@")[0], y, g, z[k].astype(float)


def main() -> int:
    from sklearn.metrics import roc_auc_score

    from bsf.utils.io import save_json
    from bsf.utils.provenance import new_results_dir
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cal-frac", type=float, default=0.3)
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()
    out = new_results_dir("calibration_xfit", seed=0, config=vars(args))
    rows = ["task,model,auc,bss_raw,ece_raw,bss_platt,ece_platt,frac_pos_platt,bss_temp,"
            "bss_intercept,anti_calibrated_frac"]
    cells = {}
    for task, model, y, g, p in sources():
        subs = np.unique(g)
        rng = np.random.default_rng(0)
        rec = {k: [] for k in ("bss_raw", "ece_raw", "bss_platt", "ece_platt", "bss_temp",
                               "bss_intercept", "anti")}
        for _ in range(args.repeats):
            cal = rng.choice(subs, max(2, int(round(args.cal_frac * len(subs)))), replace=False)
            mc = np.isin(g, cal); me = ~mc
            if len(np.unique(y[mc])) < 2 or len(np.unique(y[me])) < 2:
                continue
            z = _logit(p)
            rec["bss_raw"].append(bss(y[me], p[me])); rec["ece_raw"].append(ece(y[me], p[me]))
            for kind in ("platt", "temp", "intercept"):
                ab = fit_map(z[mc], y[mc], kind)
                if ab is None:
                    if kind == "platt":
                        rec["anti"].append(1); rec["bss_platt"].append(np.nan)
                        rec["ece_platt"].append(np.nan)
                    else:
                        rec[f"bss_{kind}"].append(np.nan)
                    continue
                q = 1 / (1 + np.exp(-(ab[0] * z[me] + ab[1])))
                if kind == "platt":
                    rec["anti"].append(0)
                    rec["bss_platt"].append(bss(y[me], q)); rec["ece_platt"].append(ece(y[me], q))
                else:
                    rec[f"bss_{kind}"].append(bss(y[me], q))
        med = {k: (float(np.nanmedian(v)) if len(v) and np.isfinite(np.nanmedian(v)) else None)
               for k, v in rec.items() if k != "anti"}
        bp = np.asarray(rec["bss_platt"], float)
        c = {"auc": round(float(roc_auc_score(y, p)), 4),
             **{k: (round(v, 4) if v is not None else None) for k, v in med.items()},
             "frac_pos_platt": round(float(np.mean(np.nan_to_num(bp, nan=-1) > 0)), 3),
             "anti_calibrated_frac": round(float(np.mean(rec["anti"])) if rec["anti"] else 0.0, 3)}
        cells.setdefault(task, {})[model] = c
        rows.append(f"{task},{model},{c['auc']},{c['bss_raw']},{c['ece_raw']},{c['bss_platt']},"
                    f"{c['ece_platt']},{c['frac_pos_platt']},{c['bss_temp']},{c['bss_intercept']},"
                    f"{c['anti_calibrated_frac']}")
    summ = {}
    for task, ms in cells.items():
        v = list(ms.values())
        summ[task] = {"n": len(v),
                      "usable_raw": sum(1 for c in v if (c["bss_raw"] or -1) > 0),
                      "usable_platt": sum(1 for c in v if (c["bss_platt"] or -1) > 0),
                      "usable_temp": sum(1 for c in v if (c["bss_temp"] or -1) > 0),
                      "usable_intercept": sum(1 for c in v if (c["bss_intercept"] or -1) > 0),
                      "anti_calibrated": sum(1 for c in v if c["anti_calibrated_frac"] >= 0.5),
                      "median_ece_raw": round(float(np.median([c["ece_raw"] for c in v])), 4),
                      "median_ece_platt": round(float(np.nanmedian([c["ece_platt"] if c["ece_platt"]
                                                                    is not None else np.nan for c in v])), 4)}
        print(f"[{task}] usable {summ[task]['usable_raw']}/{len(v)} -> platt "
              f"{summ[task]['usable_platt']} temp {summ[task]['usable_temp']} "
              f"intercept {summ[task]['usable_intercept']} | anti {summ[task]['anti_calibrated']} | "
              f"median ECE {summ[task]['median_ece_raw']} -> {summ[task]['median_ece_platt']}",
              flush=True)
    tot = {k: sum(s[k] for s in summ.values()) for k in ("n", "usable_raw", "usable_platt",
                                                         "usable_temp", "usable_intercept",
                                                         "anti_calibrated")}
    print("TOTAL", tot, flush=True)
    save_json({"note": __doc__.split("\n")[0], "summary": summ, "total": tot, "cells": cells},
              out / "results.json")
    (out / "cells.csv").write_text("\n".join(rows))
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
