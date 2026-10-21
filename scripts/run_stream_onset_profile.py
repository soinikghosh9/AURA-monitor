#!/usr/bin/env python
"""What a refreshed-baseline alarm means: each AURA feature's departure from the preceding 30 min.

For every development stream of the continuous-EEG replication, each scored window's 18 features are
robust-z-scored by the stream's own windows from 35 to 5 min before it (the refreshed baseline; no
labels, no model). For every seizure the median z of each feature over its ictal windows is taken; the
summary over seizures (median, interquartile range, share of seizures beyond +-2) says which
physiological quantities depart from the patient's recent state during a seizure, i.e. the explanation a
reviewer would be shown with an alarm. Non-ictal windows give the reference spread.

-> results/stream_onset_<stamp>/results.json
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
    from bsf.data.features import FEATURE_NAMES_PLUS
    from bsf.utils.io import save_json
    from bsf.utils.provenance import new_results_dir
    from make_paper_numbers import latest
    from run_stream_nets import load_streams
    from run_stream_replication import RULE, _rolling
    from run_streaming_monitor import _stream_features

    rep = latest("stream_replication_*")
    R = json.loads((rep / "results.json").read_text())
    subjects = sorted(R["streams"])
    streams = load_streams(subjects, RULE["max_hours"])
    out = new_results_dir("stream_onset", seed=0, config={"replication": rep.name})
    per_seizure, nonictal = [], []
    for s, st in streams.items():
        X = np.load(st["file"])["X"]
        F = _stream_features({"subject": s, "X": X}, "detect", RULE["fs"], RULE["channels"], 2000)[st["order"]]
        del X
        Z = _rolling(F, st["t"], st["cut"], RULE["rolling_min"] * 60, RULE["gap_min"] * 60,
                     RULE["min_ref_windows"])
        y = st["y"][st["cut"]:]
        t = st["t"][st["cut"]:]
        nonictal.append(Z[y == 0])
        idx = np.flatnonzero(y)
        if idx.size == 0:
            continue
        runs = np.split(idx, np.where((np.diff(idx) > 1) | (np.diff(t[idx]) > 1.5 * RULE["window_s"]))[0] + 1)
        for r in runs:
            per_seizure.append({"subject": s, "n_windows": int(len(r)), "z": np.median(Z[r], 0)})
    Zs = np.stack([p["z"] for p in per_seizure])
    Zn = np.concatenate(nonictal)
    feats = {}
    for j, name in enumerate(FEATURE_NAMES_PLUS):
        z = Zs[:, j]
        feats[name] = {"median_z": round(float(np.median(z)), 3),
                       "q25": round(float(np.percentile(z, 25)), 3), "q75": round(float(np.percentile(z, 75)), 3),
                       "frac_above2": round(float(np.mean(z > 2)), 3), "frac_below_m2": round(float(np.mean(z < -2)), 3),
                       "nonictal_abs95": round(float(np.percentile(np.abs(Zn[:, j]), 95)), 3)}
    order = sorted(feats, key=lambda k: -abs(feats[k]["median_z"]))
    res = {"n_seizures": len(per_seizure), "n_streams": len(streams), "features": feats, "ranked": order,
           "per_seizure": [{"subject": p["subject"], "n_windows": p["n_windows"],
                            "z": [round(float(v), 3) for v in p["z"]]} for p in per_seizure],
           "feature_names": FEATURE_NAMES_PLUS}
    save_json(res, out / "results.json")
    for k in order[:8]:
        f = feats[k]
        print(f"  {k:22s} median z {f['median_z']:+.2f} [{f['q25']:+.2f}, {f['q75']:+.2f}]  >2: {f['frac_above2']:.2f}"
              f"  <-2: {f['frac_below_m2']:.2f}  non-ictal |z| p95 {f['nonictal_abs95']:.2f}", flush=True)
    print(f"-> {out} ({len(per_seizure)} seizures)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
