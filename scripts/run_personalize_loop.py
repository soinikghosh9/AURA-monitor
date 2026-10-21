#!/usr/bin/env python
"""Monitor module M6 on the 25 development streams: does the clinician-in-the-loop adaptation help?

For each stream, the AURA-Ens probabilities of ``run_aura_ens.py`` (fold that held the patient out),
the threshold for the false-alarm budget from the other streams, and a sequential replay in which every
alarm is reviewed at the end of each hour (confirmed if it overlaps an annotated seizure within the
SzCORE tolerance) and the patient head and threshold are updated (``bsf.monitor.personalize``).
Scored from 10 min after the first confirmed seizure to the end of the stream, against the same span
without adaptation: seizures detected, false alarms per 24 h, F1; sign-flip test over streams on F1. The
adapted arm is scored with the threshold in force at each window (the decisions the monitor made).

-> results/personalize_loop_<stamp>/results.json
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
    import argparse

    from bsf.eval import szcore
    from bsf.monitor.personalize import Personalizer
    from bsf.utils.io import save_json
    from bsf.utils.provenance import new_results_dir
    from run_final_stats import signflip
    from run_stream_nets import load_streams
    from run_stream_replication import RULE, _rolling, smooth
    from run_streaming_monitor import _stream_features

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ens", default="results/aura_ens_db18")
    ap.add_argument("--arm", default="aura_ens|post")
    ap.add_argument("--budgets", default="6,24")
    a = ap.parse_args()
    ens = REPO / a.ens
    R0 = json.loads((ens / "results.json").read_text())
    pr = np.load(ens / "preds.npz", allow_pickle=True)
    G = pr["_groups"].astype(str); subj = sorted(np.unique(G))
    streams = load_streams(subj, RULE["max_hours"], R0["montage"])
    out = new_results_dir("personalize_loop", seed=0, config=vars(a))
    span, gap = RULE["rolling_min"] * 60, RULE["gap_min"] * 60
    Fz = {}
    for s in subj:
        st = streams[s]
        F = _stream_features({"subject": f"{s}_{R0['montage']}", "X": np.load(st["file"])["X"]}, "detect",
                             RULE["fs"], 18 if R0["montage"] == "db18" else RULE["channels"], 2000)[st["order"]]
        Fz[s] = _rolling(F, st["t"], st["cut"], span, gap, RULE["min_ref_windows"])
    R = {"arm": a.arm, "budgets": {}}
    for b in [float(x) for x in a.budgets.split(",")]:
        rows = []
        for s in subj:
            m = G == s
            y, p, t = pr["_y"][m].astype(int), pr[a.arm][m].astype(float), pr["_t"][m]
            others = [(pr["_y"][G == o], smooth(pr[a.arm][G == o], pr["_t"][G == o]), pr["_t"][G == o]) for o in subj if o != s]
            thr = szcore.threshold_for_budget(others, b)
            per = Personalizer(thr, b)
            rep = per.replay(y, p, Fz[s], t)
            first = next((e["t_on"] for e in rep["events"] if e["confirmed"]), None)
            if first is None:
                rows.append({"stream": s, "first_confirmed": None}); continue
            late = t > first + 600
            if not late.any() or y[late].sum() == 0:
                rows.append({"stream": s, "first_confirmed": first, "no_later_seizure": True}); continue
            c0 = szcore.events(y[late], smooth(p, t)[late], t[late], thr)
            online = smooth(rep["scores"], t) - rep["thr_t"]          # the decisions made online
            c1 = szcore.events(y[late], online[late], t[late], 0.0)
            rows.append({"stream": s, "first_confirmed": first, "before": c0, "after": c1, "threshold": [thr, rep["final_threshold"]]})
        ok = [r for r in rows if "after" in r]
        f1 = lambda c: szcore.summarize([c])["f1"]  # noqa: E731
        d = [f1(r["after"]) - f1(r["before"]) for r in ok]
        R["budgets"][str(int(b))] = {
            "n_streams_scored": len(ok), "n_streams": len(subj),
            "before": szcore.summarize([r["before"] for r in ok]), "after": szcore.summarize([r["after"] for r in ok]),
            "f1_delta_mean": round(float(np.mean(d)), 4) if d else None, "n_improved": int(sum(x > 0 for x in d)),
            "n_worse": int(sum(x < 0 for x in d)), "p_signflip": signflip(d) if d else None,
            "streams": [{k: v for k, v in r.items() if k not in ("before", "after")} | ({"f1": [f1(r["before"]), f1(r["after"])]} if "after" in r else {}) for r in rows]}
        q = R["budgets"][str(int(b))]
        print(f"[budget {b:g}/24h] {len(ok)} streams: before sens {q['before']['sensitivity']:.2f} FA {q['before']['fa_per_24h']} | "
              f"after sens {q['after']['sensitivity']:.2f} FA {q['after']['fa_per_24h']} | F1 delta {q['f1_delta_mean']} "
              f"({q['n_improved']} up, {q['n_worse']} down, p {q['p_signflip']})", flush=True)
    save_json(R, out / "results.json")
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
