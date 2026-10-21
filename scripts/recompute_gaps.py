#!/usr/bin/env python
"""Recompute the DEV-minus-TEST optimism gap against the CURRENT development tables.

The blind set is not re-spent. TEST predictions were scored once and are stored; this only corrects
which development number they are differenced against. That reference was wrong: the lookup kept the
OLDEST entry per model instead of the newest, so every gap was measured against a superseded DEV run,
and AURA-Net -- refit after its first entries were written -- was compared to a model that no longer
exists.

-> results/blind_gaps_<stamp>/results.json + gaps.csv
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    from bsf.utils.io import latest_result_dirs, save_json
    from bsf.utils.provenance import new_results_dir
    out = new_results_dir("blind_gaps", seed=0, config={})
    dev = {}
    for d in latest_result_dirs("unified_benchmark_*"):          # oldest first; newest overwrites
        f = Path(d) / "results.json"
        if not f.exists():
            continue
        for task, tr in json.loads(f.read_text()).get("tasks", {}).items():
            if isinstance(tr, dict):
                for m, r in tr.items():
                    if isinstance(r, dict) and "panel" in r:
                        dev.setdefault(task, {})[m] = r["panel"]["discrimination"]["auc"]
    bt = sorted(Path("results").glob("blind_test_*"))
    if not bt:
        print("no blind test result found"); return 1
    res = json.loads((bt[-1] / "results.json").read_text())["tasks"]
    rows, tasks = ["task,model,test_auc,dev_auc_current,gap"], {}
    for task, v in res.items():
        if not isinstance(v, dict):
            continue
        per, gaps = {}, []
        for m, r in v.items():
            if not isinstance(r, dict) or "panel" not in r:
                continue
            a = r["panel"]["discrimination"]["auc"]
            d0 = dev.get(task, {}).get(m)
            g = None if d0 is None else round(d0 - a, 4)
            per[m] = {"test": a, "dev": d0, "gap": g}
            if g is not None:
                gaps.append(g)
            rows.append(f"{task},{m},{a},{'' if d0 is None else d0},{'' if g is None else g}")
        if gaps:
            tasks[task] = {"models": per, "mean_gap": round(float(np.mean(gaps)), 4),
                           "median_gap": round(float(np.median(gaps)), 4),
                           "n_positive": int(sum(1 for g in gaps if g > 0)), "n": len(gaps)}
            print(f"  {task:7s} mean gap {np.mean(gaps):+.3f}  median {np.median(gaps):+.3f}  "
                  f"{sum(1 for g in gaps if g > 0)}/{len(gaps)} positive", flush=True)
    save_json({"note": "gaps recomputed against the CURRENT dev tables; TEST scores unchanged",
               "blind_test_dir": bt[-1].name, "tasks": tasks}, out / "results.json")
    (out / "gaps.csv").write_text("\n".join(rows))
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
