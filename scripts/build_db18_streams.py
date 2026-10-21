#!/usr/bin/env python
"""Build the 25 development streams of the continuous-EEG replication on the 18-channel double banana.

Same patients, stretch and window rule as ``run_stream_replication.build_stream``; each recording is
mapped to the common montage by name (``scripts/montage.py``) and recordings that cannot be mapped are
skipped. -> results/_clip_cache/streamrep_<subject>_h12_db18.npz
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


def main() -> int:
    from make_paper_numbers import latest
    from run_stream_replication import RULE, build_stream
    rep = latest("stream_replication_*")
    subjects = sorted(json.loads((rep / "results.json").read_text())["streams"])
    a = SimpleNamespace(max_hours=RULE["max_hours"])
    for subj in subjects:
        cohort, s = subj.split("_", 1)
        st = build_stream(cohort, s, a, "db18")
        nat = np.load(REPO / f"results/_clip_cache/streamrep_{subj}_h{int(a.max_hours)}.npz")
        if st is None:
            print(f"{subj}: no db18 stream", flush=True); continue
        common = np.isin(st["t0s"], nat["t0s"]).mean()
        print(f"{subj}: {st['X'].shape} windows, {int(st['y'].sum())} ictal, {common:.3f} of windows shared with native "
              f"({len(nat['t0s'])} native)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
