#!/usr/bin/env python
"""Resumable, offline, two-lane queue for the pre-submission experiment program (detached).

What the unified benchmark could not answer, run as ``run_benchmark_variants.py`` units:

  lane A  fm_*     real pretraining: BIOT and CBraMod with their published weights, the matched
                   random-init controls on identical 200 Hz input and recipe, and frozen-encoder
                   linear probes. 2 seeds.
          align_*  patient alignment at the input: none vs label-free (every clip of the patient)
                   vs baseline-referenced (a reserved set of the patient's negatives), paired on
                   identical clips, for the analytic arms and six deep models spanning the families.
  lane B  seeds_*  seeds 1 and 2 for each task's development top-8, paired with the benchmark's own
                   run as the third draw. The benchmark trained each model once, and this project has
                   measured seed spreads as large as the architecture effects being claimed.
          proto_*  the protocol factorial: the same models under segment-random and event-grouped
                   folds, so the protocol effect is measured on deep models too, not only on one
                   feature model, and on the same metric as the cross-patient number.

  lane E  the tail: detection seeds (top 3) and the detection learning-rate check, after lane B's
          core units; then a second finalize once lane D (addendum + first finalize) is done.

Rescheduled 2026-09-29: the scalp and detection units ran ~2.5x slower per arm than the iEEG ones
(2.6-3.6x the clips, one GPU shared by three lanes), projecting ~5 days. Paper-critical units now run
first and lane D no longer waits for the tail (lane E). Two units were cut: BIOT's pretraining arms on
detection (its checkpoint was pretrained on CHB-MIT chb01-chb20, most of that cohort) and the 50-epoch
detection recipe run (the typing recipe units already test the epoch budget).

Lanes share the GPU; every unit writes its own resume key and OOF file, so they never race.
Journals at results/_queue_final/journal_<lane>.json. DEV subjects only -- the sealed TEST split is
not touched by any unit here. Launch with scripts/run_final.ps1 -Lane A|B.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
QDIR = REPO / "results" / "_queue_final"
PY = str(REPO / ".venv" / "Scripts" / "python.exe")
if not Path(PY).exists():
    PY = sys.executable
V = "scripts/run_benchmark_variants.py"

FM = "BIOT_pt,CBraMod_pt,BIOT_scratch200,CBraMod_scratch200,BIOT_probe,CBraMod_probe"
# detection streams CHB-MIT, most of which BIOT's checkpoint was pretrained on: CBraMod only
FM_DETECT = "CBraMod_pt,CBraMod_scratch200,CBraMod_probe"
# Four deep models spanning the families (compact depthwise, spatial-component, transformer, and the
# repository CNN). The first launch used seven for alignment and six for the protocol factorial; on one
# GPU shared by three lanes that projected to ~3 days, and the extra models were within-family repeats.
ALIGN_MODELS = "feat_logistic,feat_gbm,feat_white,EEGNet,SCCNet,BIOT,deep_cnn_raw"
PROTO_MODELS = "feat_logistic,feat_gbm,EEGNet,SCCNet,BIOT,deep_cnn_raw"
# each task's development top-8 deep models (unified_benchmark leaderboards, AURA-Net excluded: it
# already has three seeds from its own ablation)
# top-6 (was top-8; typing had already run its first five when this was cut, and keeps them)
TOP8 = {
    "typing": "CTNet,EEGInceptionERP,SPARCNet,EEGNet,MSVTNet,EEGNeX",
    "ieeg": "BIOT,ContraWR,SCCNet,EEGNet,EEGSimpleConv,FBCNet",
    "scalp": "SCCNet,EEGTCNet,BIOT,ShallowFBCSPNet,ContraWR,SPARCNet",
    "detect": "BIOT,ContraWR,ATCNet,FBCNet,EEGSimpleConv,EEGTCNet",
}
TASKS = ["typing", "ieeg", "scalp", "detect"]


def _lane_a():
    units = []
    for t in TASKS:
        fm = {"name": f"fm_{t}", "argv": [V, "--task", t, "--models", FM_DETECT if t == "detect" else FM,
                                          "--seeds", "0,1", "--tag", "fm"]}
        al = [{"name": f"align_{t}_{a}",
               "argv": [V, "--task", t, "--align", a, "--drop-baseline",
                        "--models", ALIGN_MODELS, "--seeds", "0", "--tag", "align"]}
              for a in ("none", "ea_all", "ea_base")]
        # detection runs alignment before pretraining: alignment feeds the TEST addendum's arms
        units += (al + [fm]) if t == "detect" else ([fm] + al)
    return units


def _lane_b():
    units = []
    for t in TASKS:
        if t != "detect":   # detection seeds moved to the tail (lane E)
            units.append({"name": f"seeds_{t}", "argv": [V, "--task", t, "--models", TOP8[t],
                                                         "--seeds", "1,2", "--tag", "seeds"]})
        for pr in (("segment",) if t == "typing" else ("segment", "event")):
            units.append({"name": f"proto_{t}_{pr}",
                          "argv": [V, "--task", t, "--protocol", pr, "--models", PROTO_MODELS,
                                   "--seeds", "0", "--tag", "proto"]})
    # recipe sensitivity: is the shared recipe (lr 1e-3, 25 epochs) under-training anyone? Four
    # networks spanning the families on typing (most patients per class) at a lower learning rate and
    # at twice the epochs; detection's learning-rate check runs in the tail
    units += _recipe("typing", ("lr3e-4", "e50"))
    return units


def _recipe(t, which):
    out = []
    for w in which:
        extra = ["--lr", "3e-4"] if w == "lr3e-4" else ["--epochs", "50"]
        out.append({"name": f"recipe_{t}_{w}",
                    "argv": [V, "--task", t, "--models", "EEGNet,SCCNet,BIOT,deep_cnn_raw",
                             "--seeds", "0", *extra, "--tag", "recipe"]})
    return out


def _lane_e():
    # the tail, which the paper does not wait for: it starts when lane B's core is done (so it never
    # adds a fourth process to the GPU), and re-finalizes after lane D so its numbers are folded in
    return [# rescore the streaming detector's analytic arms on matched input once lane C's detect unit
            # (still running the old feature code) is done; its networks resume from the checkpoint
            {"name": "wait_stream_detect", "argv": ["scripts/wait_for_lanes.py", "C:stream_detect_v2"]},
            {"name": "stream_detect_v2_feats", "argv": _lane_c()[0]["argv"]},
            {"name": "wait_lane_B", "argv": ["scripts/wait_for_lanes.py", "B"]},
            {"name": "seeds_detect", "argv": [V, "--task", "detect", "--models",
                                              ",".join(TOP8["detect"].split(",")[:3]),
                                              "--seeds", "1,2", "--tag", "seeds"]},
            *_recipe("detect", ("lr3e-4",)),
            {"name": "wait_lane_D", "argv": ["scripts/wait_for_lanes.py", "D"]},
            {"name": "finalize_paper_tail", "argv": ["scripts/finalize_paper.py"]}]


def _lane_c():
    # the streaming monitor, rerun after fixing its input scaling (it streamed raw volts into models
    # trained on standardised windows); analytic arms and the development leaders now included
    return [{"name": f"stream_{m}_v2",
             "argv": ["scripts/run_streaming_monitor.py", "--mode", m, "--epochs", "25",
                      "--max-records", "6", "--max-hours", "12"]} for m in ("detect", "forecast")]


def _lane_d():
    # the single-pass TEST addendum, once the development lanes have released the GPU; its arms and
    # scoring rule were declared in results/_blind_addendum_manifest.json before they were run
    return [{"name": "wait_dev_lanes", "argv": ["scripts/wait_for_lanes.py", "A", "B", "C",
                                                "E:stream_detect_v2_feats"]},
            {"name": "blind_addendum",
             "argv": ["scripts/run_blind_addendum.py", "--epochs", "25", "--i-am-ready"]},
            # regenerate every statistic, number, table and figure once everything above is in
            {"name": "finalize_paper", "argv": ["scripts/finalize_paper.py"]}]


def _lane_f():
    # AURA-Net with a working anchor: every patient anchored from a baseline that precedes or lies
    # outside the scored data, against the unanchored model on identical clips (3 seeds each)
    return [{"name": f"aura_anchor_{c}", "argv": ["scripts/run_aura_anchor.py", "--configs", c]}
            for c in ("pure", "hybrid")]


def _lane_g():
    # the networks' baseline alignment rerun with the baseline taken from each patient's earliest
    # negatives, so that it precedes every scored clip (detection first: that is where the gain was)
    return [{"name": f"align_{t}_{a}_early",
             "argv": [V, "--task", t, "--align", a, "--drop-baseline", "--baseline", "early",
                      "--models", ALIGN_MODELS, "--seeds", "0", "--tag", "align"]}
            for t in ("detect", "scalp") for a in ("none", "ea_base")]


def _lane_h():
    return [{"name": "wait_fg", "argv": ["scripts/wait_for_lanes.py", "F", "G"]},
            {"name": "finalize_paper_fg", "argv": ["scripts/finalize_paper.py"]}]


LANES = {"A": _lane_a(), "B": _lane_b(), "C": _lane_c(), "D": _lane_d(), "E": _lane_e(),
         "F": _lane_f(), "G": _lane_g(), "H": _lane_h()}


def _now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _save(path: Path, j: dict):
    QDIR.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(QDIR), suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(j, fh, indent=2)
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lane", required=True, choices=sorted(LANES))
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    units = LANES[args.lane]
    if args.list:
        for u in units:
            print(u["name"], " ".join(u["argv"]))
        return 0
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1",
                      HF_HOME=str(REPO / "models" / "hf_cache"))
    journal_path = QDIR / f"journal_{args.lane}.json"
    journal = json.loads(journal_path.read_text()) if journal_path.exists() else {}
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    print(f"\n===== final program lane {args.lane} @ {_now()} =====", flush=True)
    print(f"units: {[u['name'] for u in units]}", flush=True)
    failed = []
    for u in units:
        name = u["name"]
        if only and name not in only:
            continue
        prev = journal.get(name, {})
        if prev.get("status") in ("done", "gave_up"):
            print(f"[skip] {name} ({prev.get('status')})", flush=True)
            continue
        attempts = int(prev.get("attempts", 0))
        while True:
            attempts += 1
            journal[name] = {"status": "running", "stamp": _now(), "attempts": attempts}
            _save(journal_path, journal)
            print(f"\n----- [{_now()}] RUN {name}: {' '.join(u['argv'])} -----", flush=True)
            t0 = dt.datetime.now()
            rc = subprocess.run([PY, *u["argv"]], cwd=str(REPO)).returncode
            dur = (dt.datetime.now() - t0).total_seconds()
            # a unit that fails three times is recorded as given up rather than retried forever by the
            # watchdog; its error is in the log and it can be re-queued by deleting its journal entry
            status = "done" if rc == 0 else ("gave_up" if attempts >= 3 else "failed")
            journal[name] = {"status": status, "rc": rc, "stamp": _now(), "seconds": round(dur, 1),
                             "attempts": attempts}
            _save(journal_path, journal)
            print(f"----- [{_now()}] {name} -> rc={rc} in {dur:.0f}s -----", flush=True)
            if status != "failed":
                break
            # the usual failure is transient (a CUDA OOM when the lanes' combined host commit
            # peaks); the unit resumes from its checkpoints, so retry it here after a pause rather
            # than leaving it for the next launch of the lane
            print(f"      retrying {name} in 10 min (attempt {attempts + 1} of 3)", flush=True)
            time.sleep(600)
        if rc != 0:
            failed.append(name)
            if not args.keep_going:
                return rc
    print(f"\n===== lane {args.lane} COMPLETE @ {_now()} failed={failed} =====", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
