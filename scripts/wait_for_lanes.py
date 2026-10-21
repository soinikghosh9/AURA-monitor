#!/usr/bin/env python
"""Block until the named lanes of the final program have finished (every unit done or given up).

Used as the first unit of lane D so the single-pass TEST addendum starts only after the development
lanes have released the GPU. Polls the journals every 10 minutes; exits 0 when all are finished.
An argument ``L:unit`` waits for that one unit of lane L instead of the whole lane.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = str(REPO / ".venv" / "Scripts" / "python.exe")
lanes = sys.argv[1:] or ["A", "B", "C"]


def remaining(spec):
    lane, _, unit = spec.partition(":")
    if unit:
        names = [unit]
    else:
        names = subprocess.run([PY, str(REPO / "scripts" / "run_final_program.py"), "--lane", lane,
                                "--list"], capture_output=True, text=True).stdout.split("\n")
        names = [n.split(" ")[0] for n in names if n.strip()]
    j = REPO / "results" / "_queue_final" / f"journal_{lane}.json"
    done = {k for k, v in json.loads(j.read_text()).items()
            if v.get("status") in ("done", "gave_up")} if j.exists() else set()
    return [n for n in names if n not in done]


while True:
    rem = {lane: remaining(lane) for lane in lanes}
    left = sum(len(v) for v in rem.values())
    print(time.strftime("%Y-%m-%dT%H:%M:%S"), {k: len(v) for k, v in rem.items()}, flush=True)
    if left == 0:
        break
    time.sleep(600)
