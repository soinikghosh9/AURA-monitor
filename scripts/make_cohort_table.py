#!/usr/bin/env python
"""Participant characteristics per task and split (TRIPOD+AI items 5 and 20), from each dataset's own metadata.

Patients per task come from the benchmark's development out-of-fold files (development) and the locked seal
(held-out); continuous recordings from the locked stream passes. Ages and sexes come from CHB-MIT SUBJECT-INFO,
Siena subject_info.csv, SeizeIT2 participants.tsv and vEpiSet base_info.csv.

  python scripts/make_cohort_table.py   -> results/cohort_table_<stamp>/results.json
"""
from __future__ import annotations

import csv
import glob
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "datasets"
SEAL = json.loads((ROOT / "results/_blind_split.json").read_text())["cohorts"]


def chb_meta():
    out = {}
    for line in (DS / "chbmit/SUBJECT-INFO").read_text(errors="ignore").splitlines()[1:]:
        p = line.split()
        if len(p) >= 3:
            out[p[0]] = {"sex": p[1].upper(), "age": float(p[2])}
    out.setdefault("chb24", {"sex": None, "age": None})    # chb24 was added later without demographics
    return out


def siena_meta():
    out = {}
    with open(DS / "siena-scalp-eeg/subject_info.csv", newline="") as f:
        for r in csv.DictReader(f, skipinitialspace=True):
            out[r["patient_id"].strip()] = {"sex": r["gender"].strip()[0].upper(), "age": float(r["age_years"]),
                                            "seizure": r["seizure"].strip(), "localization": r["localization"].strip(),
                                            "n_seizures": int(r["number_seizures"]), "rec_min": float(r["rec_time_minutes"])}
    return out


def vepiset_meta():
    out = {}
    with open(DS / "vepiset-dataset/MAT_Files/base_info.csv", newline="") as f:
        for r in csv.DictReader(f):
            age = r["age"].strip()
            sex = r["sex"].strip()
            out[r["file_name"].strip()] = {"sex": sex[0].upper() if sex and sex != "None" else None,
                                           "age": float(age) if age.replace(".", "").isdigit() else None}
    return out


def summarize(ids, meta):
    rows = [meta.get(i, {}) for i in ids]
    ages = [r["age"] for r in rows if r.get("age") is not None]
    sexes = [r.get("sex") for r in rows if r.get("sex")]
    d = {"n": len(ids)}
    if ages:
        d["age_median"] = round(float(np.median(ages)), 1)
        d["age_range"] = [round(float(min(ages)), 1), round(float(max(ages)), 1)]
    if sexes:
        d["female"] = int(sum(s == "F" for s in sexes))
    d["missing_demographics"] = int(len(ids) - len(ages))
    return d


def dev_ids(task):
    f = sorted(glob.glob(str(ROOT / f"results/_oof/unified_{task}_*.npz")))[-1]
    return sorted({str(s) for s in np.load(f, allow_pickle=True)["_groups"]})


def main() -> int:
    out = ROOT / "results" / f"cohort_table_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out.mkdir(parents=True)
    chb, sie, vep = chb_meta(), siena_meta(), vepiset_meta()
    R: dict = {"tasks": {}, "streams": {}, "seizeit2": {}}

    def split_scalp(ids):
        c = [i.split("_", 1)[1] for i in ids if i.startswith("chbmit_")]
        s = [i.split("_", 1)[1] for i in ids if i.startswith("siena_")]
        return {"CHB-MIT": summarize(c, chb), "Siena": summarize(s, sie)}

    for task in ("scalp", "detect"):
        dev = dev_ids(task)
        test = [f"chbmit_{s}" for s in SEAL["scalp_chbmit"]["test"]] + [f"siena_{s}" for s in SEAL["scalp_siena"]["test"]]
        R["tasks"][task] = {"development": split_scalp(dev), "held_out_listed": split_scalp(test)}
    dev = [i.split("_", 1)[1] if "_" in i else i for i in dev_ids("typing")]
    R["tasks"]["typing"] = {"development": summarize(dev, vep),
                            "held_out_listed": summarize(SEAL["vepiset_typing"]["test"], vep)}
    R["tasks"]["ieeg"] = {"development": {"subjects": dev_ids("ieeg"), "species": "dog (all)"},
                          "held_out_listed": {"subjects": SEAL["kaggle_ieeg"]["test"] + SEAL["canine_ieeg"]["test"],
                                              "species": "2 human (Kaggle Patient_1, Patient_2), 2 dog"}}
    # seizure types and localization for Siena (the only scalp cohort that reports them)
    R["siena_seizure_types"] = {k: {"seizure": v["seizure"], "localization": v["localization"]} for k, v in sie.items()}
    # continuous recordings
    for name, d in (("development", "stream_replication_20261004T124204Z"), ("held_out", "stream_test_20261004T200300Z")):
        z = np.load(ROOT / "results" / d / "preds.npz", allow_pickle=True)
        ids = sorted({str(s) for s in z["_groups"]})
        R["streams"][name] = {**split_scalp(ids), "hours": round(float(len(z["_y"]) * 8 / 3600), 1)}
    z = np.load(ROOT / "results/seizeit2_external_20261004T171522Z/preds.npz", allow_pickle=True)
    ids = sorted({str(s) for s in z["_groups"]})
    sz = {}
    with open(DS / "seizeit2/participants.tsv", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            sz[r["participant_id"].strip()] = {"sex": (r.get("sex") or "").strip()[:1].upper() or None,
                                               "age": float(r["age"]) if r.get("age", "").strip().replace(".", "").isdigit() else None}
    R["seizeit2"] = {"held_out_scored": summarize(ids, sz), "hours": round(float(len(z["_y"]) * 8 / 3600), 1),
                     "fields_available": list(next(iter(sz.values())).keys())}
    (out / "results.json").write_text(json.dumps(R, indent=1))
    print(json.dumps({k: v for k, v in R.items() if k != "siena_seizure_types"}, indent=1)[:3000])
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
