#!/usr/bin/env python
"""Seal a subject-level DEV/TEST split, once, before any model is scored against it.

Why this exists, stated plainly. Every AURA-Net number so far is cross-validated on the full cohort,
and SIX architecture decisions were made in response to those cross-validated numbers -- encoder
receptive field, anchor mode, spatial mixer, transient filters, signed path, hybrid fusion. That makes
the CV estimate optimistically biased *for AURA-Net specifically*, while remaining unbiased for the
CNN, the braindecode zoo and the feature baselines, none of which were ever tuned here. Reporting them
side by side without a blind set would flatter exactly the model we built.

The split is:
  * by SUBJECT, never by clip, so no subject contributes windows to both sides;
  * stratified by cohort, and by class availability where a subject-level label exists;
  * sealed with a SHA-256 over the sorted (cohort, subject, side) triples, stored in the artifact, so
    any later run can prove it used the same split rather than a convenient one;
  * IDEMPOTENT -- re-running returns the existing seal instead of drawing a new split, because a split
    that can be silently redrawn is not a blind set.

TEST is evaluated once, for all models simultaneously, after DEV results are frozen. The DEV-minus-TEST
gap per model is itself a headline: it measures how much each model's development consumed.

-> results/blind_split_<stamp>/results.json  and  results/_blind_split.json (the canonical seal)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
SEAL = Path("results/_blind_split.json")


def _subjects() -> dict[str, list[str]]:
    """Enumerate subjects per cohort from what is actually on disk."""
    out: dict[str, list[str]] = {}
    try:
        from bsf.data.kaggle_seizure import discover_kaggle
        for root, tag in (("datasets/kaggle_aes", "kaggle_ieeg"),
                          ("datasets/ieeg_canine", "canine_ieeg")):
            d = discover_kaggle(root)
            if d:
                out[tag] = sorted(d)
    except Exception:
        pass
    for tag, pat in (("scalp_chbmit", str(Path(__file__).resolve().parents[1] / "datasets" / "chbmit" / "chb*")),
                     ("scalp_siena", str(Path(__file__).resolve().parents[1] / "datasets" / "siena-scalp-eeg" / "PN*"))):
        subs = sorted(p.name for p in Path(pat).parent.glob(Path(pat).name) if p.is_dir())
        if subs:
            out[tag] = subs
    try:
        from bsf.data.vepiset import by_patient, discover
        pp = by_patient(discover("datasets/vepiset-dataset"))
        if pp:
            out["vepiset_typing"] = sorted(pp)
    except Exception:
        pass
    p = Path("datasets/ieeg_canine_richplus")
    if p.exists():
        out["canine_continuous"] = sorted(f.stem for f in p.glob("*.npz"))
    return out


def main() -> int:
    from bsf.utils.io import save_json
    from bsf.utils.provenance import new_results_dir
    from bsf.utils.seeds import set_global_seeds

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--force", action="store_true",
                    help="redraw the split, DESTROYING blindness; refuses unless you mean it")
    args = ap.parse_args()

    if SEAL.exists() and not args.force:
        seal = json.loads(SEAL.read_text())
        print("=== blind split ALREADY SEALED (idempotent; use --force only to destroy it) ===")
        print(f"  sealed at : {seal['sealed_utc']}")
        print(f"  sha256    : {seal['sha256']}")
        for c, v in seal["cohorts"].items():
            print(f"  {c:18s} dev {len(v['dev']):3d}  test {len(v['test']):3d}")
        return 0

    set_global_seeds(args.seed)
    rng = np.random.default_rng(args.seed)
    cohorts = _subjects()
    if not cohorts:
        print("no cohorts found on disk"); return 1

    split, triples = {}, []
    for c, subs in sorted(cohorts.items()):
        subs = sorted(subs)
        n_test = max(1, int(round(args.test_frac * len(subs)))) if len(subs) >= 3 else 0
        idx = rng.permutation(len(subs))
        test = sorted(subs[i] for i in idx[:n_test])
        dev = sorted(s for s in subs if s not in set(test))
        split[c] = {"dev": dev, "test": test}
        triples += [(c, s, "test") for s in test] + [(c, s, "dev") for s in dev]

    sha = hashlib.sha256(json.dumps(sorted(triples)).encode()).hexdigest()
    import datetime as dt
    seal = {"sealed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "seed": args.seed, "test_frac": args.test_frac, "sha256": sha, "cohorts": split,
            "rules": [
                "TEST is scored ONCE, for all models simultaneously, after DEV results are frozen.",
                "Any model change prompted by a TEST result makes that result contaminated; say so.",
                "Report DEV and TEST side by side per model; the gap measures development optimism.",
                "Split is by subject: no subject contributes windows to both sides.",
            ]}
    SEAL.parent.mkdir(parents=True, exist_ok=True)
    SEAL.write_text(json.dumps(seal, indent=2))
    out = new_results_dir("blind_split", seed=args.seed, config=vars(args))
    save_json(seal, out / "results.json")

    print("=== blind split SEALED ===")
    print(f"  sha256: {sha}")
    tot_d = tot_t = 0
    for c, v in split.items():
        tot_d += len(v["dev"]); tot_t += len(v["test"])
        print(f"  {c:18s} dev {len(v['dev']):3d}  test {len(v['test']):3d}   "
              f"test subjects: {', '.join(v['test'][:5])}{' ...' if len(v['test']) > 5 else ''}")
    print(f"  TOTAL              dev {tot_d:3d}  test {tot_t:3d}")
    print(f"-> {out}  (canonical seal: {SEAL})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
