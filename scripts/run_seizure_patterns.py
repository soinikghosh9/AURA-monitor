#!/usr/bin/env python
"""Monitor module M4 on the development patients: seizure fingerprints, stereotypy and lateralization.

For every annotated seizure of the development patients (CHB-MIT and Siena), the fingerprint of
``bsf.monitor.patterns`` against the windows 35 to 5 min before onset (seizure-free; extended back to
65 min if fewer than 10 windows). Reported:

  stereotypy       cosine similarity of fingerprints between seizures of the same patient vs different
                   patients (patient-label permutation test)
  seizure types    clusters per patient at similarity 0.6
  lateralization   onset-map side against the Siena clinical lateralization (patients labelled left or
                   right): per-seizure accuracy among lateralized calls, and how often the call abstains

-> results/seizure_patterns_<stamp>/{results.json, fingerprints.json}
"""
from __future__ import annotations

import csv
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
    from scipy.stats import binomtest

    from bsf.monitor import ingest, patterns
    from bsf.utils.io import save_json
    from bsf.utils.provenance import new_results_dir
    from run_unified_benchmark import dev_subjects

    meta = {r["patient_id"]: r for r in csv.DictReader(open(REPO / "data/meta/siena_subject_info.csv"))}
    pats = [("chbmit", s) for s in sorted(dev_subjects("scalp_chbmit", "dev"))]
    pats += [("siena", s) for s in sorted(dev_subjects("scalp_siena", "dev"))]
    out = new_results_dir("seizure_patterns", seed=0, config={"n_patients": len(pats)})
    FP, skipped = [], []
    for cohort, s in pats:
        pdir = REPO / "datasets" / ("chbmit" if cohort == "chbmit" else "siena-scalp-eeg") / s
        try:
            recs, spans = ingest.patient(cohort, pdir)
        except Exception as e:  # noqa: BLE001
            skipped.append({"patient": s, "reason": f"{type(e).__name__}: {e}"[:120]}); continue
        n0 = len(FP)
        for k, (on, off) in enumerate(spans):
            clear = lambda t: all(t + 8 <= a - 300 or t >= b + 300 for a, b in spans)  # noqa: E731
            Wr = Tr = None
            for back in (35, 65):
                Wr, Tr = ingest.segment(recs, on - back * 60, on - 5 * 60)
                keep = np.array([clear(t) for t in Tr], bool) if len(Tr) else np.zeros(0, bool)
                Wr = Wr[keep]
                if len(Wr) >= 10:
                    break
            if Wr is None or len(Wr) < 10:
                skipped.append({"patient": s, "seizure": k, "reason": "no seizure-free reference"}); continue
            Wp, Tp = ingest.segment(recs, on - 32, on + 64)
            if len(Wp) < 6:
                skipped.append({"patient": s, "seizure": k, "reason": "onset not readable"}); continue
            fp = patterns.fingerprint(Wr, Wp, Tp - on, duration_s=float(off - on))
            fp.update({"cohort": cohort, "patient": s, "seizure": k, "onset_abs": on, "time_of_day_h": round((on % 86400) / 3600, 2)})
            fp.pop("evidence")
            FP.append(fp)
        print(f"{cohort} {s}: {len(FP) - n0}/{len(spans)} seizures fingerprinted", flush=True)

    # stereotypy
    V = np.stack([patterns.vector(f) for f in FP]); P = np.array([f["patient"] for f in FP])
    S = V @ V.T
    iu = np.triu_indices(len(FP), 1)
    same = P[iu[0]] == P[iu[1]]
    obs = S[iu][same].mean() - S[iu][~same].mean()
    rng = np.random.default_rng(0); null = []
    for _ in range(2000):
        Pp = rng.permutation(P); sm = Pp[iu[0]] == Pp[iu[1]]
        null.append(S[iu][sm].mean() - S[iu][~sm].mean())
    R = {"n_seizures": len(FP), "n_patients": int(len(np.unique(P))), "skipped": skipped,
         "stereotypy": {"within": round(float(S[iu][same].mean()), 4), "between": round(float(S[iu][~same].mean()), 4),
                        "difference": round(float(obs), 4), "p_perm": float((np.sum(np.array(null) >= obs) + 1) / 2001),
                        "n_within_pairs": int(same.sum())}}
    # seizure types per patient
    types = {}
    for p in np.unique(P):
        idx = np.flatnonzero(P == p)
        lab = patterns.cluster([FP[i] for i in idx])
        for i, l in zip(idx, lab):
            FP[i]["type"] = int(l)
        types[p] = {"n_seizures": int(len(idx)), "n_types": int(len(set(lab)))}
    R["types"] = types
    # lateralization against Siena clinical labels
    calls = []
    for f in FP:
        m = meta.get(f["patient"])
        if f["cohort"] != "siena" or m is None or m["lateralization"] not in ("L", "R"):
            continue
        truth = "left" if m["lateralization"] == "L" else "right"
        calls.append({"patient": f["patient"], "truth": truth, "call": f["lateralization"]["side"],
                      "index": f["lateralization"]["index"], "localization": m["localization"]})
    lat = [c for c in calls if c["call"] != "bilateral"]
    ok = sum(c["call"] == c["truth"] for c in lat)
    R["lateralization"] = {"n_seizures": len(calls), "n_lateralized_calls": len(lat), "correct": ok,
                           "accuracy": round(ok / len(lat), 4) if lat else None,
                           "p_binom_vs_half": float(binomtest(ok, len(lat), 0.5, alternative="greater").pvalue) if lat else None,
                           "abstain_rate": round(1 - len(lat) / len(calls), 4) if calls else None, "calls": calls}
    print(f"stereotypy within {R['stereotypy']['within']:.3f} vs between {R['stereotypy']['between']:.3f} "
          f"(p {R['stereotypy']['p_perm']:.4f}); lateralization {ok}/{len(lat)} lateralized calls correct "
          f"({len(calls)} seizures)", flush=True)
    (out / "fingerprints.json").write_text(json.dumps(FP), encoding="utf-8")
    save_json(R, out / "results.json")
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
