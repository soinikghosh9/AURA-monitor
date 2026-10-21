#!/usr/bin/env python
"""Are any ieeg.org NeuroVista dogs (Dog_A000x) the same animals as the Kaggle AES dogs (Dog_1..Dog_5)?

Both come from the NeuroVista canine program (same device, 16 channels at 399.61 Hz), but ieeg.org renames the
channels, so identity cannot be read from metadata. Two signal-level checks:

  exact   sign patterns of 64 successive sample differences of channel 1 (scale- and offset-invariant), hashed at every offset in a sample of ieeg.org clips, probed at a stride
          of 64 in Kaggle clips: any hit means the two files contain the same samples
  spectral per-channel log power spectrum (1-100 Hz) averaged over interictal clips; same animal and electrodes give
          a channel-by-channel profile far closer than any other pair

  python scripts/check_canine_identity.py   -> results/canine_identity_<stamp>/results.json
"""
from __future__ import annotations

import glob
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
KAGGLE = [f"Dog_{i}" for i in range(1, 6)]
IEEG = sorted({Path(f).name.split("_inter")[0] for f in glob.glob(str(DATA / "ieeg_canine" / "Dog_A*_interictal_*.mat"))})
N_CLIPS = 30
RNG = np.random.default_rng(0)


def load(f):
    m = sio.loadmat(f)
    s = m[[k for k in m if not k.startswith("__")][0]][0, 0]
    x = np.asarray(s["data"], np.float64)
    return np.nan_to_num(x), float(np.squeeze(s["sampling_frequency"]))


def clips(prefix, folder):
    fs = sorted(glob.glob(str(DATA / folder / f"{prefix}_interictal_segment_*.mat")))
    return [fs[i] for i in sorted(RNG.choice(len(fs), min(N_CLIPS, len(fs)), replace=False))]


def profile(files):
    P = []
    for f in files:
        x, fs = load(f)
        fr, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=1)
        m = (fr >= 1) & (fr <= 100)
        P.append(np.log10(p[:, m] + 1e-12))
    return np.mean(P, 0)          # (channels, freqs)


def main() -> int:
    out = ROOT / "results" / f"canine_identity_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out.mkdir(parents=True)
    kf = {d: clips(d, "kaggle_aes") for d in KAGGLE}
    jf = {d: clips(d, "ieeg_canine") for d in IEEG}
    R = {"ieeg_dogs": IEEG, "kaggle_dogs": KAGGLE, "n_clips_each": N_CLIPS, "exact_hits": {}, "spectral_corr": {}}

    # exact: hash every 64-sample window (all offsets) of channel 1 in the ieeg.org clips
    W = 64
    for j, files in jf.items():
        H = set()
        for f in files[:10]:
            x, _ = load(f)
            v = np.diff(x[0]) > 0            # sign pattern: invariant to scale and offset between the two sources
            win = np.lib.stride_tricks.sliding_window_view(v, W)
            H.update(hash(r.tobytes()) for r in win if 8 < r.sum() < W - 8)
        for k, kfiles in kf.items():
            hits = 0
            for f in kfiles:
                x, _ = load(f)
                v = np.diff(x[0]) > 0
                win = np.lib.stride_tricks.sliding_window_view(v, W)[::W]
                hits += sum(hash(r.tobytes()) in H for r in win if 8 < r.sum() < W - 8)
            R["exact_hits"][f"{j}~{k}"] = int(hits)
        print("exact", j, {k: R["exact_hits"][f"{j}~{k}"] for k in kf}, flush=True)

    # spectral: channel-aligned correlation of the per-channel log spectra (16 channels; Dog_5 has 15)
    PK = {k: profile(v) for k, v in kf.items()}
    PJ = {j: profile(v) for j, v in jf.items()}
    allp = {**{f"kaggle:{k}": v for k, v in PK.items()}, **{f"ieeg:{j}": v for j, v in PJ.items()}}
    names = list(allp)
    c = min(len(v) for v in allp.values())
    grand = np.mean([v[:c] for v in allp.values()], 0)     # shared 1/f shape, removed before correlating
    resid = {n: (v[:c] - grand).ravel() for n, v in allp.items()}
    M = np.array([[np.corrcoef(resid[a], resid[b])[0, 1] for b in names] for a in names])
    R["spectral_corr"] = {"names": names, "matrix": np.round(M, 4).tolist()}
    for j in PJ:
        row = {k: round(float(M[names.index(f"ieeg:{j}"), names.index(f"kaggle:{k}")]), 4) for k in PK}
        print("spectral", j, row, flush=True)
    (out / "results.json").write_text(json.dumps(R, indent=1))
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
