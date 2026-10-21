#!/usr/bin/env python
"""Stream the NeuroVista canine continuous iEEG (ieeg.org) into Kaggle-format windows.

The full records are huge (~750 GB *per dog*, 343 days x 16 ch x 400 Hz), so we do NOT bulk
download. Instead we stream only the windows that matter — the preictal window before each marked
seizure + sampled interictal windows — and save each as a Kaggle-style `.mat`
(`<Dog>_preictal_segment_NNNN.mat` / `<Dog>_interictal_segment_NNNN.mat`) so they drop straight
into the existing `phase6` loader (src/bsf/data/kaggle_seizure.py) with no new code.

Resumable: windows already saved are skipped, so a power cut / Ctrl-C just continues. Credentials
are read from scripts/ieeg_login.json (gitignored) or the IEEG_USER / IEEG_PW env vars.

Usage:
    python scripts/fetch_ieeg_canine.py --datasets I004_A0003_D001 --interictal 50 --limit-seizures 5
    python scripts/fetch_ieeg_canine.py            # all 9 dogs, default counts
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.io import savemat

REPO = Path(__file__).resolve().parent.parent
DEST = REPO / "datasets" / "ieeg_canine"

# ieeg.org dataset id -> Kaggle-style subject label
CANINE = {
    "I004_A0001_D001": "Dog_A0001",
    "I004_A0002_D001": "Dog_A0002a",
    "I004_A0002_D002": "Dog_A0002b",
    "I004_A0003_D001": "Dog_A0003",
    "I004_A0004_D001": "Dog_A0004",
    "I004_A0006_D001": "Dog_A0006",
    "I004_A0007_D002": "Dog_A0007",
    "I004_A0008_D002": "Dog_A0008",
    "I004_A0009_D002": "Dog_A0009",
}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def credentials() -> tuple[str, str]:
    if os.environ.get("IEEG_USER") and os.environ.get("IEEG_PW"):
        return os.environ["IEEG_USER"], os.environ["IEEG_PW"]
    cfg = REPO / "scripts" / "ieeg_login.json"
    if cfg.exists():
        d = json.loads(cfg.read_text())
        return d["username"], d["password"]
    raise SystemExit("No ieeg.org credentials (set IEEG_USER/IEEG_PW or create scripts/ieeg_login.json).")


def fetch_window(ds, start_s: float, dur_s: float, n_ch: int, fs: float, chunk_s: float = 120.0) -> np.ndarray:
    """Stream [start_s, start_s+dur_s) for all channels, in chunks (the portal caps request size)."""
    chans = list(range(n_ch))
    parts = []
    t = start_s
    end = start_s + dur_s
    while t < end - 1e-6:
        c = min(chunk_s, end - t)
        block = ds.get_data(int(t * 1e6), int(c * 1e6), chans)  # (samples, channels)
        parts.append(np.asarray(block, dtype=np.float32))
        t += c
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, n_ch), np.float32)


def save_segment(path: Path, data_samples_major: np.ndarray, fs: float, labels, sequence=None) -> None:
    """Save one window in the Kaggle struct format the bsf loader expects."""
    seg = {
        "data": data_samples_major.T.astype(np.float32),           # (channels, samples)
        "sampling_frequency": float(fs),
        "channels": np.array(labels, dtype=object),
        "sequence": np.array([[sequence if sequence is not None else 0]]),
    }
    savemat(str(path), {"segment": seg}, do_compression=True)


def process_dataset(session, ds_id: str, subject: str, args) -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    ds = session.open_dataset(ds_id)
    labels = ds.get_channel_labels()
    n_ch = len(labels)
    det = ds.get_time_series_details(labels[0])
    fs = float(det.sample_rate)
    total_s = det.number_of_samples / fs
    layers = ds.get_annotation_layers()
    sz_layer = next((k for k in layers if "seiz" in k.lower()), None)
    seiz = sorted(a.start_time_offset_usec / 1e6 for a in ds.get_annotations(sz_layer)) if sz_layer else []
    log(f"{subject} ({ds_id}): {n_ch}ch {fs:.1f}Hz {total_s/86400:.0f}d, {len(seiz)} seizures")
    if not seiz:
        log(f"  {subject}: 0 seizures -> SKIP (interictal-only is not usable for forecasting AUC)")
        return

    pre_len, sph, guard = args.preictal_min * 60.0, args.sph_min * 60.0, args.interictal_guard_h * 3600.0
    win = args.win_min * 60.0

    # ---- preictal: window ending sph before each seizure onset ----
    sez = seiz[: args.limit_seizures] if args.limit_seizures else seiz
    for i, onset in enumerate(sez, 1):
        start = onset - sph - pre_len
        if start < 0:
            continue
        out = DEST / f"{subject}_preictal_segment_{i:04d}.mat"
        if out.exists():
            continue
        try:
            data = fetch_window(ds, start, pre_len, n_ch, fs)
            save_segment(out, data, fs, labels, sequence=1)
            log(f"  preictal {i}/{len(sez)} -> {out.name} ({data.shape})")
        except Exception as e:  # noqa: BLE001
            log(f"  preictal {i} FAILED: {str(e)[:80]}")

    # ---- interictal: windows >= guard from every seizure, evenly spaced across the record ----
    rng = np.random.default_rng(args.seed)
    want = args.interictal
    made = len(list(DEST.glob(f"{subject}_interictal_segment_*.mat")))
    tries = 0
    while made < want and tries < want * 20:
        tries += 1
        start = float(rng.uniform(0, max(1.0, total_s - win)))
        if any(abs(start + win / 2 - z) < guard for z in seiz):
            continue
        idx = made + 1
        out = DEST / f"{subject}_interictal_segment_{idx:04d}.mat"
        if out.exists():
            made += 1
            continue
        try:
            data = fetch_window(ds, start, win, n_ch, fs)
            save_segment(out, data, fs, labels, sequence=None)
            made += 1
            if made % 10 == 0 or made == want:
                log(f"  interictal {made}/{want}")
        except Exception as e:  # noqa: BLE001
            log(f"  interictal try {tries} FAILED: {str(e)[:80]}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default="all", help="'all' or comma list of ieeg ids / Dog labels")
    ap.add_argument("--preictal-min", type=float, default=60.0, help="preictal window length (min)")
    ap.add_argument("--sph-min", type=float, default=5.0, help="seizure prediction horizon gap (min)")
    ap.add_argument("--win-min", type=float, default=10.0, help="interictal window length (min)")
    ap.add_argument("--interictal", type=int, default=100, help="interictal windows per dog")
    ap.add_argument("--interictal-guard-h", type=float, default=4.0, help="min hours from any seizure")
    ap.add_argument("--limit-seizures", type=int, default=0, help="cap preictal windows (0=all; for testing)")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args(argv)

    from ieeg.auth import Session
    user, pw = credentials()
    log(f"using ieeg.org account {user}")

    if args.datasets == "all":
        items = list(CANINE.items())
    else:
        toks = {t.strip() for t in args.datasets.split(",")}
        items = [(i, s) for i, s in CANINE.items() if i in toks or s in toks]
    log(f"streaming {len(items)} dataset(s) -> {DEST}")

    for ds_id, subject in items:
        try:
            sess = Session(user, pw)  # fresh session per dog: bounds lifetime, avoids token expiry on long runs
            process_dataset(sess, ds_id, subject, args)
        except Exception as e:  # noqa: BLE001
            log(f"{subject} ({ds_id}) FAILED: {str(e)[:120]}")
    log("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
