"""Persistent on-disk cache of clip sub-windows and hand-crafted features.

Loading and decimating the raw clips dominates every deep run in this repo: canine records are 60-min
files (1.44M samples x 16 channels), and `within_refwhiten`, `deep_fair`, `deep_fair_null`,
`pac_window_control` and the AURA-Net sweep each re-read and re-decimate the same clips. That cost is
pure repetition -- the sub-windows are a deterministic function of (dataset, subject, extraction
params), so they only ever need computing once.

Cache entries are keyed by a hash of every parameter that affects the output, so a changed window
length or sampling rate produces a new entry rather than a silently stale hit. Sub-windows are stored
as float16 (they are per-window z-scored, so the dynamic range is ~+-5 and the precision loss is far
below the noise floor), which roughly halves the footprint.

Writes are atomic: a temp file is renamed into place, so a power cut leaves either the old entry or no
entry, never a truncated one.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

CACHE_ROOT = Path("results") / "_clip_cache"


def _key(dataset: str, subject: str, **params) -> str:
    payload = json.dumps({"dataset": str(dataset), "subject": str(subject),
                          **{k: params[k] for k in sorted(params)}}, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:20]


def cache_path(dataset: str, subject: str, **params) -> Path:
    return CACHE_ROOT / f"{Path(str(dataset)).name}_{subject}_{_key(dataset, subject, **params)}.npz"


def load(dataset: str, subject: str, **params):
    """Return the cached dict, or None on a miss/corrupt entry."""
    p = cache_path(dataset, subject, **params)
    if not p.exists():
        return None
    try:
        with np.load(p, allow_pickle=False) as z:
            out = {k: z[k] for k in z.files}
        if "W" in out:
            out["W"] = out["W"].astype(np.float32)      # stored f16, used f32
        return out
    except Exception:                                    # truncated/corrupt -> treat as a miss
        try:
            p.unlink()
        except OSError:
            pass
        return None


def save(dataset: str, subject: str, data: dict, **params) -> Path:
    """Atomically write a cache entry. ``data['W']`` is downcast to float16 for storage."""
    p = cache_path(dataset, subject, **params)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(data)
    if "W" in payload:
        payload["W"] = np.asarray(payload["W"], dtype=np.float16)
    tmp = p.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, p)
    return p


def build_or_load(dataset: str, subject: str, files: dict, *, n_subwin: int = 12, win_s: float = 8.0,
                  target_fs: float = 400.0, max_per_class: int = 60, with_features: bool = True,
                  verbose: bool = True):
    """Sub-windows (+ optional hand-crafted features) for one subject, cached across runs.

    Returns a dict with ``W`` (n_clips, n_subwin, M, L), ``y``, ``groups`` (seizure-hour ids), and,
    when ``with_features``, ``Xslow`` / ``Xrich`` / ``Xplus`` / ``Xsub`` matching the arms the
    benchmarks compare. ``groups`` collapses the provider's consecutive within-hour clip sequences, so
    grouped CV can keep one pre-seizure hour inside a single fold.
    """
    # v is a MANUAL cache epoch: bump it whenever the extraction CODE changes in a way the parameter
    # hash cannot see. v=4 corresponds to subwindows() resampling to exactly target_fs. Without this
    # bump the earlier fix was silently bypassed by cache hits and never reached a single run -- the
    # subjects at 5000 Hz kept their 416.7 Hz windows, and with them a 4% error in every band edge,
    # while the code and the commit message both claimed otherwise.
    params = dict(n_subwin=n_subwin, win_s=win_s, target_fs=target_fs,
                  max_per_class=max_per_class, with_features=with_features, v=4)
    hit = load(dataset, subject, **params)
    if hit is not None:
        if verbose:
            print(f"    (cache) {subject}: {hit['W'].shape[0]} clips", flush=True)
        return hit

    from bsf.data.features import rich_features, rich_plus
    from bsf.data.kaggle_seizure import load_kaggle_segment
    from bsf.data.windows import slow_feature
    from bsf.dynamotype.labeling import select_seizure_channel
    from bsf.models.auranet_train import subwindows

    # Segments that physically overlap must share a CV group. The canine fetcher draws interictal
    # windows at random start times and rejects a draw only for being near a SEIZURE, never for being
    # near another draw, so a few windows share samples; one preictal window is cut per seizure, so
    # two seizures under an hour apart do the same. Given their own groups they land in different
    # folds and the model is scored on data it partly trained on. scripts/audit_ieeg_overlap.py finds
    # the components; absent that file, nothing changes.
    overlap: dict[str, int] = {}
    try:
        _ov = json.loads((CACHE_ROOT.parent / "_ieeg_overlap.json").read_text())
        for k, v in _ov.get("subjects", {}).items():
            if k.endswith(f"/{subject}"):
                overlap = v.get("components", {})
                break
    except Exception:  # noqa: BLE001
        pass

    W, y, groups = [], [], []
    Xslow, Xrich, Xplus, Xsub = [], [], [], []
    gid = 0
    ov_gid: dict[int, int] = {}
    for lname, lab in (("preictal", 1), ("interictal", 0)):
        prev_seq = None
        for f in files.get(lname, [])[:max_per_class]:
            seg = load_kaggle_segment(f)
            if seg.data.ndim != 2 or seg.data.shape[0] < 120:
                continue
            sw = subwindows(seg.data, seg.fs, n=n_subwin, win_s=win_s, target_fs=target_fs)
            if sw is None:
                continue
            W.append(sw); y.append(lab)
            if with_features:
                ch = select_seizure_channel(seg.data, 0, seg.data.shape[0])
                Xslow.append(slow_feature(seg.data[:, ch], seg.fs))
                Xrich.append(rich_features(seg.data, seg.fs))
                Xplus.append(rich_plus(seg.data, seg.fs))
                L = int(round(win_s * seg.fs))
                starts = np.linspace(0, max(0, seg.data.shape[0] - L), n_subwin).astype(int)
                Xsub.append(np.mean([rich_plus(seg.data[s:s + L], seg.fs) for s in starts], 0))
            # a new group per clip, except that (a) a provider's consecutive within-hour sequence
            # stays together so grouped CV cannot split one pre-seizure hour, and (b) segments found
            # to share samples stay together so a near-duplicate cannot straddle folds
            comp = overlap.get(Path(f).name)
            if comp is not None and comp in ov_gid:
                groups.append(ov_gid[comp])
            elif lname == "preictal" and seg.sequence is not None and prev_seq is not None \
                    and seg.sequence > prev_seq:
                groups.append(gid)
            else:
                gid += 1
                groups.append(gid)
                if comp is not None:
                    ov_gid[comp] = gid
            prev_seq = seg.sequence if lname == "preictal" else None

    if not W:
        return None
    n_ch = min(w.shape[1] for w in W)
    out = {"W": np.stack([w[:, :n_ch] for w in W]).astype(np.float32),
           "y": np.asarray(y, np.int64), "groups": np.asarray(groups, np.int64)}
    if with_features:
        out.update({"Xslow": np.nan_to_num(np.asarray(Xslow, np.float32)),
                    "Xrich": np.nan_to_num(np.asarray(Xrich, np.float32)),
                    "Xplus": np.nan_to_num(np.asarray(Xplus, np.float32)),
                    "Xsub": np.nan_to_num(np.asarray(Xsub, np.float32))})
    save(dataset, subject, out, **params)
    if verbose:
        mb = out["W"].nbytes / 2 / 2**20
        print(f"    (built) {subject}: {out['W'].shape[0]} clips, {n_ch} ch -> cached {mb:.0f} MB",
              flush=True)
    return out
