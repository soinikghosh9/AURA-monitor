"""Loader for the Kaggle seizure-prediction `.mat` clips (AES 2014 + Melbourne/NeuroVista 2016).

Each file (e.g. ``Dog_1_preictal_segment_0001.mat``) holds one MATLAB struct with fields:
``data`` (channels × samples), ``sampling_frequency``, ``channels``, and ``sequence`` (which 10-min
slot within the pre-seizure hour, for preictal clips). The label (preictal / interictal / test) and
subject are parsed from the filename. Arrays are returned samples-major (T, M).

These public clips give **multiple human subjects + canine** (cross-species) already split into
preictal vs interictal — the substrate for Phase-5 multi-subject and Phase-6 cross-species evals.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

_NAME_RE = re.compile(r"(?P<subject>.+?)_(?P<label>preictal|interictal|test)_segment_(?P<idx>\d+)\.mat$")


@dataclass
class KaggleSegment:
    subject: str
    label: str  # preictal | interictal | test
    data: np.ndarray  # (T, M) samples x channels
    fs: float
    sequence: int | None  # 1..6 slot in the pre-seizure hour (preictal only)
    source: str


def _struct_field(struct, name):
    if struct.dtype.names is None or name not in struct.dtype.names:
        return None
    return struct[name][0, 0]


def load_kaggle_segment(path: str | Path) -> KaggleSegment:
    path = Path(path)
    m = _NAME_RE.search(path.name)
    subject = m.group("subject") if m else path.stem
    label = m.group("label") if m else "unknown"

    mat = loadmat(path)
    key = next(k for k in mat if not k.startswith("__"))
    struct = mat[key]
    data = np.asarray(_struct_field(struct, "data"), dtype=float)
    if data.shape[0] < data.shape[1]:  # channels << samples -> make samples-major
        data = data.T
    fs_field = _struct_field(struct, "sampling_frequency")
    fs = float(np.squeeze(fs_field))
    seq_field = _struct_field(struct, "sequence")
    sequence = int(np.squeeze(seq_field)) if seq_field is not None and np.size(seq_field) else None
    return KaggleSegment(subject, label, data, fs, sequence, str(path))


def discover_kaggle(root: str | Path) -> dict[str, dict[str, list[Path]]]:
    """Map subject -> {preictal: [...], interictal: [...], test: [...]} of .mat segment files."""
    root = Path(root)
    out: dict[str, dict[str, list[Path]]] = {}
    for p in sorted(root.rglob("*.mat")):
        m = _NAME_RE.search(p.name)
        if not m:
            continue
        out.setdefault(m.group("subject"), {}).setdefault(m.group("label"), []).append(p)
    return out


def kaggle_features(root: str | Path, subject: str, target_len: int = 120, band=(1.0, 45.0),
                    max_per_class: int | None = None, return_groups: bool = False):
    """Build (X, y[, groups]) detector-ready slow features for one subject: y=1 preictal, y=0 interictal.

    With ``return_groups=True`` also returns an event-id array: the (up to 6) preictal clips of ONE
    pre-seizure hour — which the ``sequence`` field marks as slots 1..6 and which are near-duplicates —
    share a group id, and each interictal clip is its own group. This lets the evaluation count
    independent seizure-HOURS rather than pseudo-replicated clips (event-level CIs; fixes E7).
    """
    from bsf.data.windows import slow_feature
    from bsf.dynamotype.labeling import select_seizure_channel

    found = discover_kaggle(root).get(subject, {})
    X: list[np.ndarray] = []
    y: list[int] = []
    groups: list[int] = []
    gid = 0
    for label, lab_val in (("preictal", 1), ("interictal", 0)):
        files = found.get(label, [])
        if max_per_class:
            files = files[:max_per_class]
        prev_seq = None
        for f in files:
            seg = load_kaggle_segment(f)
            ch = select_seizure_channel(seg.data, 0, seg.data.shape[0]) if seg.data.ndim == 2 else 0
            x = seg.data[:, ch] if seg.data.ndim == 2 else seg.data
            X.append(slow_feature(x, seg.fs, target_len=target_len, band=band))
            y.append(lab_val)
            if label == "preictal" and seg.sequence is not None and prev_seq is not None \
                    and seg.sequence > prev_seq:
                groups.append(gid)  # same pre-seizure hour (sequence still ascending 1->6)
            else:
                gid += 1  # new hour (sequence reset/absent) or an independent interictal clip
                groups.append(gid)
            prev_seq = seg.sequence if label == "preictal" else None
    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.int64)
    g_arr = np.asarray(groups, dtype=np.int64)
    finite = np.isfinite(X_arr).all(axis=1) if len(X_arr) else np.array([], dtype=bool)
    if return_groups:
        return X_arr[finite], y_arr[finite], g_arr[finite]
    return X_arr[finite], y_arr[finite]
