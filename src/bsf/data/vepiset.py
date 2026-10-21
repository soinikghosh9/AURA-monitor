"""VEPISET loader — scalp-EEG interictal epileptiform discharge (IED) type/localization.

A cross-TASK test of the representation thesis: the same bifurcation-aware, physiologically-grounded
18-D vector (``data/features.py``) that drives forecasting is reused, unchanged, on a *diagnostic*
task — is an IED generalized or focal, and if focal, which lobe. VEPISET (vepiset.com) has ~73
patients, roughly twice the entire forecasting cohort, so it is the substrate on which a genuine
*cross-patient* generalization claim can be made with statistical power the forecasting side lacks.

Layout (``datasets/vepiset-dataset/``): one folder per class of 4 s, 29-channel, 500 Hz ``.npy``
clips stored as ``(M, T) = (29, 2000)`` (channel-major; we transpose to the ``(T, M)`` the feature
functions expect). Filenames encode ``<patientID>_<startSample>_<endSample>_<fs>__<class>.npy``.

Class codes (folder suffix ``__k``): 0 Non-IED, 1 Generalized, 2 Frontal, 3 Temporal,
4 Centro-Parietal, 5 Occipital. The Non-IED clips are the per-patient *interictal reference*
(the "normal" state) that the paper's interictal-referenced normalization aligns across patients.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FS = 500.0
CLASS_NAME = {0: "non_ied", 1: "generalized", 2: "frontal", 3: "temporal",
              4: "centro_parietal", 5: "occipital"}
FOCAL_CLASSES = (2, 3, 4, 5)          # everything focal (by lobe)
LOBE_CLASSES = (2, 3, 5)              # frontal/temporal/occipital (centro-parietal has only 4 patients)
# patient id is any token (DA*/DC*/...) before the numeric <start>_<end>_<fs>__<class> tail.
_NAME_RE = re.compile(r"^(?P<pid>.+?)_(?P<start>\d+)_(?P<end>\d+)_(?P<fs>\d+)__(?P<cls>\d+)\.npy$")


@dataclass
class VepClip:
    path: str
    pid: str        # patient id, e.g. "DA00102Y"
    cls: int        # 0..5
    fs: float


def discover(root: str | Path) -> list[VepClip]:
    """Enumerate every labelled clip under ``root`` (recursing the class folders)."""
    root = Path(root)
    out: list[VepClip] = []
    for p in root.rglob("*.npy"):
        m = _NAME_RE.match(p.name)
        if m:
            out.append(VepClip(str(p), m.group("pid"), int(m.group("cls")), float(m.group("fs"))))
    return out


def load_clip(path: str | Path) -> np.ndarray:
    """Load one clip as ``(T, M)`` float (transpose from the on-disk channel-major ``(M, T)``)."""
    a = np.load(path)
    a = np.asarray(a, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError(f"expected 2-D clip, got {a.shape} from {path}")
    if a.shape[0] < a.shape[1]:      # (M, T) channel-major on disk -> (T, M)
        a = a.T
    return a


def by_patient(clips: list[VepClip]) -> dict[str, list[VepClip]]:
    d: dict[str, list[VepClip]] = defaultdict(list)
    for c in clips:
        d[c.pid].append(c)
    return dict(d)


def class_patient_counts(clips: list[VepClip]) -> dict[int, int]:
    """Number of distinct patients contributing at least one clip of each class."""
    pats: dict[int, set] = defaultdict(set)
    for c in clips:
        pats[c.cls].add(c.pid)
    return {k: len(v) for k, v in sorted(pats.items())}


def extract_features(clip: np.ndarray, fs: float, kind: str, channel: int = 0) -> np.ndarray:
    """Map one ``(T, M)`` clip to a feature vector under the requested representation.

    ``kind`` in {"slow", "rich", "rich_plus"} — the *same* three representations the forecasting
    ablation uses, so the cross-task comparison is like-for-like.
    """
    from bsf.data.features import rich_features, rich_plus
    from bsf.data.windows import slow_feature
    if kind == "slow":
        return slow_feature(clip[:, channel], fs)
    if kind == "rich":
        return rich_features(clip, fs)
    if kind == "rich_plus":
        return rich_plus(clip, fs)
    raise ValueError(f"unknown representation {kind!r}")


def build_matrix(clips: list[VepClip], kind: str = "rich_plus", classes=None,
                 max_per_patient_class: int | None = None, seed: int = 1234):
    """Feature matrix ``X``, integer class labels ``y``, and patient-group array ``groups``.

    ``classes`` restricts to a subset of class codes (e.g. ``FOCAL_CLASSES``). Optionally cap the
    number of clips per (patient, class) to bound cost / rebalance. Returns ``(X, y, groups, pids)``.
    """
    rng = np.random.default_rng(seed)
    keep = clips if classes is None else [c for c in clips if c.cls in set(classes)]
    # optional per-(patient,class) cap
    if max_per_patient_class is not None:
        buckets: dict[tuple, list[VepClip]] = defaultdict(list)
        for c in keep:
            buckets[(c.pid, c.cls)].append(c)
        keep = []
        for _, lst in buckets.items():
            if len(lst) > max_per_patient_class:
                idx = rng.choice(len(lst), max_per_patient_class, replace=False)
                lst = [lst[i] for i in idx]
            keep.extend(lst)
    X, y, groups = [], [], []
    for c in keep:
        try:
            v = extract_features(load_clip(c.path), c.fs, kind)
        except Exception:
            continue
        if not np.isfinite(v).all():
            continue
        X.append(v); y.append(c.cls); groups.append(c.pid)
    return (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64),
            np.asarray(groups), sorted(set(groups)))


def patient_reference(clips: list[VepClip], kind: str = "rich_plus",
                      max_ref: int = 60, seed: int = 1234) -> dict[str, np.ndarray]:
    """Per-patient interictal reference matrix from that patient's Non-IED (class 0) clips.

    This is the "normal" state the interictal-referenced normalization aligns across patients; VEPISET
    conveniently ships Non-IED clips per patient. Returns ``{pid: (n_ref, D)}`` (may be empty for a
    patient with no Non-IED clips).
    """
    rng = np.random.default_rng(seed)
    ref: dict[str, list] = defaultdict(list)
    non = [c for c in clips if c.cls == 0]
    per: dict[str, list[VepClip]] = defaultdict(list)
    for c in non:
        per[c.pid].append(c)
    out: dict[str, np.ndarray] = {}
    for pid, lst in per.items():
        if len(lst) > max_ref:
            idx = rng.choice(len(lst), max_ref, replace=False)
            lst = [lst[i] for i in idx]
        for c in lst:
            try:
                v = extract_features(load_clip(c.path), c.fs, kind)
            except Exception:
                continue
            if np.isfinite(v).all():
                ref[pid].append(v)
        if ref[pid]:
            out[pid] = np.asarray(ref[pid], dtype=np.float32)
    return out
