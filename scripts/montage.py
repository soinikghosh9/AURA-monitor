"""A common scalp montage for CHB-MIT and Siena: the 18-channel longitudinal bipolar double banana.

The native extraction kept each recording's channels in file order and truncated to the common count,
so a raw-signal model saw a different electrode at the same index in different patients: CHB-MIT
inserts dummy channels ('--0') from chb11 on, and Siena is recorded referentially with the old
temporal names. ``db18`` maps every recording to the same 18 bipolar derivations by name -- taken
directly when the recording is bipolar (CHB-MIT), computed as a difference of the two referential
channels otherwise (Siena) -- so channel k is the same derivation for every patient.
"""
from __future__ import annotations

import re

import numpy as np

DB18 = [("FP1", "F7"), ("F7", "T7"), ("T7", "P7"), ("P7", "O1"),
        ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
        ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
        ("FP2", "F8"), ("F8", "T8"), ("T8", "P8"), ("P8", "O2"),
        ("FZ", "CZ"), ("CZ", "PZ")]
ALIAS = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def _norm(name: str) -> str:
    n = name.upper().replace("EEG ", "").replace("-REF", "").replace("-LE", "").strip()
    n = re.sub(r"-\d+$", "", n) if n.count("-") >= 2 else n       # 'T8-P8-0' -> 'T8-P8'
    return "-".join(ALIAS.get(p, p) for p in n.split("-"))


def db18(data: np.ndarray, ch_names: list[str]) -> np.ndarray | None:
    """(n_ch, n_samples) in file order -> (18, n_samples) double banana, or None if a derivation is
    unavailable."""
    names = [_norm(c) for c in ch_names]
    bip = {n: i for i, n in reversed(list(enumerate(names))) if "-" in n}
    ref = {n: i for i, n in reversed(list(enumerate(names))) if "-" not in n}
    out = []
    for a, b in DB18:
        if f"{a}-{b}" in bip:
            out.append(data[bip[f"{a}-{b}"]])
        elif f"{b}-{a}" in bip:
            out.append(-data[bip[f"{b}-{a}"]])
        elif a in ref and b in ref:
            out.append(data[ref[a]] - data[ref[b]])
        else:
            return None
    return np.stack(out)
