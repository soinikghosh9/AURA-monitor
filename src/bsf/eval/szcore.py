"""Event-level seizure-detection scoring in the SzCORE convention, on windowed continuous predictions.

Reference events are runs of ictal windows; hypothesis events are runs of windows at or above a
threshold. Both are merged when closer than ``merge_s`` and split into chunks of at most ``max_s``. A
reference event counts as detected when any hypothesis event overlaps it after extending it by
``tol_pre_s`` before onset and ``tol_post_s`` after offset; a hypothesis event is a false alarm when it
overlaps no extended reference event. Defaults are the SzCORE values (30 s, 60 s, 90 s, 300 s).
Runs never bridge a recording gap (consecutive windows further apart than 1.5 window lengths).
"""
from __future__ import annotations

import numpy as np

SZCORE = {"tol_pre_s": 30.0, "tol_post_s": 60.0, "merge_s": 90.0, "max_s": 300.0}


def _runs(mask: np.ndarray, t: np.ndarray, win_s: float) -> list[tuple[float, float]]:
    out, start, prev = [], None, None
    for i in np.flatnonzero(mask):
        if start is not None and t[i] - t[prev] <= 1.5 * win_s:
            prev = i
            continue
        if start is not None:
            out.append((t[start], t[prev] + win_s))
        start = prev = i
    if start is not None:
        out.append((t[start], t[prev] + win_s))
    return out


def _merge_split(ev: list[tuple[float, float]], merge_s: float, max_s: float) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for a, b in sorted(ev):
        if merged and a - merged[-1][1] < merge_s:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    out = []
    for a, b in merged:
        while b - a > max_s:
            out.append((a, a + max_s))
            a += max_s
        out.append((a, b))
    return out


def events(y: np.ndarray, p: np.ndarray, t: np.ndarray, thr: float, win_s: float = 8.0,
           **kw) -> dict:
    """Counts for one stream at one threshold."""
    c = {**SZCORE, **kw}
    ref = _merge_split(_runs(np.asarray(y) > 0, t, win_s), c["merge_s"], c["max_s"])
    hyp = _merge_split(_runs(np.asarray(p) >= thr, t, win_s), c["merge_s"], c["max_s"])
    ext = [(a - c["tol_pre_s"], b + c["tol_post_s"]) for a, b in ref]
    ov = lambda e, f: e[0] < f[1] and f[0] < e[1]  # noqa: E731
    tp_ref = sum(any(ov(r, h) for h in hyp) for r in ext)
    tp_hyp = sum(any(ov(h, r) for r in ext) for h in hyp)
    lat = []
    for (a, _b), r in zip(ref, ext):
        hits = [h[0] for h in hyp if ov(r, h)]
        if hits:
            lat.append(max(0.0, min(hits) - a))
    return {"n_ref": len(ref), "tp_ref": tp_ref, "n_hyp": len(hyp), "tp_hyp": tp_hyp,
            "fp": len(hyp) - tp_hyp, "hours": float(len(y) * win_s / 3600.0), "latency_s": lat}


def summarize(counts: list[dict]) -> dict:
    n_ref = sum(c["n_ref"] for c in counts); tp_ref = sum(c["tp_ref"] for c in counts)
    n_hyp = sum(c["n_hyp"] for c in counts); tp_hyp = sum(c["tp_hyp"] for c in counts)
    fp = sum(c["fp"] for c in counts); hours = sum(c["hours"] for c in counts)
    sens = tp_ref / n_ref if n_ref else float("nan")
    prec = tp_hyp / n_hyp if n_hyp else float("nan")
    f1 = 2 * sens * prec / (sens + prec) if n_hyp and n_ref and (sens + prec) > 0 else 0.0
    lat = [x for c in counts for x in c["latency_s"]]
    return {"sensitivity": round(sens, 4), "precision": round(prec, 4) if n_hyp else None,
            "f1": round(f1, 4), "fa_per_24h": round(24 * fp / hours, 2) if hours else None,
            "n_ref": n_ref, "detected": tp_ref, "n_alarms": n_hyp,
            "median_latency_s": round(float(np.median(lat)), 1) if lat else None}


def threshold_for_budget(streams: list[tuple[np.ndarray, np.ndarray, np.ndarray]], budget_per_24h: float,
                         win_s: float = 8.0, n_grid: int = 120) -> float:
    """Lowest threshold whose pooled false-alarm rate on ``streams`` [(y, p, t), ...] is within budget."""
    allp = np.concatenate([p for _y, p, _t in streams])
    grid = np.unique(np.quantile(allp, np.linspace(0.5, 0.9999, n_grid)))

    def ok(thr):
        s = summarize([events(y, p, t, thr, win_s) for y, p, t in streams])
        return s["fa_per_24h"] is not None and s["fa_per_24h"] <= budget_per_24h
    lo, hi = 0, len(grid)                     # false alarms fall as the threshold rises: bisect
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(grid[mid]):
            hi = mid
        else:
            lo = mid + 1
    return float(grid[lo]) if lo < len(grid) else float(allp.max() + 1e-9)
