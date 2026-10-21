"""Data layer for the benchmark VARIANTS: the pooled cohorts plus what the pooled loaders discard.

``run_unified_benchmark`` pools each task's clips across subjects and keeps only the subject id. Three
of the questions the variant program asks need more than that:

  * **protocol**: a within-patient, event-grouped split needs each clip's seizure/hour group, so that
    one pre-seizure hour never straddles a fold while patients are still shared between folds;
  * **patient alignment**: a reference state per patient, either every clip of that patient
    (label-free, transductive) or a held-out baseline of that patient's negative clips;
  * **typing baselines**: a VEPISET patient's baseline is its Non-IED background, which the typing
    pool (IED clips only) never loads.

Everything here reads the same clip caches as the benchmark, in the same order, through the same
per-clip standardisation (``_pool``), so a variant run with no variant switched on reproduces the
benchmark's pool exactly.
"""

from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_unified_benchmark import _load_typing, _pool, dev_subjects  # noqa: E402

SFREQ = {"ieeg": 400.0, "scalp": 250.0, "detect": 250.0, "typing": 500.0}
SCALP_PARAMS = dict(n_subwin=6, win_s=8.0, target_fs=250.0, max_per_class=60, with_features=True,
                    v=5, cohort="scalp_raw")
DETECT_PARAMS = dict(n_subwin=6, win_s=8.0, target_fs=250.0, max_per_class=60, with_features=True,
                     v=5, cohort="scalp_det", task="detection", buffer_s=120.0, det_v=2)


def load_task(task: str, side: str = "dev", max_per_class: int = 60, n_subwin: int = 6,
              max_per_subject: int = 40, feat_key: str = "Xplus", montage: str = "native"):
    """(W, y, subject, event, X) for one task and one side of the sealed split.

    ``event`` is a within-patient group label, unique across patients: a seizure's pre-ictal hour, an
    interictal clip, or (typing) a single discharge. For typing every clip is its own event, so an
    event-grouped split there degenerates to a clip-random one -- which is the point: a subject's
    syndrome is fixed, so any split that shares patients between folds leaks the label through the
    patient.

    ``feat_key='Xsub'`` returns the 18-D vector averaged over the SAME sub-windows the deep models
    see, instead of computed over the whole clip ('Xplus'). On 10-min iEEG clips the deep models see
    48 s while 'Xplus' summarises all 600 s; 'Xsub' removes that information asymmetry. Typing clips
    are a single 4-s window, so there the two are the same thing.
    """
    from bsf.data.clip_cache import build_or_load, load
    if task == "typing":
        W, y, g, X = _load_typing(max_per_subject, side)
        ev = np.array([f"{s}|{i}" for i, s in enumerate(g)])
        return W, y, np.asarray(g).astype(str), ev, X
    Ws, ys, gs, Xs, evs = [], [], [], [], []
    if task == "ieeg":
        from bsf.data.kaggle_seizure import discover_kaggle
        for root, cohort in (("datasets/kaggle_aes", "kaggle_ieeg"),
                             ("datasets/ieeg_canine", "canine_ieeg")):
            keep = dev_subjects(cohort, side)
            disc = discover_kaggle(root)
            for subj in sorted(disc):
                if subj not in keep:
                    continue
                c = build_or_load(root, subj, disc[subj], n_subwin=n_subwin,
                                  max_per_class=max_per_class, verbose=False)
                if c is None or len(np.unique(c["y"])) < 2:
                    continue
                sid = f"{cohort}_{subj}"
                Ws.append(c["W"]); ys.append(c["y"]); Xs.append(c[feat_key])
                gs.append(np.array([sid] * len(c["y"])))
                evs.append(np.array([f"{sid}|{e}" for e in c["groups"]]))
    else:
        ds, params = (("scalp_raw", SCALP_PARAMS) if task == "scalp" else ("scalp_det", DETECT_PARAMS))
        params = {**params, "n_subwin": n_subwin, "max_per_class": max_per_class}
        if montage != "native":          # scalp tasks only: the 18-channel double banana (scripts/montage.py)
            params["montage"] = montage
        for coh, tag in (("chbmit", "scalp_chbmit"), ("siena", "scalp_siena")):
            for s in sorted(dev_subjects(tag, side)):
                c = load(ds, f"{coh}_{s}", **params)
                if c is None or len(np.unique(c["y"])) < 2:
                    continue
                sid = f"{coh}_{s}"
                Ws.append(c["W"]); ys.append(c["y"]); Xs.append(c[feat_key])
                gs.append(np.array([sid] * len(c["y"])))
                evs.append(np.array([f"{sid}|{e}" for e in c["groups"]]))
    pooled = _pool(Ws, ys, gs, Xs)
    if pooled is None:
        return None
    W, y, g, X = pooled
    return W, y, np.asarray(g).astype(str), np.concatenate(evs), X


def baseline_mask(y: np.ndarray, g: np.ndarray, frac: float = 1 / 3, min_n: int = 5,
                  seed: int = 1234) -> np.ndarray:
    """Per patient, a fixed random third of its NEGATIVE clips, reserved as that patient's baseline.

    Reserved clips are used only to estimate the patient's reference statistics and are removed from
    training and scoring in every arm of an alignment comparison, so aligned and unaligned models are
    scored on identical clips. The draw is seeded per patient by name, so it does not depend on which
    other patients are in the pool.
    """
    m = np.zeros(len(y), bool)
    for s in np.unique(g):
        neg = np.where((g == s) & (y == 0))[0]
        k = max(min_n, int(round(frac * len(neg))))
        if len(neg) - k < min_n:            # too few negatives to spare a baseline
            continue
        rng = np.random.default_rng(seed + zlib.crc32(s.encode()) % 100000)
        m[rng.choice(neg, k, replace=False)] = True
    return m


def typing_baseline(side: str = "dev", per_subject: int = 40, n_jobs: int = 8,
                    n_samp: int = 2000):
    """Non-IED background clips per typing patient: raw windows + 18-D features, cached.

    The same per-clip z-scoring as the typing pool, and features computed from the z-scored clip, as
    the typing pool's features are, so reference and scored clips live in one space.
    """
    from bsf.data.clip_cache import load as cload, save as csave
    from bsf.data.vepiset import by_patient, discover, load_clip
    cp = dict(per_subject=per_subject, side=side, v=2, n_samp=n_samp, cohort="vepiset_nonied")
    hit = cload("vepiset_nonied", "pooled", **cp)
    if hit is not None:
        return hit["Wf"], hit["groups"].astype(str), hit["Xplus"]
    keep = dev_subjects("vepiset_typing", side)
    rng = np.random.default_rng(4321)
    arrs, gs = [], []
    for pid, clips in sorted(by_patient(discover("datasets/vepiset-dataset")).items()):
        if pid not in keep:
            continue
        non = [c for c in clips if c.cls == 0]
        if not non:
            continue
        if len(non) > per_subject:
            non = [non[i] for i in rng.choice(len(non), per_subject, replace=False)]
        for c in non:
            a = np.nan_to_num(np.asarray(load_clip(c.path), dtype=np.float32))
            if a.ndim != 2:
                continue
            if a.shape[0] < a.shape[1]:
                a = a.T
            if a.shape[0] < n_samp:          # a few background clips are shorter than a discharge
                continue                      # window; truncating everything to them cut all to 1 s
            arrs.append(a[:n_samp].T); gs.append(pid)
    n_ch = min(w.shape[0] for w in arrs); n_t = n_samp
    Wa = np.stack([w[:n_ch, :n_t] for w in arrs]).astype(np.float32)
    Wa = np.nan_to_num((Wa - Wa.mean(-1, keepdims=True)) / (Wa.std(-1, keepdims=True) + 1e-6))
    from joblib import Parallel, delayed

    from bsf.data.features import rich_plus
    X = np.stack(Parallel(n_jobs=n_jobs)(delayed(rich_plus)(Wa[i].T, 500.0) for i in range(len(Wa))))
    X = np.nan_to_num(X).astype(np.float32)
    Wa = Wa[:, None]
    g = np.asarray(gs).astype(str)
    csave("vepiset_nonied", "pooled", {"Wf": Wa, "groups": g, "Xplus": X}, **cp)
    return Wa, g, X


# --------------------------------------------------------------------------- patient alignment
def _inv_sqrt(C: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    w, V = np.linalg.eigh((C + C.T) / 2)
    w = np.maximum(w, floor * max(w.max(), 1e-12))
    return (V / np.sqrt(w)) @ V.T


def euclidean_align(W: np.ndarray, g: np.ndarray, ref: dict[str, np.ndarray]) -> np.ndarray:
    """Euclidean alignment (He & Wu 2019): whiten each patient's spatial covariance to identity.

    ``ref[s]`` is an array (n, n_sub, C, T) of that patient's reference windows. The patient's mean
    spatial covariance R_s is estimated from them and every window of the patient is mapped by
    R_s^{-1/2}. Label-free when the reference is all of the patient's clips; baseline-referenced when
    it is a reserved set of negatives. The input-level counterpart of the interictal anchor.
    """
    out = np.empty_like(W)
    for s in np.unique(g):
        R = ref[s].astype(np.float64)
        R = R.reshape(-1, R.shape[-2], R.shape[-1])
        C = np.einsum("nct,ndt->cd", R, R) / (R.shape[0] * R.shape[-1])
        P = _inv_sqrt(C).astype(np.float32)
        idx = np.where(g == s)[0]
        out[idx] = np.einsum("cd,nsdt->nsct", P, W[idx])
    return out


def feature_align(X: np.ndarray, g: np.ndarray, ref: dict[str, np.ndarray], mode: str = "z",
                  shrink: bool = True) -> np.ndarray:
    """Per-patient reference normalisation of the 18-D vector: z-score or second-order whitening."""
    out = np.empty_like(X, dtype=np.float64)
    for s in np.unique(g):
        R = np.asarray(ref[s], np.float64)
        mu = R.mean(0)
        idx = np.where(g == s)[0]
        if mode == "z":
            sd = R.std(0) + 1e-6
            out[idx] = (X[idx] - mu) / sd
        else:
            if shrink:
                from sklearn.covariance import LedoitWolf
                C = LedoitWolf().fit(R).covariance_
            else:
                C = np.cov(R.T)
            out[idx] = (X[idx] - mu) @ _inv_sqrt(C)
    return np.nan_to_num(np.clip(out, -20, 20)).astype(np.float32)


def seal_sha() -> str:
    return json.loads(Path("results/_blind_split.json").read_text())["sha256"]
