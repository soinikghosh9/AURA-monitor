#!/usr/bin/env python
"""Does mains interference carry site and patient identity into the 18 AURA features? (audit E3, CPU only)

CHB-MIT was recorded on 60 Hz mains (harmonic 120 Hz) and Siena on 50 Hz (harmonic 100 Hz). The gamma band
(30-70 Hz) contains both fundamentals and the high-frequency band (70-124 Hz at 250 Hz) both harmonics, and no
notch filter is applied. For the scalp tasks, on the networks' own standardized sub-windows:

  line     relative power at 50, 60, 100 and 120 Hz (against the 4 Hz on either side), per patient and cohort
  variants features as published; after zero-phase notches at 50/60/100/120 Hz; after a 45 Hz low-pass
  scores  development AUC on the benchmark's folds (logistic, boosted trees), patient-identity accuracy with
           the paper's probe (20 principal components, 5-fold logistic), and the per-feature patient share

  python scripts/run_mains_confound.py   -> results/mains_confound_<stamp>/results.json
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
FS = 250.0
N_JOBS = 12


def _filter(W, kind):
    from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt
    X = W.astype(np.float64)
    if kind == "notch":
        for f0 in (50.0, 60.0, 100.0, 120.0):
            b, a = iirnotch(f0, 30.0, FS)
            X = filtfilt(b, a, X, axis=-1)
    elif kind == "lp45":
        X = sosfiltfilt(butter(4, 45.0 / (FS / 2), output="sos"), X, axis=-1)
    return X.astype(np.float32)


def _feat_clip(Wc, kind):
    from bsf.data.features import rich_plus
    Wf = _filter(Wc, kind)
    return np.mean([np.nan_to_num(rich_plus(Wf[j].T, FS)) for j in range(len(Wf))], 0)


def features(W, kind):
    from joblib import Parallel, delayed
    os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
    out = Parallel(n_jobs=N_JOBS, batch_size=8)(delayed(_feat_clip)(W[i], kind) for i in range(len(W)))
    return np.asarray(out, np.float32)


def line_power(W):
    from scipy.signal import welch
    f, p = welch(W[:, :, :, :].astype(np.float64), fs=FS, nperseg=500, axis=-1)
    p = p.mean(axis=(1, 2))                          # (clips, freqs): mean over sub-windows and channels
    out = {}
    for f0 in (50.0, 60.0, 100.0, 120.0):
        on = np.abs(f - f0) <= 0.5
        near = (np.abs(f - f0) >= 2) & (np.abs(f - f0) <= 4)
        out[f0] = np.log10(p[:, on].mean(1) / p[:, near].mean(1))
    return out


def main() -> int:
    from sklearn.base import clone
    from sklearn.decomposition import PCA
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from bench_data import load_task
    from run_identity_decomposition import shares

    os.chdir(ROOT)
    out = ROOT / "results" / f"mains_confound_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out.mkdir(parents=True)
    R: dict = {}
    for task in ("detect", "scalp"):
        W, y, g, _ev, _X = load_task(task, "dev")
        g = np.asarray(g).astype(str)
        coh = np.array([s.split("_")[0] for s in g])
        lp = line_power(W)
        R[task] = {"n_clips": int(len(y)), "n_patients": int(len(np.unique(g))), "line_power_log10": {}}
        for f0, v in lp.items():
            R[task]["line_power_log10"][f"{int(f0)}Hz"] = {c: round(float(np.median(v[coh == c])), 3) for c in np.unique(coh)}
        folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=1234).split(np.zeros(len(y)), y, g))
        for kind in ("none", "notch", "lp45"):
            X = features(W, kind)
            r = {}
            for mname, clf in (("logistic", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))),
                               ("gbm", GradientBoostingClassifier(n_estimators=200, random_state=0))):
                oof = np.zeros(len(y))
                for tr, te in folds:
                    oof[te] = clone(clf).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
                r[f"auc_{mname}"] = round(float(roc_auc_score(y, oof)), 4)
            P = PCA(min(18, X.shape[1]), random_state=0).fit_transform(StandardScaler().fit_transform(X))
            pid = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)), P, g,
                                    cv=StratifiedKFold(5, shuffle=True, random_state=0))
            r["identity_acc"] = round(float(np.mean(pid == g)), 4)
            r["identity_chance"] = round(1 / len(np.unique(g)), 4)
            ccv = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)), P, coh,
                                    cv=StratifiedKFold(5, shuffle=True, random_state=0))
            r["cohort_acc"] = round(float(np.mean(ccv == coh)), 4)
            pat, _lab = shares(X, y, g)
            r["median_patient_share"] = round(float(np.median(pat)), 4)
            R[task][kind] = r
            print(task, kind, r, flush=True)
        R[task]["line_power_log10"] = R[task]["line_power_log10"]
        print(task, "line power", R[task]["line_power_log10"], flush=True)
        (out / "results.json").write_text(json.dumps(R, indent=1))
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
