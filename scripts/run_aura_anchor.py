#!/usr/bin/env python
"""AURA-Net with a working patient anchor, against AURA-Net without one, on identical clips.

In the benchmark AURA-Net's anchor whitened every training patient and no test patient, so it was
switched off (Section IV-G). Here every patient is anchored from its own baseline, training and test
alike, and the baseline always precedes or lies outside the scored data:

  typing    the patient's non-discharge background (separate clips; nothing removed from the pool)
  others    the patient's earliest third of negative clips in recording order (removed from scoring)

The anchor layer is fitted on the baseline embeddings of each training patient, refreshed during
training as before, and set for each test patient with ``adapt_anchor`` from that patient's baseline
before scoring. The hybrid configuration also z-scores the AURA features by the same baseline. The
unanchored arm is the frozen AURA-Net on the same reduced clip set and folds. Three seeds per arm.

Resumable: every (task, config, arm, seed) is banked in results/_aura_anchor/state.json with its
out-of-fold predictions, so a restart repeats only the run in flight.

  python scripts/run_aura_anchor.py --configs pure            # AURA-Net (raw input)
  python scripts/run_aura_anchor.py --configs hybrid          # AURA-Net with the AURA features
  python scripts/run_aura_anchor.py --epochs 1 --tasks typing --seeds 0 --tag smoke

-> results/aura_anchor_<stamp>/results.json (+ results/_aura_anchor/*.npy)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
SFREQ = {"typing": 500.0, "ieeg": 400.0, "scalp": 250.0, "detect": 250.0}


def _task_data(task):
    """Clips, labels, patients, features and the per-patient baseline (windows + features)."""
    from bench_data import feature_align, load_task, typing_baseline
    from run_baseline_placement import early_mask
    W, y, g, ev, X = load_task(task, "dev")
    if task == "typing":
        Wb, gb, Xb = typing_baseline("dev")
        n_ch = min(W.shape[2], Wb.shape[2]); n_t = min(W.shape[3], Wb.shape[3])
        W = np.ascontiguousarray(W[:, :, :n_ch, :n_t]); Wb = Wb[:, :, :n_ch, :n_t]
        refW = {s: Wb[gb == s] for s in np.unique(g) if (gb == s).sum() >= 3}
        refX = {s: Xb[gb == s] for s in refW}
        keep = np.isin(g, list(refW))
    else:
        base = early_mask(y, g, ev)
        refW = {s: W[(g == s) & base] for s in np.unique(g) if ((g == s) & base).sum() >= 3}
        refX = {s: X[(g == s) & base] for s in refW}
        keep = ~base & np.isin(g, list(refW))
    W, y, g, X = W[keep], y[keep], g[keep], X[keep]
    Xa = feature_align(X, g, refX, "z")
    return W, y, g, X, Xa, refW


def _anchored_classifier(refW, **kw):
    import torch

    from bsf.models.auranet_train import AURANetClassifier

    class AnchoredAURANet(AURANetClassifier):
        """Anchor statistics from each patient's BASELINE windows, not from its training negatives."""

        @torch.no_grad()
        def _fit_anchor(self, W, y, subj):
            self.model.eval()
            for s in np.unique(subj):
                R = refW.get(str(s))
                if R is None or len(R) < 3:
                    continue
                zs = []
                for i in range(0, len(R), 16):
                    b = torch.tensor(R[i:i + 16], device=self.device)
                    B, S, M, L = b.shape
                    zz, _ = self.model.embed(b.reshape(B * S, M, L))
                    zs.append(zz.reshape(B, S, -1).mean(1))
                self.model.anchor.fit_reference(str(s), torch.cat(zs))

    return AnchoredAURANet(anchor=True, **kw)


def main() -> int:
    from sklearn.model_selection import StratifiedGroupKFold

    from bsf.eval.clinical import auc
    from bsf.models.auranet_train import AURANetClassifier
    from bsf.utils.io import save_json
    from bsf.utils.provenance import new_results_dir
    from bsf.utils.seeds import set_global_seeds

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default="typing,ieeg,scalp,detect")
    ap.add_argument("--configs", default="pure")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    state_dir = REPO / "results" / f"_aura_anchor{('_' + a.tag) if a.tag else ''}"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_f = state_dir / "state.json"
    state = json.loads(state_f.read_text()) if state_f.exists() else {}
    seeds = [int(s) for s in a.seeds.split(",")]
    out = new_results_dir("aura_anchor", seed=1234, config=vars(a))
    for task in [t for t in a.tasks.split(",") if t]:
        W, y, g, X, Xa, refW = _task_data(task)
        splits = 5 if task == "typing" else 3
        folds = list(StratifiedGroupKFold(splits, shuffle=True, random_state=1234)
                     .split(np.zeros(len(y)), y, g))
        np.save(state_dir / f"{task}_y.npy", y); np.save(state_dir / f"{task}_g.npy", g)
        print(f"[{task}] {len(y)} clips / {len(np.unique(g))} patients with a baseline | {splits} folds",
              flush=True)
        for cfg in [c for c in a.configs.split(",") if c]:
            hybrid = cfg == "hybrid"
            for arm in ("none", "anchored"):
                for sd in seeds:
                    key = f"{task}|{cfg}|{arm}|s{sd}"
                    if key in state:
                        print(f"  (resume) {key} AUC {state[key]['auc']:.3f}", flush=True)
                        continue
                    t0 = time.time()
                    oof = np.zeros(len(y))
                    for tr, te in folds:
                        set_global_seeds(sd)
                        kw = dict(fs=SFREQ[task], epochs=a.epochs, device="cuda", seed=sd,
                                  d_feat=X.shape[1] if hybrid else 0)
                        F = (Xa if arm == "anchored" else X) if hybrid else None
                        if arm == "anchored":
                            c = _anchored_classifier(refW, **kw)
                            c.fit(W[tr], y[tr], g[tr], feats=None if F is None else F[tr])
                            # each test patient is anchored from its own baseline before it is scored
                            ts = np.unique(g[te])
                            c.adapt_anchor(np.concatenate([refW[s] for s in ts]),
                                           np.concatenate([[s] * len(refW[s]) for s in ts]))
                        else:
                            c = AURANetClassifier(**kw)
                            c.fit(W[tr], y[tr], g[tr], feats=None if F is None else F[tr])
                        oof[te] = c.predict_proba(W[te], g[te], feats=None if F is None else F[te])
                        del c
                    np.save(state_dir / f"{task}_{cfg}_{arm}_s{sd}.npy", oof)
                    state[key] = {"auc": round(float(auc(y, oof)), 4), "seconds": round(time.time() - t0)}
                    state_f.write_text(json.dumps(state, indent=1))
                    print(f"  {key} AUC {state[key]['auc']:.3f} ({state[key]['seconds']} s)", flush=True)
    save_json(state, out / "results.json")
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
