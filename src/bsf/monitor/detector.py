"""Seizure detection for the monitor: the paper's AURA-Ens, loaded from its banked fold models.

``results/aura_ens_db18`` holds, per fold, every member trained on the other folds' patients: the feature
heads, each network with its post-hoc and PBN heads, AURA-Net, and the stacking weights. A development
patient is scored by the fold that held it out (no leakage); any other patient by the average of the
three folds. Members see the 18-channel double banana, standardised 8-s windows, and the patient's
refreshed reference (windows 35 to 5 min earlier).
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from bsf.models import pbn as PBN

SPAN_S, GAP_S, MIN_REF = 1800.0, 300.0, 10


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class SeizureDetector:
    def __init__(self, ens_dir: str | Path, stack: str = "post", device: str = "cpu"):
        self.dir = Path(ens_dir)
        self.R = json.loads((self.dir / "results.json").read_text())
        self.stack, self.device = stack, device
        self.folds = self.R["folds"]

    def folds_for(self, patient: str) -> list[int]:
        return [self.folds[patient]] if patient in self.folds else [0, 1, 2]

    def _members(self, f: int):
        return self.R["stacking"][f"{self.stack}|f{f}"]

    def score(self, W: np.ndarray, t: np.ndarray, F: np.ndarray, patient: str) -> dict:
        """W (n, 18, 2000) standardised windows in time order, t (n,) absolute starts, F (n, 18) AURA
        features. Returns member and ensemble probabilities per window (zeros-reference before 35 min)."""
        import torch

        from bsf.models import zoo
        from bsf.monitor.embedding import embed
        U = self.dir / "units"
        Fz = PBN.rolling_reference(np.nan_to_num(F), t, 0, SPAN_S, GAP_S, MIN_REF)
        out, ens = {}, []
        for f in self.folds_for(patient):
            comp = self._members(f)
            cols = []
            for c in comp["components"]:
                m, arm = c.split("|")
                key = f"{c}|f{f}"
                if m.startswith("feat_"):
                    with open(U / f"features_f{f}.pkl", "rb") as fh:
                        p = pickle.load(fh)[(m, arm)].predict_proba(Fz if arm == "refreshed" else F)[:, 1]
                elif m == "auranet_hybrid":
                    clf = torch.load(U / f"auranet_hybrid_{arm}_f{f}.clf", map_location=self.device, weights_only=False)
                    clf.device = self.device
                    p = clf.predict_proba(W[:, None].astype(np.float32), np.array([patient] * len(W)),
                                          feats=Fz if arm == "refreshed" else F)
                else:
                    spec = zoo.build(m, W.shape[1], W.shape[2], 250.0, device=self.device, seed=0)
                    spec.module.load_state_dict(torch.load(U / f"{m}_f{f}.pt", map_location=self.device))
                    mod = spec.module.eval()
                    if arm.startswith("probe_gbm"):
                        with open(U / f"{m}_f{f}_heads.pkl", "rb") as fh:
                            h = pickle.load(fh)["refreshed" if arm.endswith("refreshed") else "none"]
                        E = embed(mod, W, self.device)
                        p = h.predict_proba(PBN.rolling_reference(E, t, 0, SPAN_S, GAP_S, MIN_REF)
                                            if arm.endswith("refreshed") else E)[:, 1]
                    else:                                                     # pbn / pbn_adv
                        net = PBN.PBNNet(mod, PBN.embed_dim(mod, W.shape[1], W.shape[2], self.device))
                        sd = torch.load(U / f"{m}_f{f}_{arm}.pt", map_location=self.device)
                        net.load_state_dict({k: v for k, v in sd.items() if not k.startswith("adv.")}, strict=False)
                        net.to(self.device).eval()
                        E = PBN.embed_np(net, W, self.device)
                        p = PBN.classify_np(net, PBN.rolling_reference(E, t, 0, SPAN_S, GAP_S, MIN_REF), self.device)
                    del spec, mod
                out[key] = p
                cols.append(logit(p))
            z = np.column_stack(cols) @ np.asarray(comp["coef"]) + comp["intercept"]
            ens.append(1 / (1 + np.exp(-z)))
        return {"members": out, "ensemble": np.mean(ens, 0), "features_z": Fz}
