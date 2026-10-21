#!/usr/bin/env python
"""Monitor module M3: IED detection, typing and channel maps on VEPISET (development patients only).

Every 4-s epoch of the development patients' recordings, on the common double banana at 250 Hz.
Five patient-grouped folds. Detection arms (IED vs non-IED):

  gbm_raw     boosted trees on per-channel transient/spectral features + AURA's 18 global features
  gbm_patz    the same features robust-z-scored by the patient's own median epoch (label-free reference)
  cnn         the repository CNN on the standardised epoch
  cnn_patz    a boosted head on the CNN's embedding, patient-z-scored (representation-level reference)
  ens         mean of gbm_patz and cnn_patz (declared before running)

Reported: pooled and per-patient AUC and AUPRC; sensitivity at 1 false detection per minute of non-IED
EEG (threshold chosen on the training folds); per-patient IED burden (detected vs annotated count,
Spearman). Typing on the annotated IEDs: generalized vs focal, and lobe among focal (balanced accuracy,
macro AUC), from patient-z features; and whether the unsupervised channel map points to the labelled
lobe. Final models are refitted on all development patients for the monitor.

-> results/ied_monitor_<stamp>/results.json ; results/_monitor_models/ied/
"""
from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
MODELS = REPO / "results" / "_monitor_models" / "ied"


def main() -> int:
    import argparse

    import torch
    from scipy.stats import spearmanr
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import GroupKFold

    from bsf.models import zoo
    from bsf.monitor import ied as I
    from bsf.monitor import signals as S
    from bsf.utils.compute import resolve_device
    from bsf.utils.io import save_json
    from bsf.utils.provenance import new_results_dir
    from run_stream_tta import embed
    from run_unified_benchmark import dev_subjects

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--neg-ratio", type=float, default=3.0, help="non-IED:IED epochs per training fold for the CNN")
    ap.add_argument("--net", default="deep_cnn_raw")
    a = ap.parse_args()
    dev = resolve_device("auto")
    pats = dev_subjects("vepiset_typing", "dev")
    D = I.build_cache(REPO / "datasets" / "vepiset-dataset", pats, REPO / "results/_clip_cache/vepiset_db18_dev.npz")
    X, cls, pid = D["X"], D["cls"].astype(int), D["pid"]
    y = (cls > 0).astype(int)
    fcache = REPO / "results/_clip_cache/vepiset_db18_dev_feats.npz"
    if fcache.exists():
        d = np.load(fcache); Fch, Fgl = d["Fch"], d["Fgl"]
    else:
        Fch, Fgl = I.epoch_features(X); np.savez(fcache, Fch=Fch, Fgl=Fgl)
    Famp = I.amplitude_features(X)
    Zch, Zamp, Zgl = I.patient_z(Fch, pid), I.patient_z(Famp, pid), I.patient_z(Fgl, pid)
    Xraw, Xz = I.design(Fch, Famp, Fgl), I.design(Zch, Zamp, Zgl)
    out = new_results_dir("ied_monitor", seed=0, config={**vars(a), "n_patients": len(np.unique(pid))})
    print(f"=== IED monitor: {len(y)} epochs ({y.sum()} IED) / {len(np.unique(pid))} patients, device {dev} ===", flush=True)

    hgb = lambda: HistGradientBoostingClassifier(max_iter=300, max_depth=None, learning_rate=0.08,  # noqa: E731
                                                 class_weight="balanced", random_state=0)
    folds = list(GroupKFold(5).split(Xraw, y, pid))
    P = {k: np.zeros(len(y)) for k in ("gbm_raw", "gbm_patz", "cnn", "cnn_patz")}
    Wstd = None
    rng = np.random.default_rng(0)
    for f, (tr, te) in enumerate(folds):
        P["gbm_raw"][te] = hgb().fit(Xraw[tr], y[tr]).predict_proba(Xraw[te])[:, 1]
        P["gbm_patz"][te] = hgb().fit(Xz[tr], y[tr]).predict_proba(Xz[te])[:, 1]
        if Wstd is None:
            Wstd = np.stack([S.standardise(w) for w in X])
        pos = tr[y[tr] == 1]; neg = tr[y[tr] == 0]
        neg = rng.choice(neg, min(len(neg), int(a.neg_ratio * len(pos))), replace=False)
        sel = np.sort(np.r_[pos, neg])
        spec = zoo.build(a.net, 18, Wstd.shape[-1], S.FS, device=dev, seed=0)
        P["cnn"][te] = zoo.fit_predict(spec, Wstd[sel], y[sel], Wstd[te], dev, epochs=a.epochs, seed=0)
        mod = spec.module.eval()
        E = embed(mod, Wstd, dev)
        Ez = I.patient_z(E, pid)
        P["cnn_patz"][te] = HistGradientBoostingClassifier(max_iter=200, max_depth=3, class_weight="balanced",
                                                           random_state=0).fit(Ez[sel], y[sel]).predict_proba(Ez[te])[:, 1]
        MODELS.mkdir(parents=True, exist_ok=True)
        torch.save(mod.state_dict(), MODELS / f"ied_cnn_f{f}.pt")
        del spec, mod, E, Ez
        if dev == "cuda":
            torch.cuda.empty_cache()
        print(f"  fold {f}: " + ", ".join(f"{k} {roc_auc_score(y[te], v[te]):.3f}" for k, v in P.items()), flush=True)
    P["ens"] = (P["gbm_patz"] + P["cnn_patz"]) / 2

    # ---- detection metrics
    epoch_min = I.WIN_S / 60.0
    R = {"n_epochs": int(len(y)), "n_ied": int(y.sum()), "n_patients": int(len(np.unique(pid))), "detection": {}}
    for k, p in P.items():
        per = [roc_auc_score(y[pid == s], p[pid == s]) for s in np.unique(pid) if 0 < y[pid == s].sum() < (pid == s).sum()]
        sens, burden_pred = [], {}
        for tr, te in folds:
            negp = np.sort(p[tr][y[tr] == 0])[::-1]
            n_min = (y[tr] == 0).sum() * epoch_min
            thr = negp[min(len(negp) - 1, int(n_min * 1.0))]           # 1 false detection per minute
            sens.append(((p[te] >= thr) & (y[te] == 1)).sum())
            for s in np.unique(pid[te]):
                m = te[pid[te] == s]
                burden_pred[s] = int((p[m] >= thr).sum())
        true_b = {s: int(y[pid == s].sum()) for s in burden_pred}
        rho = spearmanr([true_b[s] for s in burden_pred], [burden_pred[s] for s in burden_pred]).correlation
        R["detection"][k] = {"auc": round(float(roc_auc_score(y, p)), 4), "auprc": round(float(average_precision_score(y, p)), 4),
                             "median_patient_auc": round(float(np.median(per)), 4), "n_patients_auc": len(per),
                             "sens_at_1fp_per_min": round(float(sum(sens) / y.sum()), 4),
                             "burden_spearman": round(float(rho), 4)}
        print(f"[{k}] AUC {R['detection'][k]['auc']:.3f} AUPRC {R['detection'][k]['auprc']:.3f} "
              f"patient-median {R['detection'][k]['median_patient_auc']:.3f} sens@1/min {R['detection'][k]['sens_at_1fp_per_min']:.3f} "
              f"burden rho {rho:.2f}", flush=True)

    # ---- typing on annotated IEDs
    ie = np.flatnonzero(y == 1)
    gen = (cls[ie] == 1).astype(int)
    pg = np.zeros(len(ie))
    for tr, te in GroupKFold(5).split(ie, gen, pid[ie]):
        pg[te] = hgb().fit(Xz[ie][tr], gen[tr]).predict_proba(Xz[ie][te])[:, 1]
    R["typing"] = {"generalized_vs_focal_auc": round(float(roc_auc_score(gen, pg)), 4),
                   "balanced_acc": round(float(balanced_accuracy_score(gen, pg >= 0.5)), 4)}
    fo = ie[cls[ie] > 1]
    lob = cls[fo]
    pl = np.zeros((len(fo), 4))
    for tr, te in GroupKFold(5).split(fo, lob, pid[fo]):
        c = hgb().fit(Xz[fo][tr], lob[tr])
        pl[te] = c.predict_proba(Xz[fo][te])
    pred = np.array([2, 3, 4, 5])[pl.argmax(1)]
    R["lobe"] = {"balanced_acc": round(float(balanced_accuracy_score(lob, pred)), 4), "chance": 0.25,
                 "macro_auc": round(float(roc_auc_score(lob, pl, multi_class="ovr", average="macro")), 4),
                 "n": int(len(fo)), "counts": {I.CLASS[k]: int((lob == k).sum()) for k in (2, 3, 4, 5)}}
    # unsupervised channel map: does the region with the strongest patient-z evidence match the label?
    names = {2: "frontal", 3: "temporal", 4: "centro_parietal", 5: "occipital"}
    cm = I.channel_map(Zch[fo])
    best = [max(I.region_scores(c).items(), key=lambda kv: kv[1])[0] for c in cm]
    R["lobe"]["map_region_balanced_acc"] = round(float(balanced_accuracy_score([names[k] for k in lob], best)), 4)
    print(f"typing: gen-vs-focal AUC {R['typing']['generalized_vs_focal_auc']:.3f}; lobe bal-acc {R['lobe']['balanced_acc']:.3f} "
          f"(map-only {R['lobe']['map_region_balanced_acc']:.3f}, chance 0.25)", flush=True)

    # ---- final models on all development patients, for the monitor
    MODELS.mkdir(parents=True, exist_ok=True)
    final = {"detect_gbm_patz": hgb().fit(Xz, y), "type_gen_vs_focal": hgb().fit(Xz[ie], gen), "lobe": hgb().fit(Xz[fo], lob),
             "design": "bsf.monitor.ied.design(patient_z(channel_features), patient_z(amplitude_features), patient_z(rich_plus))",
             "win_s": I.WIN_S, "fs": S.FS}
    with open(MODELS / "ied_models.pkl", "wb") as fh:
        pickle.dump(final, fh)
    np.savez_compressed(out / "preds.npz", y=y, cls=cls, pid=pid, **P)
    save_json(R, out / "results.json")
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
