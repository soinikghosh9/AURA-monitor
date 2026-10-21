#!/usr/bin/env python
"""End-to-end deep baseline on RAW MULTICHANNEL iEEG at full bandwidth (OFFLINE, resumable).

The manuscript's original raw-signal baselines (run_real_baselines.py, run_raw_ssl.py) feed the deep
model a channel-AVERAGED clip resampled to 256 samples -- roughly a 0.2 Hz effective bandwidth. The
gamma and HFO content that the 18-D feature vector depends on has already been destroyed before the
network sees anything, so those runs cannot support a claim about capacity versus representation.

This script gives the deep model the information the feature vector actually uses:
  * ALL channels (no spatial averaging), so cross-channel synchrony is learnable;
  * decimated only to ~400 Hz, so the full 70-150 Hz HFO band survives (Nyquist ~200 Hz);
  * several evenly-spaced sub-windows per clip, mean-pooled to a clip-level probability, so the model
    sees the whole clip as the features do.
The feature model is re-scored on the IDENTICAL clips and folds inside this script, so the comparison
is matched end to end. Both stratified and seizure-hour-grouped CV are reported.

Per-subject resume via resume.update_unit -> results/deep_fair_<stamp>/results.json
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

TARGET_FS = 400.0          # keeps the full HFO band (70-150 Hz) below Nyquist
SUBWIN_S = 8.0             # sub-window length in seconds
N_SUBWIN = 12              # evenly-spaced sub-windows per clip


def _subwindows(data, fs, n=N_SUBWIN, win_s=SUBWIN_S, target_fs=TARGET_FS):
    """(T, M) clip -> (n, M, L) sub-windows, decimated to ~target_fs, per-window per-channel z-scored."""
    from scipy.signal import decimate
    x = np.nan_to_num(np.asarray(data, dtype=np.float32))
    if x.ndim == 1:
        x = x[:, None]
    q = max(1, int(round(fs / target_fs)))
    if q > 1:
        x = decimate(x, q, axis=0, ftype="fir").astype(np.float32)
    fs_eff = fs / q
    L = int(round(win_s * fs_eff))
    if x.shape[0] < L:
        return None
    starts = np.linspace(0, x.shape[0] - L, n).astype(int)
    W = np.stack([x[s:s + L].T for s in starts])                 # (n, M, L)
    mu = W.mean(2, keepdims=True); sd = W.std(2, keepdims=True) + 1e-6
    return np.nan_to_num((W - mu) / sd).astype(np.float32)


def _build_model(n_ch, device):
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv1d(n_ch, 64, 15, stride=4, padding=7), nn.BatchNorm1d(64), nn.GELU(),
        nn.Conv1d(64, 64, 9, stride=4, padding=4), nn.BatchNorm1d(64), nn.GELU(),
        nn.Conv1d(64, 96, 7, stride=2, padding=3), nn.BatchNorm1d(96), nn.GELU(),
        nn.Conv1d(96, 96, 5, stride=2, padding=2), nn.BatchNorm1d(96), nn.GELU(),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        nn.Dropout(0.3), nn.Linear(96, 2),
    ).to(device)


def _fit_predict(Wtr, ytr, Wte, device, epochs=25, lr=1e-3, bs=64, seed=0, schedule=False):
    """Train the multichannel CNN on sub-windows; return per-sub-window positive probabilities.

    ``schedule`` adds the same 10% warmup + cosine decay AURA-Net uses. It defaults to OFF so the
    published deep_fair number (0.845) stays reproducible, and is switched ON by the AURA-Net
    benchmark: scoring a scheduled model against an unscheduled one is an asymmetry in the NEW model's
    favour, and every other knob in that comparison is matched.
    """
    import math

    import torch
    torch.manual_seed(seed)
    model = _build_model(Wtr.shape[1], device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    cnt = np.bincount(ytr, minlength=2).astype(float)
    w = torch.tensor((cnt.sum() / (2 * np.maximum(cnt, 1))), dtype=torch.float32, device=device)
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    Xt = torch.tensor(Wtr); yt = torch.tensor(ytr, dtype=torch.long)
    n = len(yt)
    sched = None
    if schedule:
        total = max(1, epochs * int(np.ceil(n / bs)))
        warm = max(1, int(0.1 * total))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm
            else 0.5 * (1.0 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = Xt[idx].to(device, non_blocking=True); yb = yt[idx].to(device, non_blocking=True)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward(); opt.step()
            if sched is not None:
                sched.step()
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(Wte), 256):
            xb = torch.tensor(Wte[i:i + 256]).to(device)
            outs.append(torch.softmax(model(xb), 1)[:, 1].cpu().numpy())
    return np.concatenate(outs)


def _cv(y_clip, groups, grouped, seed=0, n_splits=3):
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    if grouped:
        return StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(
            np.zeros(len(y_clip)), y_clip, groups)
    return StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(
        np.zeros(len(y_clip)), y_clip)


def main() -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from bsf.data.features import rich_features
    from bsf.data.kaggle_seizure import discover_kaggle, load_kaggle_segment
    from bsf.dynamotype.labeling import select_seizure_channel
    from bsf.eval.metrics import roc_auc
    from bsf.utils import resume
    from bsf.utils.compute import resolve_device
    from bsf.utils.io import save_json
    from bsf.utils.provenance import new_results_dir
    from bsf.utils.seeds import set_global_seeds

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-per-class", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--n-subwin", type=int, default=N_SUBWIN)
    args = ap.parse_args()
    set_global_seeds(1234)
    dev = resolve_device("auto")
    out = new_results_dir("deep_fair", seed=1234, config={**vars(args), "device": dev,
                                                          "target_fs": TARGET_FS,
                                                          "subwin_s": SUBWIN_S})
    name = "deep_fair"
    done = resume.done_units(name)
    print(f"=== fair deep baseline: device={dev}, {args.n_subwin}x{SUBWIN_S}s @ {TARGET_FS}Hz, "
          f"all channels ===", flush=True)

    rows = {}
    for root in ("datasets/kaggle_aes", "datasets/ieeg_canine"):
        disc = discover_kaggle(root)
        for subj in sorted(disc):
            label = f"{'kaggle' if 'kaggle' in root else 'canine'}_{subj}"
            if label in done:
                rows[label] = done[label]
                print(f"  (resume) {label} = {done[label]}", flush=True)
                continue

            # ---- load once: rich features, slow control, raw sub-windows, hour groups ----
            from bsf.data.windows import slow_feature
            Xr, Xsl, W, y, groups = [], [], [], [], []
            gid, prev_seq = 0, None
            for ln, lab in (("preictal", 1), ("interictal", 0)):
                prev_seq = None
                for f in disc[subj].get(ln, [])[: args.max_per_class]:
                    seg = load_kaggle_segment(f)
                    if seg.data.ndim != 2 or seg.data.shape[0] < 120:
                        continue
                    sw = _subwindows(seg.data, seg.fs, n=args.n_subwin)
                    if sw is None:
                        continue
                    ch = select_seizure_channel(seg.data, 0, seg.data.shape[0])
                    Xsl.append(slow_feature(seg.data[:, ch], seg.fs))
                    Xr.append(rich_features(seg.data, seg.fs))
                    W.append(sw); y.append(lab)
                    if ln == "preictal" and seg.sequence is not None and prev_seq is not None \
                            and seg.sequence > prev_seq:
                        groups.append(gid)
                    else:
                        gid += 1; groups.append(gid)
                    prev_seq = seg.sequence if ln == "preictal" else None
            if len(y) < 6:
                continue
            Xr = np.asarray(Xr, np.float32); Xsl = np.asarray(Xsl, np.float32)
            y = np.asarray(y, int); groups = np.asarray(groups, int)
            fin = np.isfinite(Xr).all(1) & np.isfinite(Xsl).all(1)
            keep = np.where(fin)[0]
            if len(np.unique(y[keep])) < 2 or int(np.bincount(y[keep]).min()) < 3:
                continue
            Xr, Xsl, y, groups = Xr[keep], Xsl[keep], y[keep], groups[keep]
            W = [W[i] for i in keep]
            n_ch = min(w.shape[1] for w in W)                 # w is (n_sub, n_ch, L)
            Wall = np.stack([w[:, :n_ch] for w in W])         # (Nclip, n_sub, n_ch, L)

            rec = {"n_clips": int(len(y)), "n_channels": int(n_ch),
                   "n_hours": int(len(np.unique(groups[y == 1])))}
            for tag, grouped in (("stratified", False), ("grouped", True)):
                a_deep, a_rich, a_slow = [], [], []
                for tr, te in _cv(y, groups, grouped):
                    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                        continue
                    # deep: train on sub-windows, mean-pool sub-window probs per clip
                    Wtr = Wall[tr].reshape(-1, n_ch, Wall.shape[-1])
                    ytr_sub = np.repeat(y[tr], Wall.shape[1])
                    Wte = Wall[te].reshape(-1, n_ch, Wall.shape[-1])
                    try:
                        p_sub = _fit_predict(Wtr, ytr_sub, Wte, dev, epochs=args.epochs)
                        p_clip = p_sub.reshape(len(te), Wall.shape[1]).mean(1)
                        a_deep.append(roc_auc(y[te], p_clip))
                    except Exception as e:  # noqa: BLE001
                        print(f"    {label} deep fold failed: {str(e)[:70]}", flush=True)
                    for store, X in ((a_rich, Xr), (a_slow, Xsl)):
                        clf = make_pipeline(StandardScaler(),
                                            LogisticRegression(max_iter=1000, class_weight="balanced"))
                        store.append(roc_auc(y[te], clf.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]))

                def m(a):
                    a = [v for v in a if v is not None and np.isfinite(v)]
                    return round(float(np.mean(a)), 3) if a else None
                rec[tag] = {"deep_multichannel_raw": m(a_deep), "rich_logistic": m(a_rich),
                            "slow_envelope": m(a_slow)}
                print(f"  {label:18s} [{tag:10s}] deep {rec[tag]['deep_multichannel_raw']} | "
                      f"rich {rec[tag]['rich_logistic']} | slow {rec[tag]['slow_envelope']}", flush=True)
            rows[label] = rec
            resume.update_unit(name, label, rec)

    def agg(tag, key):
        v = [r[tag][key] for r in rows.values()
             if tag in r and r[tag].get(key) is not None]
        return round(float(np.mean(v)), 3) if v else None

    R = {"protocol": f"raw multichannel iEEG, {args.n_subwin}x{SUBWIN_S}s sub-windows @ ~{TARGET_FS}Hz, "
                     "clip-level mean pooling; feature models scored on identical clips and folds",
         "per_subject": rows, "n_subjects": len(rows),
         "mean": {tag: {k: agg(tag, k) for k in
                        ("deep_multichannel_raw", "rich_logistic", "slow_envelope")}
                  for tag in ("stratified", "grouped")}}
    # paired per-subject deep-vs-rich contrast (the claim under test)
    for tag in ("stratified", "grouped"):
        d = [(r[tag]["rich_logistic"] - r[tag]["deep_multichannel_raw"]) for r in rows.values()
             if tag in r and r[tag].get("deep_multichannel_raw") is not None
             and r[tag].get("rich_logistic") is not None]
        if len(d) >= 3:
            d = np.asarray(d, float)
            rng = np.random.default_rng(0)
            boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
            from scipy.stats import wilcoxon
            R.setdefault("paired_rich_minus_deep", {})[tag] = {
                "n": int(len(d)), "mean_delta": round(float(d.mean()), 3),
                "median_delta": round(float(np.median(d)), 3),
                "ci95": [round(float(np.percentile(boots, 2.5)), 3),
                         round(float(np.percentile(boots, 97.5)), 3)],
                "wilcoxon_p": round(float(wilcoxon(d).pvalue), 4) if np.any(d != 0) else 1.0,
                "frac_rich_better": round(float(np.mean(d > 0)), 2)}
    save_json(R, out / "results.json")
    print(f"\n=== MEAN === {R['mean']}", flush=True)
    print(f"=== PAIRED (rich - deep) === {R.get('paired_rich_minus_deep')}", flush=True)
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
