#!/usr/bin/env python
"""What is the model actually using? Occlusion in frequency, space and time.

A clinical tool has to be able to say why it warned, and "the CNN said so" is not an answer a
neurologist can act on or contest. This measures the question directly rather than through a saliency
heuristic: remove one thing from the input of an ALREADY-TRAINED model, score it again, and report
what performance the removal cost. No gradients, no surrogate model, no attribution axioms to argue
about -- just the counterfactual, which is also the thing a clinician can reason about ("if this
electrode had failed, the warning would not have fired").

Three occlusions, each answering a question the manuscript actually asks:

  BAND      A band-stop filter removes one rhythm and leaves everything else. This is the direct test
            of the representation thesis: the 18-D analytic vector is built from specific bands, so if
            the raw-waveform models depend on the SAME bands, the two families are reading the same
            physiology and the architecture is not what separates them. It also checks the HFO claim,
            which no amount of AUC can support on its own.
  CHANNEL   Zeroing one electrode gives a spatial importance map, and the shape of the curve is a
            deployability fact: a model whose performance rests on one lead fails when that lead does.
  TIME      Zeroing one sub-window asks WHEN in the clip the evidence sits -- uniformly spread, or
            concentrated at the end, which for a pre-ictal clip would mean the model is keying on
            proximity to the horizon rather than on state.

Every number is an AUC DROP from the intact model on the same held-out clips, so it is comparable
across models with very different absolute performance. Agreement ACROSS architectures is reported
too: if twenty independently-designed models all lean on the same band, that is evidence about the
signal rather than about any one of them.

-> results/interpretability_<stamp>/results.json + interpretability.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_unified_benchmark import _load_detect, _load_ieeg, _load_scalp, _load_typing  # noqa: E402

SEAL = Path("results/_blind_split.json")
BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
         "beta": (13.0, 30.0), "gamma": (30.0, 80.0), "hfo": (80.0, 150.0)}


def _bandstop(W, fs, lo, hi):
    """Remove one band and leave the rest, via FFT masking on the last axis.

    A zero-phase FFT notch rather than an IIR filter: the point is to delete a band cleanly without
    imposing a phase shift that would itself change what a temporal model sees.
    """
    n = W.shape[-1]
    hi = min(hi, fs / 2 - 1e-6)
    if lo >= hi:
        return None                       # band lies above Nyquist for this cohort: not testable
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (f >= lo) & (f < hi)
    # Chunked: numpy's rfft promotes to complex128 regardless of input dtype, so transforming a whole
    # scalp fold at once (~180M samples) would allocate several GB for a filter that is not the point
    # of the experiment.
    out = np.empty_like(W, dtype=np.float32)
    flat = W.reshape(-1, n)
    oflat = out.reshape(-1, n)
    for i in range(0, len(flat), 4096):
        F = np.fft.rfft(flat[i:i + 4096], axis=-1)
        F[:, mask] = 0
        oflat[i:i + 4096] = np.fft.irfft(F, n=n, axis=-1).astype(np.float32)
    return out


def main() -> int:
    import torch
    from sklearn.model_selection import StratifiedGroupKFold

    from bsf.eval.clinical import auc
    from bsf.models import zoo
    from bsf.utils import resume
    from bsf.utils.compute import resolve_device
    from bsf.utils.io import save_json
    from bsf.utils.provenance import new_results_dir
    from bsf.utils.seeds import set_global_seeds

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default="scalp,ieeg,detect,typing")
    ap.add_argument("--models", default="", help="default: the 6 best from the frozen leaderboard")
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--max-per-class", type=int, default=60)
    ap.add_argument("--n-subwin", type=int, default=6)
    ap.add_argument("--max-per-subject", type=int, default=40)
    ap.add_argument("--micro-bs", type=int, default=16)
    args = ap.parse_args()
    set_global_seeds(1234)
    dev = resolve_device("auto")
    seal = json.loads(SEAL.read_text())
    out = new_results_dir("interpretability", seed=1234,
                          config={**vars(args), "seal": seal["sha256"]})
    print(f"=== interpretability by occlusion (DEV only, device={dev}) ===", flush=True)

    loaders = {"ieeg": lambda: _load_ieeg(args.max_per_class, args.n_subwin),
               "scalp": lambda: _load_scalp(args.max_per_class, args.n_subwin),
               "detect": lambda: _load_detect(args.max_per_class, args.n_subwin),
               "typing": lambda: _load_typing(args.max_per_subject)}
    sfreqs = {"ieeg": 400.0, "scalp": 250.0, "detect": 250.0, "typing": 500.0}

    def _leaderboard(task):
        """Model names ranked by the frozen DEV benchmark, so we explain what actually did well."""
        from bsf.utils.io import latest_result_dirs
        best = {}
        for d in latest_result_dirs("unified_benchmark_*")[-12:]:
            f = Path(d) / "results.json"
            if not f.exists():
                continue
            tr = json.loads(f.read_text()).get("tasks", {}).get(task)
            if not isinstance(tr, dict):
                continue
            for m, r in tr.items():
                if isinstance(r, dict) and "panel" in r and r.get("input") == "raw":
                    best.setdefault(m, r["panel"]["discrimination"]["auc"])
        return sorted(best, key=lambda k: -best[k])

    results = {}
    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        data = loaders[task]()
        if data is None:
            print(f"  [{task}] no DEV data -- skipped", flush=True)
            continue
        W, y, g, X = data
        n_clip, n_sub, n_ch, n_t = W.shape
        sf = sfreqs[task]
        names = ([m.strip() for m in args.models.split(",") if m.strip()]
                 or _leaderboard(task)[:args.top] or ["EEGNet", "deep_cnn_raw"])
        print(f"  [{task}] {n_clip} clips, {n_ch}ch x {n_t} @ {sf}Hz | explaining {names}",
              flush=True)
        unit = f"interp_{task}_e{args.epochs}_n{n_clip}"
        task_res = dict(resume.done_units(unit))
        folds = list(StratifiedGroupKFold(args.splits, shuffle=True,
                                          random_state=1234).split(np.zeros(len(y)), y, g))

        # AURA-Net is not in the zoo: it consumes the sub-window axis jointly rather than as
        # independent samples, so zoo.build rejects it by name. Occluding it therefore needs its own
        # path -- and it needs one, because it ranks FIRST on detection and leaving it unexplained
        # would mean the only model we cannot account for is our own winner.
        for mname, kw in (("auranet_pure", {}), ("auranet_hybrid", {"d_feat": X.shape[1]})):
            if mname in task_res:
                continue
            try:
                from bsf.models.auranet_train import AURANetClassifier
                base = np.zeros(len(y)); band = {b: np.zeros(len(y)) for b in BANDS}
                chan = np.zeros((n_ch, len(y))); tim = np.zeros((n_sub, len(y)))
                for tr, te in folds:
                    clf = AURANetClassifier(fs=sf, epochs=args.epochs, device=dev, seed=0, **kw)
                    fe = X if kw else None
                    clf.fit(W[tr], y[tr], g[tr], feats=fe[tr] if fe is not None else None)

                    def sc(Wx, _c=clf, _te=te, _f=fe):
                        return _c.predict_proba(Wx, g[_te],
                                                feats=_f[_te] if _f is not None else None)

                    Wte = W[te]
                    base[te] = sc(Wte)
                    for b, (lo, hi) in BANDS.items():
                        Wb = _bandstop(Wte, sf, lo, hi)
                        band[b][te] = sc(Wb) if Wb is not None else base[te]
                    for c in range(n_ch):
                        Wc = Wte.copy(); Wc[:, :, c, :] = 0.0
                        chan[c, te] = sc(Wc)
                    for s_ in range(n_sub):
                        Ws = Wte.copy(); Ws[:, s_, :, :] = 0.0
                        tim[s_, te] = sc(Ws)
                    del clf
                    if dev == "cuda":
                        torch.cuda.empty_cache()
                a0 = auc(y, base)
                rec = {"auc_intact": round(a0, 4),
                       "band_drop": {b: round(a0 - auc(y, band[b]), 4) for b in BANDS
                                     if _bandstop(W[:1], sf, *BANDS[b]) is not None},
                       "channel_drop": [round(a0 - auc(y, chan[c]), 4) for c in range(n_ch)],
                       "time_drop": [round(a0 - auc(y, tim[s_]), 4) for s_ in range(n_sub)]}
                cd = np.array(rec["channel_drop"])
                rec["channel_concentration"] = round(float(cd.max() / (cd.sum() + 1e-9)), 4)                     if cd.sum() > 0 else None
                rec["top_band"] = max(rec["band_drop"], key=rec["band_drop"].get)                     if rec["band_drop"] else None
                task_res[mname] = rec
                resume.update_unit(unit, mname, rec)
                print(f"    {mname:18s} intact {a0:.3f} top={rec['top_band']} "
                      f"worst-channel {cd.max():+.3f}", flush=True)
            except Exception as e:  # noqa: BLE001
                task_res[mname] = {"error": f"{type(e).__name__}: {str(e)[:100]}"}
                print(f"    {mname:18s} FAILED {str(e)[:70]}", flush=True)

        for mname in names:
            if mname.startswith("auranet"):
                continue                       # handled above by its own occlusion path
            if mname in task_res:
                print(f"    (resume) {mname}", flush=True); continue
            try:
                base = np.zeros(len(y))
                band = {b: np.zeros(len(y)) for b in BANDS}
                chan = np.zeros((n_ch, len(y)))
                tim = np.zeros((n_sub, len(y)))
                for tr, te in folds:
                    spec = zoo.build(mname, n_ch, n_t, sf, device=dev)
                    zoo.fit_predict(spec, W[tr].reshape(-1, n_ch, n_t), np.repeat(y[tr], n_sub),
                                    W[tr][:1].reshape(-1, n_ch, n_t), dev,
                                    epochs=args.epochs, micro_bs=args.micro_bs)
                    m = spec.module.eval()

                    def score(Wx, _m=m):
                        flat = Wx.reshape(-1, n_ch, n_t)
                        ps = []
                        with torch.no_grad():
                            for i in range(0, len(flat), 128):
                                xb = torch.tensor(flat[i:i + 128], dtype=torch.float32, device=dev)
                                ps.append(torch.softmax(_m(xb), 1)[:, 1].cpu().numpy())
                        return np.concatenate(ps).reshape(-1, n_sub).mean(1)

                    Wte = W[te]
                    base[te] = score(Wte)
                    for b, (lo, hi) in BANDS.items():
                        Wb = _bandstop(Wte, sf, lo, hi)
                        band[b][te] = score(Wb) if Wb is not None else base[te]
                    for c in range(n_ch):
                        Wc = Wte.copy(); Wc[:, :, c, :] = 0.0
                        chan[c, te] = score(Wc)
                    for s in range(n_sub):
                        Ws = Wte.copy(); Ws[:, s, :, :] = 0.0
                        tim[s, te] = score(Ws)
                    del spec, m
                    if dev == "cuda":
                        torch.cuda.empty_cache()

                a0 = auc(y, base)
                rec = {"auc_intact": round(a0, 4),
                       "band_drop": {b: round(a0 - auc(y, band[b]), 4)
                                     for b in BANDS
                                     if _bandstop(W[:1], sf, *BANDS[b]) is not None},
                       "channel_drop": [round(a0 - auc(y, chan[c]), 4) for c in range(n_ch)],
                       "time_drop": [round(a0 - auc(y, tim[s]), 4) for s in range(n_sub)]}
                cd = np.array(rec["channel_drop"])
                rec["channel_concentration"] = round(float(cd.max() / (cd.sum() + 1e-9)), 4) \
                    if cd.sum() > 0 else None
                rec["top_band"] = max(rec["band_drop"], key=rec["band_drop"].get) \
                    if rec["band_drop"] else None
                task_res[mname] = rec
                resume.update_unit(unit, mname, rec)
                bd = "  ".join(f"{b}:{v:+.3f}" for b, v in rec["band_drop"].items())
                print(f"    {mname:18s} intact {a0:.3f} | {bd} | worst-channel "
                      f"{cd.max():+.3f}", flush=True)
            except Exception as e:  # noqa: BLE001
                task_res[mname] = {"error": f"{type(e).__name__}: {str(e)[:100]}"}
                print(f"    {mname:18s} FAILED {str(e)[:70]}", flush=True)
                if dev == "cuda":
                    torch.cuda.empty_cache()

        # do independently-designed architectures agree about which band carries the information?
        prof = {m: r["band_drop"] for m, r in task_res.items()
                if isinstance(r, dict) and r.get("band_drop")}
        if len(prof) >= 2:
            bands = sorted({b for v in prof.values() for b in v})
            M = np.array([[prof[m].get(b, 0.0) for b in bands] for m in prof])
            cons = {b: round(float(M[:, i].mean()), 4) for i, b in enumerate(bands)}
            agree = [m for m in prof if max(prof[m], key=prof[m].get) == max(cons, key=cons.get)]
            task_res["_consensus"] = {"mean_band_drop": cons,
                                      "consensus_band": max(cons, key=cons.get),
                                      "models_agreeing": len(agree), "models_compared": len(prof)}
            print(f"    consensus band: {max(cons, key=cons.get)} "
                  f"({len(agree)}/{len(prof)} models agree)", flush=True)
        task_res["_meta"] = {"n_clips": int(n_clip), "n_channels": int(n_ch), "sfreq": sf}
        results[task] = task_res

    rows = ["task,model,auc_intact," + ",".join(f"drop_{b}" for b in BANDS)
            + ",worst_channel_drop,channel_concentration,top_band"]
    for task, tr in results.items():
        for m, r in tr.items():
            if not isinstance(r, dict) or "band_drop" not in r:
                continue
            cd = max(r["channel_drop"]) if r["channel_drop"] else ""
            rows.append(f"{task},{m},{r['auc_intact']},"
                        + ",".join(str(r["band_drop"].get(b, "")) for b in BANDS)
                        + f",{cd},{r.get('channel_concentration','')},{r.get('top_band','')}")
    save_json({"note": "AUC drop when one band / channel / sub-window is removed from an "
                       "already-trained model, on the same held-out clips", "tasks": results},
              out / "results.json")
    (out / "interpretability.csv").write_text("\n".join(rows))
    print(f"\n-> {out}  ({len(rows) - 1} rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
