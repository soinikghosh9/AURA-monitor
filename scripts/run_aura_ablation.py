"""Do augmentation and confidence-weighted pooling earn their place? Benchmark config throughout.

Arms isolate each change against the current best (anchor off, fixed encoder, no in-fold calibration),
whose recorded numbers are the baselines below. iEEG is included because it is AURA-Net's WORST task
(0.582, rank 18/25) and the one with the fewest subjects, so it is where an anti-overfitting change
should show up most and where a null result is most informative.
"""
import sys, warnings, json, time
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from run_unified_benchmark import _load_scalp, _load_detect, _load_ieeg, _load_typing
from bsf.models.auranet_train import AURANetClassifier
from bsf.eval.clinical import auc, brier_skill, calibration_error
from sklearn.model_selection import StratifiedGroupKFold
from bsf.utils.seeds import set_global_seeds

# recorded for the anchor-off, fixed-encoder model at the same 3 splits / 25 epochs
BASE = {"scalp": 0.679, "detect": 0.767, "ieeg": 0.582, "typing": 0.729}
SEEDS = (0, 1, 2)   # a single run cannot separate a 0.01 effect from a 0.017 harness drift
# Each arm removes ONE component from the frozen model, so a difference is attributable. Reported,
# not acted on: the architecture is frozen at results/_aura_freeze.json and anything these suggest is
# future work.
ARMS = (("frozen",     dict()),
        ("no_gate",    dict(gated_fusion=False)),   # additive fusion, the pre-freeze behaviour
        ("no_mixer",   dict(n_virtual=0)),          # the mixer assumes channel correspondence
        ("no_aug",     dict(augment=False)),
        ("mean_pool",  dict(subwin_pool="mean")))
# Resume from whatever previous runs banked. The JSON is written after every arm, so a power cut
# costs the arm in flight and nothing else -- without this the script restarts at the first task and
# silently recomputes hours of finished work.
OUT_JSON = Path("results/_auranet_improve_ab.json")
out = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
if out:
    print(f"resuming: {len(out)} arm-task combos already banked", flush=True)
# iEEG and typing first: they are the two tasks AURA-Net still loses, so they are where an ablation
# is informative rather than confirmatory.
for task, load, fs in (("ieeg", lambda: _load_ieeg(60, 6), 400.0),
                       ("typing", lambda: _load_typing(40), 500.0),
                       ("scalp", lambda: _load_scalp(60, 6), 250.0),
                       ("detect", lambda: _load_detect(60, 6), 250.0)):
    W, y, g, X = load()
    print(f"[{task}] {len(y)} clips / {len(np.unique(g))} subjects", flush=True)
    folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=1234)
                 .split(np.zeros(len(y)), y, g))
    for tag, kw in ARMS:
      key = f"{task}_{tag}"
      have = {r["seed"] for r in out.get(key, [])}
      if len(have) >= len(SEEDS):
          v = [r["auc"] for r in out[key]]
          print(f"{task:7s} {tag:10s} (resume) MEAN AUC {np.mean(v):.3f} +- {np.std(v):.3f}",
                flush=True)
          continue
      for seed in SEEDS:
        if seed in have:
            continue
        oof = np.zeros(len(y))
        for k, (tr, te) in enumerate(folds, 1):
            ts = time.time()
            set_global_seeds(seed)      # cudnn.deterministic, deterministic algos, CUBLAS config
            c = AURANetClassifier(fs=fs, epochs=25, device="cuda", seed=seed,
                                  d_feat=X.shape[1], **kw)
            c.fit(W[tr], y[tr], g[tr], feats=X[tr])
            oof[te] = c.predict_proba(W[te], g[te], feats=X[te])
            print(f"  [{task}/{tag}/s{seed}] fold {k}/3 {time.time()-ts:.0f}s", flush=True)
            del c
        a, b, e = auc(y, oof), brier_skill(y, oof), calibration_error(y, oof)
        print(f"{task:7s} {tag:10s} seed{seed} AUC {a:.3f}  BSS {b:+.3f}  ECE {e:.3f}", flush=True)
        out.setdefault(f"{task}_{tag}", []).append(
            {"seed": seed, "auc": round(a, 4), "bss": round(b, 4), "ece": round(e, 4)})
        np.save(f"results/_oof_ab_{task}_{tag}_s{seed}.npy", oof)
        json.dump(out, open(OUT_JSON, "w"), indent=2)
      v = [r["auc"] for r in out[f"{task}_{tag}"]]
      print(f"{task:7s} {tag:10s} MEAN AUC {np.mean(v):.3f} +- {np.std(v):.3f} over {len(v)} seeds",
            flush=True)
print("DONE", flush=True)
