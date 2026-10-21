#!/usr/bin/env python
"""Protocol decomposition on one metric at a time: how much of a reported AUC is the fold, not the model.

Replaces run_transfer_gap.py for the manuscript. That script reported its two within-patient columns
as the MEDIAN PER-PATIENT AUC and its cross-patient column as a POOLED AUC, so part of each "gap" was a
change of metric: a pooled AUC across patients is depressed by between-patient offsets that a
per-patient AUC never sees. Here every protocol is scored on BOTH metrics, so each gap is read with
the metric held fixed, and every gap carries a subject-bootstrap CI.

Five protocols, one feature set (the 18-D vector), two classifiers (logistic, GBM):

  W-rand   within patient, clips split at random (the protocol much of the literature reports)
  W-event  within patient, one seizure/hour never straddles a fold
  P-seg    all patients pooled, clips split at random (patient AND event leakage)
  P-event  all patients pooled, event-grouped (patients shared across folds, events not)
  X        cross-patient, subject-grouped (the benchmark's protocol)

Typing has no within-patient protocol (a patient's discharge type is essentially fixed), so it is
scored under P-seg and X only -- the difference is exactly the leakage of patient identity.

-> results/protocol_decomp_<stamp>/results.json + table.csv
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
REPO = Path(__file__).resolve().parents[1]


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan


def per_subject_auc(y, p, g):
    return {s: _auc(y[g == s], p[g == s]) for s in np.unique(g) if len(np.unique(y[g == s])) == 2}


def main() -> int:
    from sklearn.base import clone
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from bench_data import load_task
    from bsf.utils.io import save_json
    from bsf.utils.provenance import new_results_dir

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default="scalp,ieeg,detect,typing")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--input", default="matched", choices=["matched", "first_run"],
                    help="matched = the 18-D vector on the networks' own input (default)")
    args = ap.parse_args()
    out = new_results_dir("protocol_decomp", seed=1234, config=vars(args))
    clfs = {"feat_logistic": make_pipeline(StandardScaler(), LogisticRegression(
                max_iter=2000, class_weight="balanced")),
            "feat_gbm": GradientBoostingClassifier(n_estimators=200, random_state=0)}
    R, rows = {}, ["task,model,protocol,pooled_auc,median_subject_auc,n_subjects_scored"]
    for task in [t for t in args.tasks.split(",") if t]:
        W, y, g, ev, X = load_task(task, "dev")
        if args.input == "matched" and task != "typing":
            # the 18-D vector on exactly the standardised sub-windows the networks receive (the
            # manuscript's reference arm); cached by run_feature_fairness.py
            from run_feature_fairness import _all_subwindows
            X = _all_subwindows(W, {"ieeg": 400.0}.get(task, 250.0), 6)
        del W
        splits = 5 if task == "typing" else 3
        R[task] = {}
        for mname, clf in clfs.items():
            oof = {}
            # pooled protocols
            folds = {
                "P-seg": StratifiedKFold(splits, shuffle=True, random_state=1234).split(X, y),
                "X": StratifiedGroupKFold(splits, shuffle=True, random_state=1234).split(X, y, g)}
            if task != "typing":
                folds["P-event"] = StratifiedGroupKFold(splits, shuffle=True,
                                                        random_state=1234).split(X, y, ev)
            for name, fs in folds.items():
                o = np.zeros(len(y))
                for tr, te in fs:
                    o[te] = clone(clf).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
                oof[name] = o
            # within-patient protocols: one model per patient; a patient too thin for 3 folds, or
            # with fewer than 3 events holding a positive, is not scored rather than imputed
            if task != "typing":
                for name in ("W-rand", "W-event"):
                    o = np.full(len(y), np.nan)
                    for s in np.unique(g):
                        m = np.where(g == s)[0]
                        ys, Xs, es = y[m], X[m], ev[m]
                        if np.bincount(ys, minlength=2).min() < splits:
                            continue
                        try:
                            it = (StratifiedKFold(splits, shuffle=True, random_state=0).split(Xs, ys)
                                  if name == "W-rand" else
                                  StratifiedGroupKFold(splits, shuffle=True, random_state=0)
                                  .split(Xs, ys, es))
                            os_ = np.full(len(ys), np.nan)
                            for tr, te in it:
                                if len(np.unique(ys[tr])) < 2:
                                    raise ValueError("single-class fold")
                                os_[te] = clone(clf).fit(Xs[tr], ys[tr]).predict_proba(Xs[te])[:, 1]
                            o[m] = os_
                        except Exception:  # noqa: BLE001
                            continue
                    oof[name] = o
            # score: pooled AUC over scored clips, median per-subject AUC; subjects common to all
            if task == "typing":
                # most typing patients carry one discharge class, so a per-patient AUC exists for
                # only a handful; the pooled AUC over every patient is the metric here
                common = list(np.unique(g))
            else:
                common = [s for s in np.unique(g) if all(np.isfinite(oof[k][g == s]).all() and
                                                          len(np.unique(y[g == s])) == 2 for k in oof)]
            mask = np.isin(g, common)
            res = {}
            for name, o in oof.items():
                ps = per_subject_auc(y[mask], o[mask], g[mask])
                res[name] = {"pooled_auc": round(_auc(y[mask], o[mask]), 4),
                             "median_subject_auc": round(float(np.nanmedian(list(ps.values()))), 4),
                             "per_subject": {k: round(v, 4) for k, v in ps.items()}}
                rows.append(f"{task},{mname},{name},{res[name]['pooled_auc']},"
                            f"{res[name]['median_subject_auc']},{len(common)}")
            # subject-bootstrap CIs for the gaps, both metrics, identical subject draws
            order = [k for k in ("W-rand", "W-event", "P-seg", "P-event", "X") if k in oof]
            pairs = [(a, b) for a, b in (("W-rand", "W-event"), ("W-event", "X"), ("P-seg", "P-event"),
                                         ("P-event", "X"), ("P-seg", "X"), ("W-rand", "X"))
                     if a in oof and b in oof]
            rng = np.random.default_rng(0)
            idx_by = {s: np.where(g == s)[0] for s in common}
            boots = {p: {"pooled": [], "median": []} for p in pairs}
            for _ in range(args.n_boot):
                draw = rng.choice(common, len(common), replace=True)
                idx = np.concatenate([idx_by[s] for s in draw])
                for a, b in pairs:
                    boots[(a, b)]["pooled"].append(_auc(y[idx], oof[a][idx]) - _auc(y[idx], oof[b][idx]))
                    da = [res[a]["per_subject"].get(s, np.nan) for s in draw]
                    db = [res[b]["per_subject"].get(s, np.nan) for s in draw]
                    boots[(a, b)]["median"].append(np.nanmedian(da) - np.nanmedian(db))
            gaps = {}
            for (a, b), d in boots.items():
                gaps[f"{a} - {b}"] = {
                    "pooled": [round(res[a]["pooled_auc"] - res[b]["pooled_auc"], 4),
                               round(float(np.nanpercentile(d["pooled"], 2.5)), 4),
                               round(float(np.nanpercentile(d["pooled"], 97.5)), 4)],
                    "median_subject": [round(res[a]["median_subject_auc"] - res[b]["median_subject_auc"], 4),
                                       round(float(np.nanpercentile(d["median"], 2.5)), 4),
                                       round(float(np.nanpercentile(d["median"], 97.5)), 4)]}
            R[task][mname] = {"protocols": res, "gaps": gaps, "subjects_scored": len(common),
                              "subjects_total": int(len(np.unique(g))), "order": order}
            line = " | ".join(f"{k} {res[k]['pooled_auc']:.3f}/{res[k]['median_subject_auc']:.3f}"
                              for k in order)
            print(f"[{task}] {mname} ({len(common)}/{len(np.unique(g))} subj)  pooled/median: {line}",
                  flush=True)
            for k, v in gaps.items():
                print(f"      gap {k:18s} pooled {v['pooled'][0]:+.3f} [{v['pooled'][1]:+.3f},"
                      f"{v['pooled'][2]:+.3f}]  median {v['median_subject'][0]:+.3f} "
                      f"[{v['median_subject'][1]:+.3f},{v['median_subject'][2]:+.3f}]", flush=True)
    save_json(R, out / "results.json")
    (out / "table.csv").write_text("\n".join(rows))
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
