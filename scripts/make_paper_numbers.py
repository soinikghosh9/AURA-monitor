#!/usr/bin/env python
"""Every quantitative statement in the manuscript, generated from the result artifacts.

The manuscript never types a result. It writes ``\\N{key}``, and this script writes
manuscript/generated/numbers.tex defining each key from a results file, plus the generated tables
(manuscript/generated/tab_*.tex) and numbers.json. Re-run after any experiment; ``check_latex.py``
fails if the text uses a key this script does not define. That makes "no number is hard-coded" a
property of the build rather than a promise.

Each source is read from the NEWEST provenance-stamped directory of its kind (smoke runs excluded),
except where a claim is historical and pinned to a named directory, which is stated at the pin.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
GEN = REPO / "manuscript" / "generated"
N: dict[str, str] = {}
MISSING: list[str] = []
TASKS = ["typing", "ieeg", "scalp", "detect"]
# established names: preictal-vs-interictal classification is "seizure prediction"; generalized-vs-focal
# IED classification; the splitting schemes follow the patient-specific / patient-independent usage
SPLIT_LABEL = {"W-rand": "PS-seg", "W-event": "PS-event", "P-seg": "Pool-seg", "P-event": "Pool-event",
               "X": "PI"}
TASK_NAME = {"typing": "IED classification", "ieeg": "iEEG seizure prediction",
             "scalp": "Scalp seizure prediction",
             "detect": "Scalp seizure detection"}
PRETTY = {"feat_logistic": "AURA + logistic", "feat_gbm": "AURA + GBM",
          "feat_logistic_sub": "AURA + logistic", "feat_gbm_sub": "AURA + GBM",
          "auranet_hybrid": "AURA-Net", "auranet_pure": "AURA-Net (raw only)",
          "deep_cnn_raw": "Repo CNN", "BIOT_pt": "BIOT (pretrained)", "CBraMod_pt": "CBraMod (pretrained)",
          "BIOT_scratch200": "BIOT (scratch, 200 Hz)", "CBraMod_scratch200": "CBraMod (scratch, 200 Hz)",
          "BIOT_probe": "BIOT probe (frozen)", "CBraMod_probe": "CBraMod probe (frozen)",
          "feat_logistic_base": "AURA + logistic, baseline-normalized",
          "feat_white_base": "AURA + logistic, baseline-whitened",
          "SCCNet_ea_base": "SCCNet, EA to baseline", "deep_cnn_raw_ea_base": "Repo CNN, EA to baseline",
          "ensemble": "AURA + networks (mean)"}


def pretty(m):
    return PRETTY.get(m, m)


def put(key, val, fmt="{:.3f}"):
    if val is None or (isinstance(val, float) and not math.isfinite(val)):
        MISSING.append(key)
        return
    if isinstance(val, (int, np.integer)):
        N[key] = str(int(val))
    elif isinstance(val, (float, np.floating)):
        N[key] = fmt.format(float(val))
    else:
        N[key] = str(val)


def signed(v, nd=3):
    return (f"{v:+.{nd}f}").replace("-", "$-$") if v is not None else None


def minus(v, nd=3):
    """A number for running text: a typographic minus, no plus sign."""
    return f"{v:.{nd}f}".replace("-", "$-$") if v is not None else None


def latest(pattern):
    from bsf.utils.io import latest_result_dirs
    ds = [d for d in latest_result_dirs(pattern, root=REPO / "results") if (d / "results.json").exists()]
    return ds[-1] if ds else None


def load(pattern):
    d = latest(pattern)
    return (json.loads((d / "results.json").read_text(encoding="utf-8")), d) if d else (None, None)


def table(name, header, rows, colspec, caption=None):
    GEN.mkdir(parents=True, exist_ok=True)
    body = ["\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        body.append("\\midrule" if r == "MID" else " & ".join(str(c) for c in r) + " \\\\")
    body.append("\\bottomrule")
    (GEN / f"tab_{name}.tex").write_text(
        f"\\begin{{tabular}}{{{colspec}}}\n" + "\n".join(body) + "\n\\end{tabular}\n", encoding="utf-8")


# ----------------------------------------------------------------------------------- sections
def design():
    seal = json.loads((REPO / "results/_blind_split.json").read_text())
    put("seal.sha", seal["sha256"][:12])
    put("seal.date", seal["sealed_utc"][:10])
    put("seal.testfrac", int(round(100 * seal["test_frac"])))
    c = seal["cohorts"]
    count = {"typing": ("vepiset_typing",), "ieeg": ("kaggle_ieeg", "canine_ieeg"),
             "scalp": ("scalp_chbmit", "scalp_siena"), "detect": ("scalp_chbmit", "scalp_siena")}
    for t, cs in count.items():
        put(f"seal.{t}.dev_listed", sum(len(c[k]["dev"]) for k in cs))
        put(f"seal.{t}.test_listed", sum(len(c[k]["test"]) for k in cs))
    # BIOT's six-dataset checkpoint was pretrained (without labels) on its authors' CHB-MIT training
    # split, chb01-chb20 (BIOT repository, datasets/CHB-MIT/process2.py): count our patients inside it
    biot_seen = {f"chb{i:02d}" for i in range(1, 21)}
    for side in ("dev", "test"):
        ids = [s.split("_")[-1].lower() for s in c["scalp_chbmit"][side]]
        put(f"design.biot_chb.{side}_seen", sum(i in biot_seen for i in ids))
        put(f"design.biot_chb.{side}_n", len(ids))
    from bsf.models.zoo import ALL_BRAINDECODE
    put("design.n_braindecode", len(ALL_BRAINDECODE))
    fm = [m for m in ALL_BRAINDECODE if m in ("BIOT", "EEGPT", "CBraMod")]
    put("design.n_fm_arch", len(fm))
    put("design.n_models", len(ALL_BRAINDECODE) + 1 + 2 + 2 + 2)   # + CNN, AURA-Net x2, 18-D x2, FM x2
    # development clip/subject counts and GBM size, from the benchmark OOF files
    from sklearn.ensemble import GradientBoostingClassifier
    for t in TASKS:
        f = sorted(glob.glob(str(REPO / f"results/_oof/unified_{t}_*.npz")))
        if f:
            with np.load(f[-1]) as z:
                put(f"dev.{t}.clips", len(z["_y"])); put(f"dev.{t}.subj", len(np.unique(z["_groups"])))
                put(f"dev.{t}.prev", float(z["_y"].mean()), "{:.2f}")
    bt, _ = load("blind_test_*")
    if bt:
        for t, v in bt["tasks"].items():
            m = v.get("_meta", {})
            put(f"test.{t}.clips", m.get("n_test")); put(f"test.{t}.subj", m.get("n_test_subjects"))
    # GBM size on the typing development set (node count of the fitted ensemble)
    try:
        sys.path.insert(0, str(REPO / "scripts"))
        from bench_data import load_task
        W, y, g, ev, X = load_task("typing", "dev"); del W
        gbm = GradientBoostingClassifier(n_estimators=200, random_state=0).fit(X, y)
        put("design.gbm_nodes", int(sum(t_.tree_.node_count for t_ in np.ravel(gbm.estimators_))))
    except Exception as e:  # noqa: BLE001
        print("gbm size:", e)


def final_stats():
    R, d = load("final_stats_*")
    if not R:
        return
    rows = []
    for t in TASKS:
        r = R.get(t)
        if not r:
            continue
        cs = r["contrasts"]
        put(f"fs.{t}.ref", pretty(r["reference_arm"])); put(f"fs.{t}.refauc", r["reference_auc"])
        put(f"fs.{t}.nmodels", r["n_models"]); put(f"fs.{t}.nsubj", r["n_subjects"])
        for m, v in cs.items():               # every model's mean development AUC, by name
            put(f"fs.{t}.auc.{m}", v["auc_mean"])
        deep = {m: v for m, v in cs.items() if not m.startswith("feat")}
        put(f"fs.{t}.ndeep", len(deep))
        put(f"fs.{t}.nbetter", len(r["sig_better_than_features"]))
        put(f"fs.{t}.nworse", len(r["sig_worse_than_features"]))
        lo = r["largest_open_advantage"]
        put(f"fs.{t}.openmodel", pretty(lo["model"])); put(f"fs.{t}.openhi", signed(lo["upper95"]))
        put(f"fs.{t}.opendelta", signed(lo["delta"]))
        best = max(deep, key=lambda m: deep[m]["auc_mean"])
        put(f"fs.{t}.bestdeep", pretty(best)); put(f"fs.{t}.bestdeepauc", deep[best]["auc_mean"])
        put(f"fs.{t}.bestdeepdelta", signed(deep[best]["delta"]))
        put(f"fs.{t}.bestdeeplo", signed(deep[best]["lo"])); put(f"fs.{t}.bestdeephi", signed(deep[best]["hi"]))
        put(f"fs.{t}.bestdeeppbh", deep[best]["p_bh"], "{:.2f}")
        put(f"fs.{t}.bestdeepwins", f"{deep[best]['subjects_model_wins']}/{deep[best]['subjects_scored']}")
        put(f"fs.{t}.aucmin", r["auc_range"][0]); put(f"fs.{t}.aucmax", r["auc_range"][1])
        put(f"fs.{t}.span", r["auc_range"][1] - r["auc_range"][0])
        put(f"fs.{t}.sdmodels", r["auc_sd_across_models"])
        aucs = sorted([v["auc_mean"] for v in cs.values()] + [r["reference_auc"]], reverse=True)
        put(f"fs.{t}.refrank", 1 + sum(a > r["reference_auc"] + 1e-12 for a in aucs))
        put(f"fs.{t}.seedsd", r["median_seed_sd"])
        rows.append([TASK_NAME[t], pretty(r["reference_arm"]), f"{r['reference_auc']:.3f}",
                     f"{pretty(best)} {deep[best]['auc_mean']:.3f}",
                     f"{signed(deep[best]['delta'])} [{signed(deep[best]['lo'])}, {signed(deep[best]['hi'])}]",
                     f"{len(r['sig_better_than_features'])}/{len(r['sig_worse_than_features'])}",
                     f"{signed(lo['upper95'])}"])
    table("paired", ["Task", "AURA classifier", "AUC", "Best learned model (mean AUC)", r"$\Delta$ [95\% CI]",
                     "Better/worse", "Max.\\ upper"], rows, "@{}lllllcc@{}")
    # seed volatility: the development leader's seeds
    for t in TASKS:
        sa = R.get(t, {}).get("seed_aucs", {})
        if sa:
            spreads = {m: max(v) - min(v) for m, v in sa.items() if len(v) >= 3 and not m.startswith("auranet")}
            if spreads:
                mx = max(spreads, key=spreads.get)
                put(f"seed.{t}.maxspread", spreads[mx]); put(f"seed.{t}.maxspreadmodel", pretty(mx))
                put(f"seed.{t}.maxspreadvals", ", ".join(f"{a:.3f}" for a in sa[mx]))
                put(f"seed.{t}.nseeded", len(spreads))
                put(f"seed.{t}.medsd", float(np.median([np.std(v, ddof=1) for m, v in sa.items()
                                                         if len(v) >= 3])))
    # development leader regression: the single-run top-1 vs its own seed mean
    allsd = [np.std(v, ddof=1) for t in TASKS for m, v in R.get(t, {}).get("seed_aucs", {}).items()
             if len(v) >= 3 and not m.startswith("auranet")]
    if allsd:
        put("seed.all.medsd", float(np.median(allsd))); put("seed.all.maxsd", float(np.max(allsd)))
        put("seed.all.n", len(allsd))


def protocol():
    R, d = load("protocol_decomp_*")
    if not R:
        return
    rows = []
    for t in ("scalp", "ieeg", "detect", "typing"):
        for m in ("feat_logistic", "feat_gbm"):
            r = R.get(t, {}).get(m)
            if not r:
                continue
            P = r["protocols"]
            for k, v in P.items():
                kk = k.replace("-", "")
                put(f"pd.{t}.{m}.{kk}.pool", v["pooled_auc"]); put(f"pd.{t}.{m}.{kk}.med", v["median_subject_auc"])
            for k, v in r["gaps"].items():
                kk = k.replace(" ", "").replace("-", "")
                put(f"pd.{t}.{m}.gap.{kk}.pool", signed(v["pooled"][0]))
                put(f"pd.{t}.{m}.gap.{kk}.poolci", f"[{signed(v['pooled'][1])}, {signed(v['pooled'][2])}]")
                put(f"pd.{t}.{m}.gap.{kk}.med", signed(v["median_subject"][0]))
            put(f"pd.{t}.{m}.nsubj", r["subjects_scored"])
            if t != "typing":
                g = r["gaps"]
                rows.append([TASK_NAME[t] if m == "feat_logistic" else "", pretty(m).replace("AURA + ", ""),
                             *[f"{P[k]['pooled_auc']:.3f}/{P[k]['median_subject_auc']:.3f}"
                               for k in ("W-rand", "W-event", "P-seg", "P-event", "X")],
                             f"{signed(g['W-rand - X']['pooled'][0])}"])
    table("protocol", ["Task", "Head", *[SPLIT_LABEL[k] for k in ("W-rand", "W-event", "P-seg", "P-event", "X")],
                       "PS-seg$-$PI"],
          rows, "@{}llcccccc@{}")
    # the widest gap anywhere, and the range of the full W-rand -> X drop
    drops = [R[t][m]["gaps"]["W-rand - X"]["pooled"][0] for t in ("scalp", "ieeg", "detect")
             for m in ("feat_logistic", "feat_gbm") if t in R and m in R[t]]
    if drops:
        put("pd.drop.min", signed(min(drops))); put("pd.drop.max", signed(max(drops)))
    segx = {(t, m): R[t][m]["gaps"]["P-seg - X"]["pooled"][0] for t in ("scalp", "ieeg", "detect")
            for m in ("feat_logistic", "feat_gbm") if t in R and m in R[t]}
    if segx:
        put("pd.segx.log.min", signed(min(v for (t, m), v in segx.items() if m == "feat_logistic")))
        put("pd.segx.log.max", signed(max(v for (t, m), v in segx.items() if m == "feat_logistic")))
        put("pd.segx.gbm.min", signed(min(v for (t, m), v in segx.items() if m == "feat_gbm")))
        put("pd.segx.gbm.max", signed(max(v for (t, m), v in segx.items() if m == "feat_gbm")))


def variants():
    R, d = load("variant_stats_*")
    if not R:
        return
    BP, _ = load("baseline_placement_*")
    # factorial
    rows = []
    for t in ("scalp", "ieeg", "detect", "typing"):
        f = (R.get("factorial") or {}).get(t) or {}
        if "variance_share" not in f:
            continue
        vs = f["variance_share"]
        put(f"fac.{t}.vprot", 100 * vs["protocol"], "{:.0f}"); put(f"fac.{t}.vmodel", 100 * vs["model"], "{:.0f}")
        put(f"fac.{t}.vres", 100 * vs["interaction_residual"], "{:.0f}")
        put(f"fac.{t}.nmodels", len(f["models"]))
        for p, v in f["inflation_vs_subject"].items():
            put(f"fac.{t}.infl.{p}.mean", signed(v["mean"])); put(f"fac.{t}.infl.{p}.min", signed(v["min"]))
            put(f"fac.{t}.infl.{p}.max", signed(v["max"]))
        for k, v in f["rank_spearman"].items():
            put(f"fac.{t}.rho.{k.replace('~', '')}", v, "{:.2f}")
        for p, v in f["spread_across_models"].items():
            put(f"fac.{t}.spread.{p}", v)
        rows.append([TASK_NAME[t], len(f["models"]),
                     *[(f["inflation_vs_subject"].get(p, {}).get("mean")) for p in ("segment", "event")],
                     f"{vs['protocol']*100:.0f}\\%", f"{vs['model']*100:.0f}\\%",
                     f"{vs['interaction_residual']*100:.0f}\\%"])
    rows = [[r[0], r[1], signed(r[2]) if r[2] is not None else "--", signed(r[3]) if r[3] is not None else "--",
             *r[4:]] for r in rows]
    table("factorial", ["Task", "Models", "Segment infl.", "Event infl.", "Scheme", "Model", "Residual"],
          rows, "@{}lcccccc@{}")
    # alignment
    arows = []
    for t in TASKS:
        a = (R.get("align") or {}).get(t) or {}
        for mode in ("ea_all", "ea_base"):
            s = a.get(f"summary_{mode}")
            if not s:
                continue
            put(f"al.{t}.{mode}.deepmean", signed(s["deep_mean_delta"]))
            put(f"al.{t}.{mode}.deepimp", f"{s['deep_improved']}/{s['deep_n']}")
            for fm, dv in s["feature_deltas"].items():
                put(f"al.{t}.{mode}.{fm}", signed(dv))
            put(f"al.{t}.{mode}.nsiggain", len(s["significant_gains"]))
            put(f"al.{t}.{mode}.nsigloss", len(s["significant_losses"]))
        ms = a.get("models", {})
        for m in ("feat_logistic", "feat_white", "feat_gbm"):
            if m in ms:
                v = ms[m]
                put(f"al.{t}.{m}.none", v.get("ea_base", v.get("ea_all", {})).get("auc_none"))
                for mode in ("ea_all", "ea_base"):
                    if mode in v:
                        put(f"al.{t}.{m}.{mode}", v[mode]["auc_aligned"])
                        put(f"al.{t}.{m}.{mode}.ci", f"[{signed(v[mode]['lo'])}, {signed(v[mode]['hi'])}]")
                        put(f"al.{t}.{m}.{mode}.pbh", v[mode].get("p_bh"), "{:.3f}")
        if a.get("summary_ea_base") or a.get("summary_ea_all"):
            sa, sb = a.get("summary_ea_all", {}), a.get("summary_ea_base", {})
            # the boosted vector with the baseline from the patient's earliest negatives instead of a
            # random third (run_baseline_placement.py); typing's baseline is separate background EEG
            early = ((BP or {}).get(t) or {}).get("early", {}).get("feat_gbm")
            # the networks with the earliest-negatives baseline (lane G; detection and scalp only)
            se = (((R.get("align_early") or {}).get(t) or {}).get("summary_ea_base"))

            def nets(s):
                return (f"{signed(s['deep_mean_delta'])} ({s['deep_improved']}/{s['deep_n']})"
                        if s else "--")
            # AURA-Net with its baseline layer for every patient (lane F; aura_anchor() runs first)
            arows.append([TASK_NAME[t], nets(sa), nets(sb), nets(se),
                          signed(sb.get("feature_deltas", {}).get("feat_logistic")) if sb else "--",
                          signed(sb.get("feature_deltas", {}).get("feat_gbm")) if sb else "--",
                          signed(early["delta"]) if early else "--",
                          N.get(f"aa.{t}.pure.delta", "--"), N.get(f"aa.{t}.hybrid.delta", "--")])
    table("align", ["Task", "Nets, EA all", "Nets, BN", "Nets, BN earliest",
                    "AURA log., BN", "AURA GBM, BN", "AURA GBM, BN earliest",
                    "AURA-Net raw, BN", "AURA-Net, BN"], arows,
          "@{}lcccccccc@{}")
    # the same alignment with the baseline from each patient's earliest negatives (lane G)
    for t_ in TASKS:
        ae = (R.get("align_early") or {}).get(t_) or {}
        se = ae.get("summary_ea_base")
        if se:
            put(f"ale.{t_}.deepmean", signed(se["deep_mean_delta"]))
            put(f"ale.{t_}.deepimp", f"{se['deep_improved']}/{se['deep_n']}")
            put(f"ale.{t_}.nsiggain", len(se["significant_gains"]))
            for m, v in ae.get("models", {}).items():
                if "ea_base" in v:
                    put(f"ale.{t_}.{m}.delta", signed(v["ea_base"]["delta"]))
                    put(f"ale.{t_}.{m}.ci", f"[{signed(v['ea_base']['lo'])}, {signed(v['ea_base']['hi'])}]")
                    put(f"ale.{t_}.{m}.none", v["ea_base"]["auc_none"])
                    put(f"ale.{t_}.{m}.base", v["ea_base"]["auc_aligned"])
    # recipe sensitivity
    for t in TASKS:
        rc = (R.get("recipe") or {}).get(t)
        if rc:
            put(f"rc.{t}.meanabs", rc["mean_abs_delta"]); put(f"rc.{t}.max", signed(rc["max_delta"]))
            put(f"rc.{t}.min", signed(rc["min_delta"]))
            # the best AUC any network reached under either alternative recipe
            alt = [(v[k], m) for m, v in rc["models"].items() for k in ("lr3e-4", "e50") if k in v]
            if alt:
                a, m = max(alt)
                put(f"rc.{t}.bestalt", a); put(f"rc.{t}.bestaltmodel", pretty(m))
    # pretraining
    prow = []
    for t in TASKS:
        p = (R.get("pretrain") or {}).get(t) or {}
        au = p.get("auc", {})
        for m, v in au.items():
            put(f"pt.{t}.{m}", float(np.mean(v)))
        for k, v in (p.get("contrasts") or {}).items():
            put(f"pt.{t}.{k}", signed(v["delta"])); put(f"pt.{t}.{k}.ci", f"[{signed(v['lo'])}, {signed(v['hi'])}]")
        if au:
            def g(m):
                return f"{np.mean(au[m]):.3f}" if m in au else "--"
            c = p.get("contrasts", {})
            prow.append([TASK_NAME[t], g("BIOT_pt"), g("BIOT_scratch200"),
                         signed(c["BIOT_pt_minus_scratch"]["delta"]) if "BIOT_pt_minus_scratch" in c else "--",
                         g("BIOT_probe"), g("CBraMod_pt"), g("CBraMod_scratch200"),
                         signed(c["CBraMod_pt_minus_scratch"]["delta"]) if "CBraMod_pt_minus_scratch" in c else "--",
                         g("CBraMod_probe")])
    table("pretrain", ["Task", "BIOT pt", "scratch", "$\\Delta$", "probe", "CBraMod pt", "scratch",
                       "$\\Delta$", "probe"], prow, "@{}lcccccccc@{}")


def calibration():
    R, d = load("calibration_xfit_*")
    if not R:
        return
    tot = R["total"]
    put("cal.n", tot["n"]); put("cal.raw", tot["usable_raw"]); put("cal.platt", tot["usable_platt"])
    put("cal.temp", tot["usable_temp"]); put("cal.icpt", tot["usable_intercept"]); put("cal.anti", tot["anti_calibrated"])
    rows = []
    for t in TASKS:
        s = R["summary"].get(t)
        if not s:
            continue
        for k in ("n", "usable_raw", "usable_platt", "usable_temp", "median_ece_raw", "median_ece_platt"):
            put(f"cal.{t}.{k}", s[k])
        rows.append([TASK_NAME[t], s["n"], s["usable_raw"], s["usable_temp"], s["usable_platt"],
                     f"{s['median_ece_raw']:.3f}", f"{s['median_ece_platt']:.3f}"])
    table("calib", ["Task", "Cells", "Raw", "Temperature", "Platt", "ECE raw", "ECE Platt"], rows,
          "@{}lcccccc@{}")


def fairness():
    R, d = load("feature_fairness_*")
    if not R:
        return
    for t, v in R.items():
        for m, x in v.items():
            put(f"ff.{t}.{m}.whole", x["whole_clip"]); put(f"ff.{t}.{m}.sub", x["same_subwindows"])


def blind():
    bt, d = load("blind_test_*")
    gaps, _ = load("blind_gaps_*")
    if not bt:
        return
    rows = []
    cur = {}
    if gaps:
        for t, v in gaps.get("tasks", {}).items():
            for m, x in v.items():
                if isinstance(x, dict) and "dev_auc_current" in x:
                    cur[(t, m)] = x["dev_auc_current"]
    csvf = d / "blind_leaderboard.csv"
    L = list(csv.DictReader(open(csvf))) if csvf.exists() else []
    for t in TASKS:
        rs = sorted([r for r in L if r["task"] == t], key=lambda r: -float(r["test_auc"]))
        if not rs:
            continue
        lead = rs[0]
        put(f"bl.{t}.leader", pretty(lead["model"])); put(f"bl.{t}.leaderauc", float(lead["test_auc"]))
        put(f"bl.{t}.leaderci", f"[{float(lead['test_auc_lo']):.2f}, {float(lead['test_auc_hi']):.2f}]")
        for m in ("feat_logistic", "feat_gbm"):
            r = next((x for x in rs if x["model"] == m), None)
            if r:
                put(f"bl.{t}.{m}", float(r["test_auc"]))
                put(f"bl.{t}.{m}.ci", f"[{float(r['test_auc_lo']):.2f}, {float(r['test_auc_hi']):.2f}]")
                put(f"bl.{t}.{m}.rank", rs.index(r) + 1)
        deep = [x for x in rs if not x["model"].startswith("feat")]
        put(f"bl.{t}.bestdeep", pretty(deep[0]["model"])); put(f"bl.{t}.bestdeepauc", float(deep[0]["test_auc"]))
        # optimism gap vs CURRENT development AUC (blind_gaps), median over models
        gs = [cur.get((t, x["model"]), float(x["dev_auc"]) if x["dev_auc"] else None) - float(x["test_auc"])
              for x in rs if cur.get((t, x["model"])) is not None or x["dev_auc"]]
        gs = [g for g in gs if g is not None]
        put(f"bl.{t}.gapmed", signed(float(np.median(gs)))); put(f"bl.{t}.gappos", f"{sum(g > 0 for g in gs)}/{len(gs)}")
        fl = next((x for x in rs if x["model"] == "feat_logistic"), None)
        fg = next((x for x in rs if x["model"] == "feat_gbm"), None)
        rows.append([TASK_NAME[t], f"{len(rs)}", f"{pretty(lead['model'])}{'$^\\ast$' if 'auranet' in lead['model'] else ''}",
                     f"{float(lead['test_auc']):.3f} [{float(lead['test_auc_lo']):.2f}, {float(lead['test_auc_hi']):.2f}]",
                     f"{float(fl['test_auc']):.3f}" if fl else "--", f"{float(fg['test_auc']):.3f}" if fg else "--",
                     signed(float(np.median(gs)))])
    table("blind", ["Task", "Models", "Leader", "Test AUC [95\\% CI]", "AURA log.", "AURA GBM",
                    "Median gap"], rows, "@{}lclcccc@{}")
    # the contamination record: AURA-Net's TEST before and after the architecture changes
    log1 = REPO / "results/_queue_bench/run_20260829T081140Z.log"
    if log1.exists():
        import re
        first = {}
        task = None
        for line in log1.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = re.match(r"\s+\[(\w+)\] DEV", line)
            if m:
                task = m.group(1); continue
            m = re.match(r"\s+(auranet_\w+)\s+TEST ([\d.]+)", line)
            if m:
                first[(task, m.group(1))] = float(m.group(2))
        for (t, m), v in first.items():
            put(f"bl.first.{t}.{m}", v)
            r = next((x for x in L if x["task"] == t and x["model"] == m), None)
            if r:
                put(f"bl.final.{t}.{m}", float(r["test_auc"]))
        put("bl.first.ntasks", len({t for t, _ in first}))


def addendum():
    R, d = load("blind_addendum_*")
    if not R:
        return
    rows = []
    for t in TASKS:
        v = R.get("tasks", {}).get(t, {})
        for m, x in v.items():
            if isinstance(x, dict) and "auc" in x:
                put(f"ad.{t}.{m}", x["auc"])
                if "ci" in x:
                    put(f"ad.{t}.{m}.ci", f"[{x['ci'][0]:.2f}, {x['ci'][1]:.2f}]")
        if v:
            def g(m):
                return f"{v[m]['auc']:.3f}" if m in v and isinstance(v[m], dict) and "auc" in v[m] else "--"
            rows.append([TASK_NAME[t], g("feat_logistic_sub"), g("feat_gbm_sub"), g("feat_logistic_base"),
                         g("feat_white_base"), g("BIOT_pt"), g("CBraMod_pt"), g("BIOT_probe"),
                         g("CBraMod_probe"), g("SCCNet_ea_base"), g("deep_cnn_raw_ea_base")])
    table("addendum", ["Task", "AURA log.", "AURA GBM", "AURA BN", "AURA BW", "BIOT pt",
                       "CBraMod pt", "BIOT probe", "CBraMod probe", "SCCNet EA", "CNN EA"], rows,
          "@{}l" + "c" * 10 + "@{}")


def streaming():
    from bsf.utils.io import latest_result_dirs
    for mode in ("detect", "forecast"):
        ds = [x for x in latest_result_dirs("streaming_monitor_*", root=REPO / "results")
              if (x / "results.json").exists()]
        pick = None
        for x in reversed(ds):
            R = json.loads((x / "results.json").read_text())
            ms = R.get("models", {})
            # analytic arms must carry the matched-input marker (the first v2 run fitted them on
            # clip-level vectors and scored standardised windows); a partial run with fewer than
            # the full model set is a test, not the table
            ok_feat = all(v.get("feature_input") == "standardised_8s_window" for k, v in ms.items()
                          if k.startswith("feat") and isinstance(v, dict) and "pooled" in v)
            if f"[{mode}]" in R.get("protocol", "") and ok_feat and len(ms) >= 6 and any(
                    isinstance(v, dict) and "recalibrated" in v.get("pooled", {}) for v in ms.values()):
                pick = R; break
        if not pick:
            continue
        rows = []
        put(f"st.{mode}.nstreams", len(pick["streams"]))
        evs = [v.get("n_events") for v in pick["models"].values() if isinstance(v, dict) and "n_events" in v]
        if evs:
            put(f"st.{mode}.nevents", evs[0])
        for m, v in pick["models"].items():
            if not isinstance(v, dict) or "pooled" not in v:
                continue
            pl = v["pooled"]; b10 = pl["clinical"].get("budget_10pct", {}); rc = pl["recalibrated"]
            put(f"st.{mode}.{m}.auc", pl["discrimination"]["auc"])
            put(f"st.{mode}.{m}.sens10", b10.get("event_sensitivity"), "{:.2f}")
            put(f"st.{mode}.{m}.bssraw", minus(rc["raw"]["brier_skill"], 2))
            put(f"st.{mode}.{m}.bssprior", minus(rc["prior_shift"]["brier_skill"]))
            put(f"st.{mode}.{m}.bssplatt", minus(rc["loso_platt"]["brier_skill"]))
            put(f"st.{mode}.prev", 100 * pl["prevalence"], "{:.1f}")
            rows.append([pretty(m), f"{pl['discrimination']['auc']:.3f}",
                         f"{b10.get('event_sensitivity', float('nan')):.2f}",
                         "yes" if b10.get("beats_chance") else "no",
                         minus(rc["raw"]["brier_skill"], 2), minus(rc["prior_shift"]["brier_skill"]),
                         minus(rc["loso_platt"]["brier_skill"])])
        # summaries: networks against the analytic arms, and the best Brier skill any correction reached
        ok = {m: v["pooled"] for m, v in pick["models"].items() if isinstance(v, dict) and "pooled" in v}
        deep = [p["discrimination"]["auc"] for m, p in ok.items() if not m.startswith("feat")]
        feat = [p["discrimination"]["auc"] for m, p in ok.items() if m.startswith("feat")]
        if deep and feat:
            put(f"st.{mode}.deepmin", min(deep)); put(f"st.{mode}.deepmax", max(deep))
            put(f"st.{mode}.featmin", min(feat)); put(f"st.{mode}.featmax", max(feat))
            put(f"st.{mode}.ndeep", len(deep))
            put(f"st.{mode}.ndeepabovefeat", sum(d > max(feat) for d in deep))
        bss = [p["recalibrated"][c]["brier_skill"] for p in ok.values() for c in ("prior_shift", "loso_platt")]
        put(f"st.{mode}.bssbest", minus(max(bss)))
        put(f"st.{mode}.nskilful", sum(max(p["recalibrated"][c]["brier_skill"]
                                            for c in ("prior_shift", "loso_platt")) > 0.01
                                        for p in ok.values()))
        put(f"st.{mode}.nmodels", len(ok))
        rows.sort(key=lambda r: -float(r[1]))
        table(f"stream_{mode}", ["Model", "AUC", "Sens.\\ @10\\% TiW", "Beats chance", "BSS raw",
                                 "BSS prior-shift", "BSS LOSO-Platt"], rows, "@{}lcccccc@{}")


def per_subject_worst():
    """Per-patient AUC for every model from the stored predictions: the case a mean hides."""
    from sklearn.metrics import roc_auc_score
    for t in ("scalp", "ieeg", "detect"):
        f = sorted(glob.glob(str(REPO / f"results/_oof/unified_{t}_*.npz")))
        if not f:
            continue
        with np.load(f[-1]) as z:
            y, g = z["_y"].astype(int), z["_groups"].astype(str)
            below, worst = [], []
            for k in z.files:
                if k.startswith("_") or k == "auranet_anchor":
                    continue
                p = z[k]
                a = [roc_auc_score(y[g == s], p[g == s]) for s in np.unique(g) if len(np.unique(y[g == s])) == 2]
                below.append(sum(v < 0.5 for v in a)); worst.append(min(a))
            n_sc = len([s for s in np.unique(g) if len(np.unique(y[g == s])) == 2])
        put(f"ws.{t}.nsubj", n_sc); put(f"ws.{t}.belowmin", min(below)); put(f"ws.{t}.belowmax", max(below))
        put(f"ws.{t}.belowmed", float(np.median(below)), "{:.0f}"); put(f"ws.{t}.worst", min(worst), "{:.2f}")


def occlusion():
    d = REPO / "results/interpretability_20260828T100235Z"   # the top-6-per-task sweep (pinned)
    R = json.loads((d / "results.json").read_text())
    rows = []
    for t in ("ieeg", "detect", "typing", "scalp"):
        v = R["tasks"].get(t, {})
        ms = [m for m in v if isinstance(v[m], dict) and "band_drop" in v[m]]
        tops = [max(v[m]["band_drop"], key=v[m]["band_drop"].get) for m in ms]
        from collections import Counter
        band, n = Counter(tops).most_common(1)[0]
        mean_drop = float(np.mean([v[m]["band_drop"][band] for m in ms]))
        hfo = float(np.mean([v[m]["band_drop"].get("hfo", np.nan) for m in ms]))
        put(f"oc.{t}.band", {"delta": "$\\delta$", "theta": "$\\theta$", "alpha": "$\\alpha$",
                             "beta": "$\\beta$", "gamma": "$\\gamma$", "hfo": "HFO"}[band])
        put(f"oc.{t}.agree", f"{n}/{len(ms)}"); put(f"oc.{t}.drop", mean_drop); put(f"oc.{t}.hfo", signed(hfo))
        rows.append([TASK_NAME[t], N[f"oc.{t}.band"], f"{n}/{len(ms)}", f"{mean_drop:.3f}", signed(hfo)])
    table("occlusion", ["Task", "Consensus band", "Agreement", "Mean drop", "HFO drop"], rows, "@{}llccc@{}")
    hf = [float(N[f"oc.{t}.hfo"].replace("$-$", "-")) for t in ("ieeg", "detect", "typing", "scalp")]
    put("oc.hfo.max", signed(max(hf)))


def within_subject():
    R, d = load("deep_fair_*")
    if R:
        m, pr = R["mean"], R["paired_rich_minus_deep"]
        put("ws.deep.grouped", m["grouped"]["deep_multichannel_raw"]); put("ws.feat.grouped", m["grouped"]["rich_logistic"])
        put("ws.deep.strat", m["stratified"]["deep_multichannel_raw"]); put("ws.feat.strat", m["stratified"]["rich_logistic"])
        put("ws.n", R["n_subjects"]); put("ws.delta", signed(-pr["grouped"]["mean_delta"]))
        put("ws.ci", f"[{signed(-pr['grouped']['ci95'][1])}, {signed(-pr['grouped']['ci95'][0])}]")
        put("ws.p", pr["grouped"]["wilcoxon_p"], "{:.3f}")
        put("ws.wins", f"{round((1 - pr['grouped']['frac_rich_better']) * R['n_subjects'])}/{R['n_subjects']}")


def auranet():
    f = REPO / "results/_auranet_fix_ab.json"
    if f.exists():
        A = json.loads(f.read_text())
        put("an.scalp.anchor", A["scalp_fixed"]["auc"]); put("an.scalp.noanchor", A["scalp_fixed_noanchor"]["auc"])
        put("an.detect.anchor", A["detect_fixed"]["auc"]); put("an.detect.noanchor", A["detect_fixed_noanchor"]["auc"])
        put("an.scalp.orig", A["scalp_fixed"]["baseline"][0])
    f = REPO / "results/_auranet_improve_ab.json"
    if f.exists():
        A = json.loads(f.read_text())
        rows = []
        arms = ["frozen", "no_gate", "no_mixer", "no_aug", "mean_pool"]
        label = {"frozen": "full (frozen)", "no_gate": "$-$ gated fusion", "no_mixer": "$-$ spatial mixer",
                 "no_aug": "$-$ augmentation", "mean_pool": "$-$ softmax pooling"}
        for a in arms:
            cells = []
            for t in ("ieeg", "typing", "scalp", "detect"):
                v = [x["auc"] for x in A.get(f"{t}_{a}", [])]
                if v:
                    cells.append(f"{np.mean(v):.3f}\\,{{\\tiny$\\pm${np.std(v, ddof=1):.3f}}}")
                    put(f"an.abl.{t}.{a}", float(np.mean(v)))
                    if a == "frozen":
                        put(f"an.abl.{t}.frozen.sd", float(np.std(v, ddof=1)))
                else:
                    cells.append("--")
            rows.append([label[a], *cells])
        table("auranet_ablation", ["Arm", "iEEG pred.", "IED", "Scalp pred.", "Detection"], rows, "@{}lcccc@{}")


CITE = {"EEGNet": "lawhern2018eegnet", "EEGTCNet": "ingolfsson2020eegtcnet", "SCCNet": "wei2019sccnet",
        "FBCNet": "mane2021fbcnet", "SincShallowNet": "borra2020sinc",
        "ShallowFBCSPNet": "schirrmeister2017deep", "Deep4Net": "schirrmeister2017deep",
        "EEGInceptionERP": "santamaria2020inception", "EEGNeX": "chen2024eegnex",
        "EEGSimpleConv": "elouahidi2023simpleconv", "EEGConformer": "song2023conformer",
        "ATCNet": "altaheri2023atcnet", "CTNet": "zhao2024ctnet", "MSVTNet": "liu2024msvtnet",
        "TIDNet": "kostas2020tidnet", "SPARCNet": "jing2023sparcnet", "ContraWR": "yang2023contrawr",
        "BIOT": "yang2023biot", "EEGPT": "wang2024eegpt", "CBraMod": "wang2025cbramod"}


def models_and_leaderboard():
    """Table S2 (models, family, parameters per task) and S3 (every model's development AUC)."""
    from bsf.models.zoo import BRAINDECODE_MODELS
    fam = {m: f for f, ms in BRAINDECODE_MODELS.items() for m in ms}
    params = {}
    for t in TASKS:
        for d in sorted(glob.glob(str(REPO / "results/unified_benchmark_*/leaderboard.csv"))):
            for r in csv.DictReader(open(d)):
                if r["task"] == t and r["model"] not in ("auranet_anchor",):
                    params.setdefault(r["model"], {})[t] = int(float(r["n_params"] or 0))
    for R_, dd in [load("variant_stats_*")]:
        pass
    rows = []
    def fmt(n):
        return f"{n/1e6:.1f}\\,M" if n >= 1e6 else (f"{n/1e3:.0f}\\,k" if n >= 1e3 else str(n))
    order = [m for fs_ in BRAINDECODE_MODELS.values() for m in fs_] + ["deep_cnn_raw", "auranet_pure",
                                                                      "auranet_hybrid"]
    for m in order:
        ps = params.get(m, {})
        cite = f"~\\cite{{{CITE[m]}}}" if m in CITE else ""
        rows.append([f"{pretty(m)}{cite}", fam.get(m, "ours" if m.startswith("auranet") else "repository").replace("_", " "),
                     *[fmt(ps[t]) if t in ps else "--" for t in TASKS]])
    table("models", ["Model", "Family", "IED", "iEEG pred.", "Scalp pred.", "Detect."], rows, "@{}llcccc@{}")
    for m in ("auranet_pure", "auranet_hybrid", "BIOT", "CBraMod", "EEGPT"):
        if params.get(m):
            put(f"par.{m}", f"{max(params[m].values()) / 1e6:.1f}\\,M" if max(params[m].values()) >= 1e6
                else f"{max(params[m].values()) / 1e6:.2f}\\,M")
    R, _ = load("final_stats_*")
    if not R:
        return
    rows = []
    allm = sorted({m for t in TASKS for m in R.get(t, {}).get("contrasts", {})} |
                  {R[t]["reference_arm"] for t in TASKS if t in R})
    def cell(t, m):
        r = R.get(t, {})
        if t == "typing" and m.endswith("_sub"):
            m = m[:-4]                  # a typing clip is one window: network input == whole clip
        if m == r.get("reference_arm"):
            return f"\\textbf{{{r['reference_auc']:.3f}}}"
        c = r.get("contrasts", {}).get(m)
        if not c:
            return "--"
        sd = f"$\\pm${c['auc_sd']:.3f}" if c.get("auc_sd") else ""
        mark = "$^{-}$" if m in r.get("sig_worse_than_features", []) else ("$^{+}$" if m in r.get("sig_better_than_features", []) else "")
        return f"{c['auc_mean']:.3f}{sd}{mark}"
    key = lambda m: -np.nanmean([R[t]["contrasts"][m]["auc_mean"] if m in R[t]["contrasts"] else np.nan for t in TASKS if t in R] or [0])
    for m in sorted(allm, key=key):
        if m in ("feat_logistic", "feat_gbm"):
            label = pretty(m) + " (first-run input)"
        elif m.endswith("_sub"):
            label = pretty(m) + " (network input)"
        else:
            label = pretty(m)
        rows.append([label, *[cell(t, m) if not (t == "typing" and m in ("feat_logistic", "feat_gbm"))
                              else "=" for t in TASKS]])
    table("leaderboard", ["Model", "IED", "iEEG pred.", "Scalp pred.", "Detection"], rows, "@{}lcccc@{}")


def identity():
    R, _ = load("identity_decomp_*")
    if not R:
        return
    for t, v in R.items():
        for inp, r in v.items():
            k = f"id.{t}.{inp}"
            put(f"{k}.pat", r["patient_share_median"], "{:.2f}")
            put(f"{k}.lab", r["label_share_median"], "{:.2f}")
            put(f"{k}.ratio", r["ratio_median"], "{:.1f}")
            if r["identity_acc"] is not None:
                put(f"{k}.acc", 100 * r["identity_acc"], "{:.0f}")
                put(f"{k}.chance", 100 * r["identity_chance"], "{:.0f}")
            put(f"{k}.n", r["n_patients"])


def scaling():
    R, _ = load("patient_scaling_*")
    if not R:
        return
    for t, v in R.items():
        for m, r in v.items():
            c = {int(k): x["mean"] for k, x in r["curve"].items()}
            ks = sorted(c)
            k = f"sc.{t}.{m}"
            put(f"{k}.kmin", ks[0]); put(f"{k}.kmax", ks[-1])
            put(f"{k}.aucmin", c[ks[0]]); put(f"{k}.aucmax", c[ks[-1]])
            put(f"{k}.slope", signed(r["slope_per_doubling"])); put(f"{k}.slopehi", signed(r["slope_upper_half"]))
    sl = [r["slope_per_doubling"] for v in R.values() for r in v.values()]
    put("sc.slope.min", signed(min(sl))); put("sc.slope.max", signed(max(sl)))


def stream_anchor():
    R, _ = load("stream_anchor_*")
    if not R:
        return
    put("sa.basemin", int(R["baseline_min"])); put("sa.nstreams", R["n_streams"]); put("sa.nevents", R["n_events"])
    for m, v in R["models"].items():
        for arm in ("none", "anchored"):
            put(f"sa.{m}.{arm}.auc", v[arm]["auc"])
            put(f"sa.{m}.{arm}.sens10", v[arm]["sens10"], "{:.2f}")
        put(f"sa.{m}.nimproved", v["n_streams_improved"]); put(f"sa.{m}.meandelta", signed(v["mean_stream_delta"]))
        put(f"sa.{m}.psf", v["p_signflip"], "{:.2f}")
        if "rolling" in v:      # exploratory: the reference refreshed from the preceding minutes
            put(f"sa.{m}.rolling.auc", v["rolling"]["auc"])
            put(f"sa.{m}.rolling.sens10", v["rolling"]["sens10"], "{:.2f}")
            put(f"sa.{m}.rolling.nimproved", v["rolling_n_improved"])
            put(f"sa.{m}.rolling.meandelta", signed(v["rolling_mean_delta"]))
            put(f"sa.{m}.rolling.psf", v["rolling_p_signflip"], "{:.2f}")
    if "rolling_min" in R:
        put("sa.rollmin", int(R["rolling_min"])); put("sa.gapmin", int(R["gap_min"]))


def clinical():
    """Threshold-based metrics (run_clinical_metrics.py): development thresholds cross-fitted over
    patients; test thresholds fixed on development predictions before the test predictions are read."""
    R, _ = load("clinical_metrics_*")
    if not R:
        return
    F = {"auc": "AUC", "sens": "Sens.", "spec": "Spec.", "ppv": "PPV", "npv": "NPV", "f1": "F1",
         "bal_acc": r"Bal.\ acc.", "mcc": "MCC"}

    def cell(v, k):
        x = v.get(k)
        if x is None or (isinstance(x, float) and not math.isfinite(x)):
            return "--"
        return minus(x, 3 if k == "auc" else 2)
    for split in ("dev", "test"):
        for t, models in R.get(split, {}).items():
            for m, v in models.items():
                for k in F:
                    if v.get(k) is not None and math.isfinite(v[k]):
                        put(f"cm.{split}.{t}.{m}.{k}", v[k], "{:.3f}" if k == "auc" else "{:.2f}")
                    ci = v.get("ci", {}).get(k)
                    if ci:
                        put(f"cm.{split}.{t}.{m}.{k}.ci", f"[{ci[0]:.2f}, {ci[1]:.2f}]")
                for k in ("tp", "fp", "tn", "fn", "n", "n_pos"):
                    if k in v:
                        put(f"cm.{split}.{t}.{m}.{k}", int(v[k]))
                if v.get("n"):
                    put(f"cm.{split}.{t}.{m}.prev", 100.0 * v["n_pos"] / v["n"], "{:.0f}")
    # development ranks of the AURA reference head by MCC and F1 among all models on identical clips, and
    # the share of patients each model ranks below chance (per-patient AUC < 0.5)
    FS, _ = load("final_stats_*")
    for t in TASKS:
        D = R.get("dev", {}).get(t, {})
        if not D or not FS or t not in FS:
            continue
        ref = FS[t]["reference_arm"]
        if ref not in D:
            ref = ref.replace("_sub", "")
        pool = {m: v for m, v in D.items() if not (t != "typing" and m in ("feat_logistic", "feat_gbm"))}
        for k in ("mcc", "f1", "bal_acc"):
            order = sorted(pool, key=lambda m: -pool[m][k] if pool[m][k] == pool[m][k] else 9)
            put(f"cm.dev.{t}.ref.{k}rank", order.index(ref) + 1 if ref in order else None)
            best = next(m for m in order if not m.startswith("feat"))
            put(f"cm.dev.{t}.bestnet.{k}", pool[best][k], "{:.2f}"); put(f"cm.dev.{t}.bestnet.{k}model", pretty(best))
        put(f"cm.dev.{t}.nmodels", len(pool)); put(f"cm.dev.{t}.ref", pretty(ref))
        PP = R.get("per_patient_auc", {}).get(t, {})
        if t != "typing" and ref in PP:
            v = list(PP[ref].values())
            put(f"pp.{t}.ref.below", int(sum(x < 0.5 for x in v))); put(f"pp.{t}.n", len(v))
            nets = [m for m in PP if not m.startswith("feat") and not m.endswith("_probe")]
            fr = [np.mean([x < 0.5 for x in PP[m].values()]) for m in nets if PP[m]]
            put(f"pp.{t}.netbelow.med", int(round(np.median(fr) * len(v))))
            put(f"pp.{t}.netbelow.max", int(round(np.max(fr) * len(v))))
    # held-out test: AURA, AURA with baseline normalization, and the two pretrained foundation models
    arms = ("feat_logistic_sub", "feat_gbm_sub", "feat_logistic_base", "BIOT_pt", "CBraMod_pt")
    rows = []
    for t in TASKS:
        T = R.get("test", {}).get(t, {})
        first = True
        for a in arms:
            if a not in T:
                continue
            v = T[a]
            nm = pretty(a).replace("AURA + logistic, baseline-normalized", "AURA + logistic, BN")
            rows.append([TASK_NAME[t] if first else "", nm, *[cell(v, k) for k in
                                                             ("auc", "sens", "spec", "ppv", "npv", "f1", "mcc")]])
            first = False
        if T and t != TASKS[-1]:
            rows.append("MID")
    table("clinical_test", ["Task", "Model", *[F[k] for k in ("auc", "sens", "spec", "ppv", "npv", "f1", "mcc")]],
          rows, "@{}ll" + "c" * 7 + "@{}")
    # development: the AURA heads, AURA-Net, the three best other networks and the pretrained models
    rows = []
    for t in TASKS:
        D = R.get("dev", {}).get(t, {})
        if not D:
            continue
        ref = ("feat_logistic", "feat_gbm") if t == "typing" else ("feat_logistic_sub", "feat_gbm_sub")
        fixed = [*ref, "auranet_hybrid", "auranet_pure", "BIOT_pt", "CBraMod_pt"]
        others = sorted([m for m in D if m not in fixed and not m.startswith("feat") and
                         not m.endswith("_probe") and not m.endswith("_scratch200")],
                        key=lambda m: -D[m]["auc"])[:3]
        first = True
        for m in [*ref, "auranet_hybrid", "auranet_pure", *others, "BIOT_pt", "CBraMod_pt"]:
            if m not in D:
                continue
            rows.append([TASK_NAME[t] if first else "", pretty(m), *[cell(D[m], k) for k in F]])
            first = False
        if t != TASKS[-1]:
            rows.append("MID")
    table("clinical_dev", ["Task", "Model", *F.values()], rows, "@{}ll" + "c" * 8 + "@{}")


def fmt_p(p):
    if p is None:
        return None
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def stream_replication():
    """Pre-registered replication of the refreshed baseline on every development stream."""
    R, _ = load("stream_replication_*")
    if not R:
        return
    put("sr.sha", R["manifest_sha256"][:12]); put("sr.nstreams", R["n_streams"])
    put("sr.nrep", R["n_replication"])
    put("sr.hours", int(round(sum(v["hours_scored"] for v in R["streams"].values()))))
    put("sr.npat.chb", sum(s.startswith("chbmit") for s in R["streams"]))
    put("sr.npat.siena", sum(s.startswith("siena") for s in R["streams"]))
    for m, v in R["models"].items():
        for st in ("replication", "original", "all"):
            b = v.get(st)
            if not b:
                continue
            k = f"sr.{st}.{m}"
            put(f"sr.{st}.nevents", b["n_events"]); put(f"sr.{st}.nstreams", b["n_streams"])
            for arm in ("none", "fixed", "refreshed"):
                put(f"{k}.{arm}.auc", b[arm]["auc"])
                put(f"{k}.{arm}.sens10", b[arm]["sens10_pooled"], "{:.2f}")
                for ev in ("raw", "smoothed"):
                    e = b[arm][f"events_{ev}"]
                    put(f"{k}.{arm}.{ev}.sens", e["sensitivity"], "{:.2f}")
                    put(f"{k}.{arm}.{ev}.caught", f"{e['caught']}/{e['n_events']}")
                    put(f"{k}.{arm}.{ev}.fa", e["fa_per_24h"], "{:.1f}")
                    put(f"{k}.{arm}.{ev}.lat", e["median_latency_s"], "{:.0f}")
            for arm in ("fixed", "refreshed"):
                d = b[f"delta_{arm}"]
                put(f"{k}.{arm}.delta", signed(d["mean"])); put(f"{k}.{arm}.median", signed(d["median"]))
                put(f"{k}.{arm}.nimp", d["n_improved"]); put(f"{k}.{arm}.n", d["n"])
                put(f"{k}.{arm}.psf", fmt_p(d["p_signflip"]))
    # stricter alarm budgets (descriptive): seizures detected and false alarms per 24 h, all streams,
    # 2-min causal average, threshold = the other streams' (1 - budget) quantile
    _, d = load("stream_replication_*")
    from run_stream_replication import events_at, smooth
    with np.load(d / "preds.npz", allow_pickle=True) as z:
        Y, G, T = z["_y"].astype(int), z["_groups"].astype(str), z["_t"].astype(float)
        subj = list(dict.fromkeys(G))
        for arm in ("none", "refreshed"):
            P = z[f"feat_gbm|{arm}"].astype(float)
            S = {s: smooth(P[G == s], T[G == s]) for s in subj}
            for tiw in (0.05, 0.01):
                cau = nev = fa = 0; hrs = 0.0
                for s in subj:
                    thr = float(np.quantile(np.concatenate([S[o] for o in subj if o != s]), 1 - tiw))
                    e = events_at(Y[G == s], S[s], T[G == s], thr, 8.0)
                    cau += e["caught"]; nev += e["n_events"]; fa += e["false_alarms"]
                    hrs += (G == s).sum() * 8 / 3600
                k = f"sr.op{int(round(tiw * 100))}.{arm}"
                put(f"{k}.caught", f"{cau}/{nev}"); put(f"{k}.fa", 24 * fa / hrs, "{:.0f}")


def stream_onset():
    """Feature departures from the preceding 30 min during seizures (run_stream_onset_profile.py)."""
    R, _ = load("stream_onset_*")
    if not R:
        return
    put("so.nseiz", R["n_seizures"]); put("so.nstreams", R["n_streams"])
    F = R["features"]
    for k, v in F.items():
        put(f"so.{k}.up", v["frac_above2"], "{:.2f}"); put(f"so.{k}.down", v["frac_below_m2"], "{:.2f}")
        put(f"so.{k}.median", signed(v["median_z"], 2))
    # similarity of seizure profiles: same patient against different patients (Pearson r of 18-D profiles)
    Z = np.array([p["z"] for p in R["per_seizure"]], float)
    S = np.array([p["subject"] for p in R["per_seizure"]])
    C = np.corrcoef(Z)
    iu = np.triu_indices(len(Z), 1)
    same = S[iu[0]] == S[iu[1]]
    put("so.r.within", float(np.median(C[iu][same])), "{:.2f}")
    put("so.r.between", float(np.median(C[iu][~same])), "{:.2f}")
    put("so.npairs.within", int(same.sum()))


def stream_personalize():
    """One-shot personalization from the first seizure (run_stream_personalize.py; exploratory)."""
    R, _ = load("stream_personalize_*")
    if not R:
        return
    put("sp.nstreams", R["n_streams"]); put("sp.nseiz", R["n_seizures_after"])
    for a, v in R["arms"].items():
        put(f"sp.{a}.median", v["median_stream_auc"]); put(f"sp.{a}.pooled", v["pooled_auc"])
        put(f"sp.{a}.delta", signed(v["delta_vs_population"])); put(f"sp.{a}.nimp", v["n_improved"])
        if v["p_signflip"] is not None:
            put(f"sp.{a}.psf", fmt_p(v["p_signflip"]))
        put(f"sp.{a}.caught", f"{v['caught']}/{v['n_events']}")
        put(f"sp.{a}.fa", v["fa_per_24h"], "{:.0f}")


TTA_ARMS = (("adabn_stream", "AdaBN, stream"), ("adabn_refreshed", "AdaBN, rolling"), ("tent", "TENT"),
            ("ea_refreshed", "EA, rolling"), ("probe_gbm_refreshed", "Embedding, rolling"))


def _stream_keys(prefix, models):
    """Shared key writer for the continuous-EEG passes: per model/arm AUC, events, and paired deltas."""
    for key, r in models.items():
        m, arm = key.split("|") if "|" in key else key
        k = f"{prefix}.{m}.{arm}"
        put(f"{k}.auc", r["auc"]); put(f"{k}.med", r.get("median_stream_auc"))
        if "caught" in r:
            put(f"{k}.caught", f"{r['caught']}/{r['n_events']}"); put(f"{k}.fa", r["fa_per_24h"], "{:.0f}")
        for dk, lab in (("delta_vs_none", "dnone"), ("delta_vs_aura_refreshed", "daura")):
            d = r.get(dk)
            if d:
                put(f"{k}.{lab}", signed(d["mean"]))
                put(f"{k}.{lab}.n", d.get("n_improved", d.get("n_better")))
                put(f"{k}.{lab}.psf", fmt_p(d["p_signflip"]))


def nn_tta():
    """Test-time adaptation of the networks in continuous EEG and what their embeddings encode."""
    f = REPO / "results" / "stream_tta" / "results.json"
    if not f.exists():
        return
    R = json.loads(f.read_text())
    flat = {f"{m}|{arm}": v for m, arms in R["models"].items() for arm, v in arms.items()}
    _stream_keys("tta", flat)
    put("tta.nstreams", R["n_streams"])
    # E7: identity and label decodability, averaged over folds
    agg = {}
    for key, e in R.get("E7", {}).items():
        if not e:
            continue
        m = key.split("|")[0]
        for who in ("net", "aura"):
            name = m if who == "net" else "aura"
            for q in ("id_acc", "chance", "label_auc_cv", "patient_share", "label_share"):
                agg.setdefault(name, {}).setdefault(q, []).append(e[who][q])
    for name, d in agg.items():
        for q, v in d.items():
            put(f"tta.e7.{name}.{q}", float(np.mean(v)), "{:.2f}")
    nets = [m for m in STREAM_NETS if m in agg]
    if nets:
        put("tta.e7.nets.id_min", min(np.mean(agg[m]["id_acc"]) for m in nets), "{:.2f}")
        put("tta.e7.nets.id_max", max(np.mean(agg[m]["id_acc"]) for m in nets), "{:.2f}")
        put("tta.e7.nets.lab_min", min(np.mean(agg[m]["label_auc_cv"]) for m in nets), "{:.2f}")
        put("tta.e7.nets.lab_max", max(np.mean(agg[m]["label_auc_cv"]) for m in nets), "{:.2f}")
    # table: none AUC and per-stream change for each adaptation
    rows = []
    for m in STREAM_NETS:
        a = R["models"].get(m, {})
        if "none" not in a:
            continue
        row = [pretty(m), f"{a['none']['auc']:.3f}"]
        for arm, _lab in TTA_ARMS:
            d = a.get(arm, {}).get("delta_vs_none")
            row.append("--" if not d else signed(d["mean"]) + ("$^\\ast$" if (d["p_signflip"] or 1) < 0.05 else ""))
        rows.append(row)
    for m, lab, arm in (("feat_gbm", "AURA + GBM", "refreshed"), ("feat_logistic", "AURA + logistic", "refreshed"),
                        ("auranet_hybrid", "AURA-Net", "refreshed")):
        a = R["models"].get(m, {})
        if "none" in a and arm in a:
            d = a[arm].get("delta_vs_none") or {}
            rows.append([lab, f"{a['none']['auc']:.3f}", "--", "--", "--", "--",
                         (signed(d.get("mean")) + ("$^\\ast$" if (d.get("p_signflip") or 1) < 0.05 else ""))
                         + " (features)" if d else "--"])
    if rows:
        table("tta", ["Model", "No adaptation", *[lab for _, lab in TTA_ARMS]], rows, "@{}lcccccc@{}")


def nn_stream_test():
    R, _ = load("stream_test_*")
    if not R:
        return
    put("tst.nstreams", R["n_streams"]); put("tst.sha", R["manifest"][:12])
    put("tst.nseiz", R["models"]["feat_gbm|none"].get("n_events", sum(v["n_seizures"] for v in R["streams"].values())))
    put("tst.hours", int(round(sum(v["hours"] for v in R["streams"].values()))))
    p = R["primary"]
    put("tst.primary.delta", signed(p["mean"])); put("tst.primary.nimp", p["n_improved"]); put("tst.primary.n", p["n"])
    put("tst.primary.psf", fmt_p(p["p_signflip"]))
    _stream_keys("tst", R["models"])
    _pass_table("stream_test", R["models"])
    pz = R.get("personalization")
    if pz:
        put("tst.pers.n", pz["n_streams"]); put("tst.pers.pop", pz["median_population"]); put("tst.pers.comb", pz["median_combined"])
        put("tst.pers.delta", signed(pz["mean_delta"])); put("tst.pers.nimp", pz["n_improved"]); put("tst.pers.psf", fmt_p(pz["p_signflip"]))


def _pass_table(name, models):
    """Pooled AUC per model and arm, with the per-stream change over the unadapted model in parentheses
    (* sign-flip p < 0.05). Shared by the held-out and external passes."""
    def cell(key):
        r = models.get(key)
        if not r:
            return "--"
        d = r.get("delta_vs_none")
        s = f"{r['auc']:.3f}"
        if d:
            s += f" ({signed(d['mean'], 2)}{'$^\\ast$' if (d['p_signflip'] or 1) < 0.05 else ''})"
        return s
    rows = []
    for m, lab, refr in (("feat_gbm", "AURA + GBM", "refreshed"), ("feat_logistic", "AURA + logistic", "refreshed"),
                         ("EEGNet", "EEGNet", "ea_refreshed"), ("SCCNet", "SCCNet", "ea_refreshed"),
                         ("BIOT", "BIOT", "ea_refreshed"), ("deep_cnn_raw", "Repo CNN", "ea_refreshed"),
                         ("auranet_hybrid", "AURA-Net", "refreshed"), ("ensemble", "AURA + networks", "refreshed")):
        if f"{m}|none" not in models:
            continue
        rows.append([lab, cell(f"{m}|none"), cell(f"{m}|{refr}"), cell(f"{m}|adabn_refreshed"), cell(f"{m}|tent"),
                     cell(f"{m}|probe_gbm_refreshed")])
    if rows:
        table(name, ["Model", "None", "Rolling reference", "AdaBN, rolling", "TENT", "Embedding, rolling"],
              rows, "@{}lccccc@{}")


def nn_seizeit2():
    R, _ = load("seizeit2_external_*")
    if not R:
        return
    # separate seizures as the event metrics count them (two seizures whose buffers touch stay two events)
    put("sz2.nstreams", R["n_streams"]); put("sz2.nseiz", R["models"]["feat_gbm|none"]["n_events"])
    put("sz2.hours", int(round(R["hours"])))
    put("sz2.sha", R["manifest"][:12])
    p = R["primary"]
    put("sz2.primary.delta", signed(p["mean"])); put("sz2.primary.median", signed(p.get("median")))
    put("sz2.primary.nimp", p["n_improved"]); put("sz2.primary.n", p["n"]); put("sz2.primary.psf", fmt_p(p["p_signflip"]))
    _stream_keys("sz2", R["models"])
    _pass_table("external", R["models"])
    pz = R.get("personalization")
    if pz:
        put("sz2.pers.n", pz["n_streams"]); put("sz2.pers.pop", pz["median_population"]); put("sz2.pers.comb", pz["median_combined"])
        put("sz2.pers.delta", signed(pz["mean_delta"])); put("sz2.pers.nimp", pz["n_improved"]); put("sz2.pers.psf", fmt_p(pz["p_signflip"]))


def nn_seizeit2_embedding():
    """Post-hoc: embedding-level refreshed normalization in SeizeIT2 (run_seizeit2_embedding.py)."""
    R, _ = load("seizeit2_embedding_*")
    if not R:
        return
    for m, arms in R["models"].items():
        for arm, v in arms.items():
            k = f"sz2e.{m}.{arm}"
            put(f"{k}.auc", v["auc"])
            d = v.get("delta_vs_none")
            if d:
                put(f"{k}.dnone", signed(d["mean"])); put(f"{k}.dnone.n", d["n_improved"])
                put(f"{k}.dnone.psf", fmt_p(d["p_signflip"]))


def nn_tuned():
    rows = []
    for task in ("typing", "detect"):
        R, _ = load(f"tuned_{task}_*")
        if not R:
            continue
        ref = R["aura_reference"]
        put(f"tun.{task}.aura", R["aura"][ref])
        for m, v in R["models"].items():
            k = f"tun.{task}.{m}"
            put(f"{k}.default", v["default_auc"]); put(f"{k}.nested", v["nested_auc"])
            d = v["nested_vs_aura"]
            put(f"{k}.daura", signed(d["delta"])); put(f"{k}.daura.ci", f"[{signed(d['lo'])}, {signed(d['hi'])}]")
            g = v["nested_vs_default"]
            put(f"{k}.gain", signed(g["delta"])); put(f"{k}.gain.ci", f"[{signed(g['lo'])}, {signed(g['hi'])}]")
            rows.append([TASK_NAME[task], pretty(m), f"{v['default_auc']:.3f}", f"{v['nested_auc']:.3f}",
                         f"{signed(g['delta'])} [{signed(g['lo'])}, {signed(g['hi'])}]",
                         f"{signed(d['delta'])} [{signed(d['lo'])}, {signed(d['hi'])}]"])
        rows.append([TASK_NAME[task], pretty(ref) + " (reference)", f"{R['aura'][ref]:.3f}", "--", "--", "--"])
        mx = max(v["nested_vs_aura"]["hi"] for v in R["models"].values())
        put(f"tun.{task}.maxhi", signed(mx))
        put(f"tun.{task}.nbetter", sum(v["nested_vs_aura"]["lo"] > 0 for v in R["models"].values()))
    if rows:
        table("tuned", ["Task", "Model", "Default", "Tuned (nested)", "Tuning gain [95\\% CI]",
                        "Tuned $-$ AURA [95\\% CI]"], rows, "@{}llcccc@{}")


def nn_montage():
    """Networks retrained in the 18-channel double banana, beside their benchmark (native montage) AUC.

    The native benchmark is seed 0, so the file-order vs double-banana table and its counts use seed 0 of
    each network (one entry per network however many seed runs exist); significance against the matched
    reference comes from the seed-averaged contrast below."""
    rows = []
    per_task: dict = {}
    for d in sorted((REPO / "results").glob("bench_variant_*")):
        pv = d / "provenance.json"
        if not pv.exists() or not (d / "results.json").exists():
            continue
        cfg = json.loads(pv.read_text()).get("config", {})
        if cfg.get("montage") != "db18":
            continue
        res = json.loads((d / "results.json").read_text())["results"]
        for key, v in res.items():
            if key.startswith("_") or "panel" not in v:
                continue
            m, _, seed = key.partition("@")
            if m == "feat_white" or seed not in ("", "s0"):
                continue
            per_task.setdefault(cfg["task"], {})[m] = v["panel"]["discrimination"]["auc"]
    for task, aucs in per_task.items():
        nets_db, nets_nat, feats_db, feats_nat, deltas = {}, {}, {}, {}, []
        for m, auc in aucs.items():
            put(f"mon.{task}.{m}", auc)
            nat = N.get(f"fs.{task}.auc.{m}")          # same computation in file-order channels
            if nat:
                put(f"mon.{task}.{m}.delta", signed(auc - float(nat)))
            (feats_db if m.startswith("feat_") else nets_db)[m] = auc
            if nat:
                (feats_nat if m.startswith("feat_") else nets_nat)[m] = float(nat)
                if not m.startswith("feat_"):
                    deltas.append(auc - float(nat))
            rows.append([TASK_NAME[task], pretty(m).replace(" (network input)", ""), nat or "--", f"{auc:.3f}",
                         signed(auc - float(nat)) if nat else "--"])
        if deltas:
            put(f"mon.{task}.dmin", signed(min(deltas))); put(f"mon.{task}.dmax", signed(max(deltas)))
            put(f"mon.{task}.nup", sum(x > 0 for x in deltas)); put(f"mon.{task}.nnets", len(deltas))
        if nets_db and feats_db:
            bn = max(nets_db, key=nets_db.get)
            put(f"mon.{task}.bestnet", pretty(bn)); put(f"mon.{task}.bestnet.auc", nets_db[bn])
            put(f"mon.{task}.bestfeat.auc", max(feats_db.values()))
            put(f"mon.{task}.gap", signed(nets_db[bn] - max(feats_db.values())))
        if nets_nat and feats_nat:
            put(f"mon.{task}.gapnative", signed(max(nets_nat.values()) - max(feats_nat.values())))
    # paired contrasts against the matched-input reference in the common montage (run_montage_contrast.py)
    C, _ = load("montage_contrast_*")
    for task, r in ((C or {}).get("tasks") or {}).items():
        put(f"mon.{task}.ref.auc", r["reference_auc"]); put(f"mon.{task}.ref.name", pretty(r["reference"]))
        nets = r["networks"]
        for m, v in nets.items():
            put(f"mon.{task}.{m}.dref", signed(v["delta"])); put(f"mon.{task}.{m}.dref.ci", f"[{signed(v['lo'])}, {signed(v['hi'])}]")
        if nets:
            put(f"mon.{task}.nbetter", sum(v["significantly_better"] for v in nets.values()))
            put(f"mon.{task}.maxhi", signed(max(v["hi"] for v in nets.values())))
            bm = max(nets, key=lambda k: nets[k]["delta"])
            put(f"mon.{task}.bestdelta", signed(nets[bm]["delta"])); put(f"mon.{task}.bestdelta.name", pretty(bm))
            put(f"mon.{task}.bestdelta.ci", f"[{signed(nets[bm]['lo'])}, {signed(nets[bm]['hi'])}]")
            ns = max(v.get("n_seeds", 1) for v in nets.values())
            put(f"mon.{task}.nseeds", ns)
            put(f"mon.{task}.allseedspos", sum(v.get("seeds_positive", int(v["delta"] > 0)) == v.get("n_seeds", 1) and v["delta"] > 0
                                               for v in nets.values()))
            sig = [m for m, v in nets.items() if v["significantly_better"]]
            put(f"mon.{task}.better.names", ", ".join(pretty(m) for m in sig) if sig else "none")
    if rows:
        table("montage", ["Task", "Network", "File-order channels", "Double banana", "$\\Delta$"], rows, "@{}llccc@{}")


STREAM_NETS = ("EEGNet", "SCCNet", "BIOT", "deep_cnn_raw")
STREAM_MODELS = ("feat_gbm", "feat_logistic", *STREAM_NETS, "ensemble")


def stream_nets():
    """Networks and AURA streamed on the same windows, with and without a refreshed baseline."""
    f = REPO / "results" / "stream_nets" / "results.json"
    if not f.exists():
        return
    R = json.loads(f.read_text())
    put("sn.nstreams", R["n_streams"])
    rows = []
    for m in STREAM_MODELS:
        v = R["models"].get(m)
        if not v:
            continue
        for arm in ("none", "refreshed"):
            a, e = v[arm], v[arm]["events_smoothed"]
            put(f"sn.{m}.{arm}.auc", a["auc"]); put(f"sn.{m}.{arm}.sens", e["sensitivity"], "{:.2f}")
            put(f"sn.{m}.{arm}.fa", e["fa_per_24h"], "{:.1f}")
            put(f"sn.{m}.{arm}.lat", e["median_latency_s"], "{:.0f}")
            put(f"sn.nevents", e["n_events"])
            rows.append([pretty(m) if arm == "none" else "", "rolling" if arm == "refreshed" else "none",
                         f"{a['auc']:.3f}", f"{e['caught']}/{e['n_events']} ({e['sensitivity']:.2f})",
                         f"{e['fa_per_24h']:.0f}"])
        d = v.get("delta_refreshed")
        if d:
            put(f"sn.{m}.delta", signed(d["mean"])); put(f"sn.{m}.nimp", d["n_improved"])
            put(f"sn.{m}.psf", fmt_p(d["p_signflip"]))
    nets = [m for m in STREAM_NETS if m in R["models"]]
    if nets:
        put("sn.nets.none.min", min(R["models"][m]["none"]["auc"] for m in nets))
        put("sn.nets.none.max", max(R["models"][m]["none"]["auc"] for m in nets))
        put("sn.nets.refreshed.min", min(R["models"][m]["refreshed"]["auc"] for m in nets))
        put("sn.nets.refreshed.max", max(R["models"][m]["refreshed"]["auc"] for m in nets))
        put("sn.nets.nimp", sum(R["models"][m]["delta_refreshed"]["mean"] > 0 for m in nets))
        put("sn.nets.n", len(nets))
    for key, d in R.get("vs_aura_refreshed", {}).items():
        k = "sn.vs." + key.replace("|", ".")
        put(f"{k}.delta", signed(d["mean"])); put(f"{k}.nbetter", d["n_better"])
        put(f"{k}.psf", fmt_p(d["p_signflip"]))
    if rows:
        table("stream_nets", ["Model", "Baseline", "AUC", "Seizures detected", "False alarms/24\\,h"],
              rows, "@{}llccc@{}")


def aura_anchor():
    """AURA-Net with a working anchor vs without, paired on identical clips (lane F)."""
    sd = REPO / "results" / "_aura_anchor"
    st = sd / "state.json"
    if not st.exists():
        return
    S = json.loads(st.read_text())
    from run_final_stats import hierarchical_contrast, signflip
    from run_variant_stats import _per_subject_delta
    rows = []
    for t in TASKS:
        yf, gf = sd / f"{t}_y.npy", sd / f"{t}_g.npy"
        if not yf.exists():
            continue
        y, g = np.load(yf), np.load(gf, allow_pickle=True).astype(str)
        for cfg in ("pure", "hybrid"):
            P = {arm: [np.load(sd / f"{t}_{cfg}_{arm}_s{s_}.npy") for s_ in (0, 1, 2)
                       if (sd / f"{t}_{cfg}_{arm}_s{s_}.npy").exists()] for arm in ("none", "anchored")}
            if len(P["none"]) < 3 or len(P["anchored"]) < 3:
                continue
            k = f"aa.{t}.{cfg}"
            a_none = [S[f"{t}|{cfg}|none|s{s_}"]["auc"] for s_ in (0, 1, 2)]
            a_anch = [S[f"{t}|{cfg}|anchored|s{s_}"]["auc"] for s_ in (0, 1, 2)]
            put(f"{k}.none", float(np.mean(a_none))); put(f"{k}.anchored", float(np.mean(a_anch)))
            r = hierarchical_contrast(y, g, P["anchored"], np.mean(P["none"], 0), n_boot=2000)
            put(f"{k}.delta", signed(r["delta"])); put(f"{k}.ci", f"[{signed(r['lo'])}, {signed(r['hi'])}]")
            put(f"{k}.p", r["p_two_sided"], "{:.3f}")
            if t != "typing":
                put(f"{k}.psf", signflip(_per_subject_delta(y, g, np.mean(P["anchored"], 0),
                                                            np.mean(P["none"], 0))), "{:.3f}")
            rows.append([TASK_NAME[t], "AURA-Net" if cfg == "hybrid" else "AURA-Net (raw input)",
                         f"{np.mean(a_none):.3f}", f"{np.mean(a_anch):.3f}", signed(r["delta"]),
                         f"[{signed(r['lo'])}, {signed(r['hi'])}]"])
    if rows:
        table("aura_anchor", ["Task", "Configuration", "Without BN", "With BN", "$\\Delta$", "95\\% CI"],
              rows, "@{}llcccc@{}")


def placement():
    """Baseline-referenced alignment with the baseline drawn at random vs from the earliest clips."""
    R, _ = load("baseline_placement_*")
    if not R:
        return
    for t, v in R.items():
        for design in ("random", "early"):
            for m in ("feat_logistic", "feat_gbm"):
                r = (v.get(design) or {}).get(m)
                if not r:
                    continue
                k = f"bp.{t}.{design}.{m}"
                put(f"{k}.none", r["auc_none"]); put(f"{k}.base", r["auc_aligned"])
                put(f"{k}.delta", signed(r["delta"]))
                put(f"{k}.ci", f"[{signed(r['lo'])}, {signed(r['hi'])}]")
                put(f"{k}.p", r["p_two_sided"], "{:.3f}"); put(f"{k}.psf", r["p_signflip"], "{:.3f}")
                put(f"{k}.wins", r["subjects_model_wins"]); put(f"{k}.nsub", r["subjects_scored"])


def parta():
    """Pre-seal within-patient analyses (supplement), via the audit registry's own extractors."""
    import importlib
    A = importlib.import_module("audit_manuscript_numbers")
    spec = {
        "ladder.slow": A.within_subject_ladder("slow"), "ladder.richphys": A.within_subject_ladder("rich_plus"),
        "kaggle.grouped": A.clip_leak("rich", "grouped_cv"), "kaggle.shuffle": A.clip_leak("rich", "shuffle_cv"),
        "cap.env": A.capacity("envelope+logistic"), "cap.rich": A.capacity("rich+logistic"),
        "cap.mlp": A.capacity("rich+MLP"),
        "rep.delta": A.contrast("envelope->rich (representation)"),
        "capd.delta": A.contrast("rich->rich+MLP (capacity)"),
        "hz.1": A.horizon(1), "hz.3": A.horizon(3), "hz.24": A.horizon(24),
        "hzfdr.1": A.horizon_fdr(1), "hzfdr.6": A.horizon_fdr(6),
        "cont.iso": A.cont_eval(["aggregate", "ssm_auc"]),
        "cont.bssplatt": A.cont_eval(["aggregate", "bss_platt_noheldout"]),
        "cont.bssiso": A.cont_eval(["aggregate", "bss_isotonic_heldout"]),
        "cont.sens10": A.cont_eval(["aggregate", "sens_at_10pct_tiw"]),
        "cycle.multidien": A.cycle_daily("multidien"), "tpp.24": A.tpp_mean(24), "tpp.1": A.tpp_mean(1),
        "typ.refwhiten": A.typing("gen_vs_focal", "refwhiten_logistic"),
        "typ.zscore": A.typing("gen_vs_focal", "interictal_zscore"),
        "typ.pooled": A.typing("gen_vs_focal", "vanilla_logistic"),
        "fs.scalp.conv": A.multitask("refwhiten+few-shot (ours)", modality="scalp"),
        "fs.scalp.causal": A.causal_fewshot("scalp_causal_fewshot"),
        "zs.scalp.causal": A.causal_fewshot("scalp_causal_zeroshot"),
        "fs.ieeg.conv": A.multitask("refwhiten+few-shot (ours)", modality="iEEG"),
        "fs.ieeg.causal": A.causal_fewshot("ieeg_eventdisjoint_fewshot"),
        "hth.first": A.headtohead("patient_adaptive_1st"), "hth.second": A.headtohead("aura_refwhiten_2nd"),
        "hth.zero": A.headtohead("zero_shot_prior"),
        "pac.full": A.pac_control("stratified", "rich_physio_pac_full"),
        "pac.cap": A.pac_control("stratified", "rich_physio_pac_cap60s"),
        "pac.fullg": A.pac_control("grouped", "rich_physio_pac_full"),
        "pac.capg": A.pac_control("grouped", "rich_physio_pac_cap60s"),
        "null.real": A.deepnull("deep_real"), "null.perm": A.deepnull("deep_perm_null_mean"),
        "null.lp45": A.deepnull("deep_lowpass45"),
        "hum.slow": A.human_ieeg("slow"), "hum.rich": A.human_ieeg("rich"),
    }
    for k, f in spec.items():
        try:
            v = f()
        except Exception:  # noqa: BLE001
            v = None
        put(f"pa.{k}", v, "{:.3f}" if not isinstance(v, int) else "{}")


# ----------------------------------------------------------------------------------- AURA-Ens and the monitor
def nn_aura_ens():
    """AURA-Ens on the common-montage development streams (run_aura_ens.py): R1/R2/R3/R4."""
    f = REPO / "results" / "aura_ens_db18" / "results.json"
    if not f.exists():
        return
    R = json.loads(f.read_text())
    put("ae.nstreams", R["n_streams"])
    for m, arms in R["models"].items():
        for arm, v in arms.items():
            k = f"ae.{m}.{arm}"
            put(f"{k}.auc", v["auc"]); put(f"{k}.med", v["median_stream_auc"])
            for tag, dk in (("dnone", "delta_vs_none"), ("daura", "delta_vs_aura_refreshed"), ("dpost", "delta_vs_posthoc"),
                            ("dpbn", "delta_vs_pbn")):
                d = v.get(dk)
                if d:
                    put(f"{k}.{tag}", signed(d["mean"])); put(f"{k}.{tag}.n", d.get("n_better", d.get("n_improved")))
                    put(f"{k}.{tag}.psf", fmt_p(d["p_signflip"]))
            for b, e in v.get("szcore", {}).items():
                put(f"{k}.b{b}.sens", e["sensitivity"], "{:.2f}"); put(f"{k}.b{b}.fa", e["fa_per_24h"], "{:.1f}")
                put(f"{k}.b{b}.f1", e["f1"], "{:.2f}"); put(f"{k}.b{b}.det", f"{e['detected']}/{e['n_ref']}")
                if e.get("precision") is not None:
                    put(f"{k}.b{b}.prec", e["precision"], "{:.2f}")
                if e.get("median_latency_s") is not None:
                    put(f"{k}.b{b}.lat", e["median_latency_s"], "{:.0f}")
    nets = [m for m in R["models"] if m not in ("feat_gbm", "feat_logistic", "auranet_hybrid", "ens_mean", "aura_ens")]
    put("ae.nnets", len(nets))
    pb = [R["models"][m]["pbn"]["delta_vs_posthoc"] for m in nets
          if "pbn" in R["models"][m] and "delta_vs_posthoc" in R["models"][m]["pbn"]]
    put("ae.pbn.nbetter", sum(d["mean"] > 0 for d in pb))
    put("ae.pbn.nsig", sum(d["mean"] > 0 and (d["p_signflip"] or 1) < 0.05 for d in pb))
    put("ae.pbn.nworse", sum(d["mean"] < 0 and (d["p_signflip"] or 1) < 0.05 for d in pb))
    adv = [m for m in nets if "pbn" in R["models"][m] and "pbn_adv" in R["models"][m]]
    put("ae.adv.n", len(adv))
    put("ae.adv.nbetter", sum(R["models"][m]["pbn_adv"]["auc"] > R["models"][m]["pbn"]["auc"] for m in adv))
    ph = [R["models"][m]["probe_gbm_refreshed"]["delta_vs_none"] for m in nets if "probe_gbm_refreshed" in R["models"][m]]
    put("ae.post.nbetter", sum(d["mean"] > 0 for d in ph))
    put("ae.post.nsig", sum(d["mean"] > 0 and (d["p_signflip"] or 1) < 0.05 for d in ph))
    e7 = {}
    for key, v in R.get("E7", {}).items():
        if v:
            e7.setdefault(key.split("|")[0], []).append(v["net"]["id_acc"])
    for m, v in e7.items():
        put(f"ae.e7.{m}.id", float(np.mean(v)), "{:.2f}")
    rows = []

    def row(name, m, arm):
        v = R["models"].get(m, {}).get(arm)
        if not v:
            return
        d = v.get("delta_vs_aura_refreshed")
        e6, e24 = v["szcore"].get("6", {}), v["szcore"].get("24", {})
        rows.append([name, f"{v['auc']:.3f}", f"{v['median_stream_auc']:.3f}",
                     (f"{signed(d['mean'])} ({d['n_better']}/{R['n_streams']}, $p$={fmt_p(d['p_signflip'])})" if d else "reference"),
                     f"{e6.get('sensitivity', float('nan')):.2f} / {e6.get('fa_per_24h', float('nan')):.1f}",
                     f"{e24.get('sensitivity', float('nan')):.2f} / {e24.get('fa_per_24h', float('nan')):.1f}"])
    row("AURA + GBM, rolling reference", "feat_gbm", "refreshed"); row("AURA + GBM, raw", "feat_gbm", "none")
    row("AURA-Net, rolling reference", "auranet_hybrid", "refreshed"); rows.append("MID")
    for m in nets:
        row(f"{pretty(m)}", m, "none"); row(f"{pretty(m)}, embedding, rolling reference", m, "probe_gbm_refreshed")
        row(f"{pretty(m)}, PBN", m, "pbn")
    rows.append("MID")
    for arm, name in (("post", "post hoc"), ("pbn", "PBN"), ("all", "both")):
        row(f"Mean ensemble ({name})", "ens_mean", arm); row(f"AURA-Ens, stacked ({name})", "aura_ens", arm)
    if rows:
        table("aura_ens", ["Model / adaptation", "Pooled AUC", "Median stream AUC", "$-$ AURA, rolling ref. (better, $p$)",
                           "Sens. / FA at 6/24\\,h", "Sens. / FA at 24/24\\,h"], rows, "@{}lccccc@{}")


def nn_monitor():
    """AURA Monitor modules on development patients: IED, fingerprints, risk, personalization loop."""
    R, _ = load("ied_monitor_*")
    if R:
        put("ied.nep", f"{R['n_epochs']:,}"); put("ied.nied", f"{R['n_ied']:,}"); put("ied.npat", R["n_patients"])
        for k, v in R["detection"].items():
            put(f"ied.{k}.auc", v["auc"]); put(f"ied.{k}.auprc", v["auprc"]); put(f"ied.{k}.med", v["median_patient_auc"])
            if "sens_at_1fp_per_min" in v:
                put(f"ied.{k}.sens", v["sens_at_1fp_per_min"], "{:.2f}")
            if "burden_spearman" in v:
                put(f"ied.{k}.rho", v["burden_spearman"], "{:.2f}")
        put("ied.gen.auc", R["typing"]["generalized_vs_focal_auc"], "{:.2f}")
        put("ied.lobe.bacc", R["lobe"]["balanced_acc"], "{:.2f}"); put("ied.lobe.auc", R["lobe"]["macro_auc"], "{:.2f}")
        put("ied.lobe.map", R["lobe"]["map_region_balanced_acc"], "{:.2f}")
        if "monitor_arm" in R:
            put("ied.arm", R["monitor_arm"].replace("_", " "))
        if "typing_gated" in R:
            put("ied.gated.gen", R["typing_gated"]["generalized_vs_focal_auc"], "{:.2f}")
            put("ied.gated.lobe", R["typing_gated"]["lobe_balanced_acc"], "{:.2f}")
    R, _ = load("seizure_patterns_*")
    if R:
        st = R["stereotypy"]
        put("pat.nsz", R["n_seizures"]); put("pat.npat", R["n_patients"])
        put("pat.within", st["within"], "{:.2f}"); put("pat.between", st["between"], "{:.2f}"); put("pat.p", fmt_p(st["p_perm"]))
        nt = [v["n_types"] for v in R["types"].values() if v["n_seizures"] >= 2]
        put("pat.multitype", sum(n > 1 for n in nt)); put("pat.ntyped", len(nt))
        la = R["lateralization"]
        put("pat.lat.n", la["n_seizures"]); put("pat.lat.calls", la["n_lateralized_calls"]); put("pat.lat.correct", la["correct"])
        put("pat.lat.abstain", 100 * la["abstain_rate"], "{:.0f}")
        calls = la["calls"]
        fc = sum((c["index"] > 0) == (c["truth"] == "left") for c in calls)
        from scipy.stats import binomtest
        put("pat.lat.forced", f"{fc}/{len(calls)}")
        put("pat.lat.forced.p", fmt_p(binomtest(fc, len(calls), 0.5, alternative="greater").pvalue))
    R, _ = load("risk_monitor_*")
    if R:
        put("risk.npat", R["n_patients"]); put("risk.hours", R["hours"], "{:.0f}"); put("risk.nsz", R["n_seizures"])
        for H, v in R["horizons"].items():
            for k in ("clock", "aura_risk"):
                put(f"risk.h{H}.{k}.bss", signed(v[k]["brier_skill"])); put(f"risk.h{H}.{k}.auc", v[k]["auc"])
                put(f"risk.h{H}.{k}.med", v[k]["median_patient_auc"]); put(f"risk.h{H}.{k}.sens", v[k]["sens_at_20pct_tiw"], "{:.2f}")
                put(f"risk.h{H}.{k}.p", fmt_p(v[k]["p_vs_chance"]))
    R, _ = load("personalize_loop_*")
    if R:
        for b, v in R["budgets"].items():
            k = f"pl.b{b}"
            put(f"{k}.n", v["n_streams_scored"])
            for side in ("before", "after"):
                put(f"{k}.{side}.sens", v[side]["sensitivity"], "{:.2f}"); put(f"{k}.{side}.fa", v[side]["fa_per_24h"], "{:.1f}")
                put(f"{k}.{side}.f1", v[side]["f1"], "{:.2f}"); put(f"{k}.{side}.det", f"{v[side]['detected']}/{v[side]['n_ref']}")
            put(f"{k}.df1", signed(v["f1_delta_mean"] or 0)); put(f"{k}.nup", v["n_improved"]); put(f"{k}.ndown", v["n_worse"])
            put(f"{k}.p", fmt_p(v["p_signflip"]))


def main() -> int:
    sys.path.insert(0, str(REPO / "scripts"))
    for fn in (design, final_stats, protocol, aura_anchor, variants, calibration, fairness, blind, addendum,
               streaming, per_subject_worst, occlusion, within_subject, auranet, parta,
               models_and_leaderboard, identity, scaling, placement, stream_anchor, stream_replication, stream_onset, stream_personalize,
               stream_nets, clinical, nn_tta, nn_stream_test, nn_seizeit2, nn_seizeit2_embedding, nn_tuned, nn_montage, nn_aura_ens, nn_monitor):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"[{fn.__name__}] FAILED: {type(e).__name__}: {e}")
    GEN.mkdir(parents=True, exist_ok=True)
    lines = ["% GENERATED by scripts/make_paper_numbers.py -- do not edit by hand.",
             "\\makeatletter"]
    for k in sorted(N):
        lines.append(f"\\@namedef{{num@{k}}}{{{N[k]}}}")
    lines.append("\\makeatother")
    (GEN / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (GEN / "numbers.json").write_text(json.dumps(N, indent=1, sort_keys=True), encoding="utf-8")
    print(f"{len(N)} numbers written; {len(MISSING)} missing: {MISSING[:30]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
