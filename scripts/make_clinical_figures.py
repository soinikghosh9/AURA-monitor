#!/usr/bin/env python
"""Clinical-evaluation figures, drawn from stored predictions and result artifacts only.

  fig_patients.pdf      per-patient AUC distributions (development, patient-independent folds)
  fig_test.pdf          held-out test patients: ROC curves with the development-fixed operating point,
                        and confusion matrices at that threshold
  fig_baseline.pdf      baseline normalization: per-patient change, per-stream change, and a 12-h
                        stream timeline with and without a refreshed baseline
  fig_utility.pdf       reliability before/after recalibration on development patients, decision
                        curve, and event sensitivity against time in warning in continuous EEG
  fig_features.pdf      what the 18 features carry: patient variance against label variance
  fig_metrics_dev.pdf   (supplement) threshold metrics of the leading models on development patients

Colour roles (validated all-pairs for colour-vision deficiency): AURA orange, AURA with baseline
normalization violet, pretrained foundation models blue, other networks gray. Text in ink colours.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from make_final_figures import BLUE, GRID, INK, INK2, ORANGE, OUT, SURF  # noqa: E402
from make_paper_numbers import TASK_NAME, latest, pretty  # noqa: E402

from make_final_figures import GRAY, INDIGO as VIOLET  # noqa: E402
SEIZ = "#e9c3bf"                 # pastel seizure shading
TASKS = ("typing", "ieeg", "scalp", "detect")
REF = {"typing": "feat_gbm", "ieeg": "feat_gbm_sub", "scalp": "feat_gbm_sub", "detect": "feat_logistic_sub"}
SHORT = {"typing": "IED classification", "ieeg": "iEEG seizure prediction",
         "scalp": "Scalp seizure prediction", "detect": "Scalp seizure detection"}


def _res(pattern):
    d = latest(pattern)
    return json.loads((d / "results.json").read_text()) if d else None, d


def _save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _addendum(task):
    fs = sorted(glob.glob(str(REPO / "results" / "blind_addendum_2*" / f"preds_{task}.npz")))
    with np.load(fs[-1], allow_pickle=True) as z:
        return z["_y"].astype(int), z["_groups"].astype(str), {k: z[k].astype(float) for k in z.files
                                                                if not k.startswith("_")}


def _panel_letter(ax, s):
    ax.text(-0.02, 1.06, s, transform=ax.transAxes, fontsize=8, fontweight="bold", color=INK,
            ha="right", va="bottom")


# ------------------------------------------------------------------ per-patient distributions
def fig_patients():
    R, _ = _res("clinical_metrics_*")
    tasks = ("ieeg", "scalp", "detect")
    models = [("REF", "AURA", ORANGE), ("auranet_hybrid", "AURA-Net", ORANGE),
              ("EEGNet", "EEGNet", GRAY), ("ShallowFBCSPNet", "ShallowFBCSP", GRAY),
              ("EEGConformer", "EEGConformer", GRAY), ("BIOT_pt", "BIOT (pt)", BLUE),
              ("CBraMod_pt", "CBraMod (pt)", BLUE)]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5), sharey=True)
    rng = np.random.default_rng(0)
    for ax, t in zip(axes, tasks):
        PP = R["per_patient_auc"][t]
        ax.axhline(0.5, color=INK2, lw=0.6, ls=(0, (2, 2)), zorder=1)
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        for i, (m, lab, col) in enumerate(models):
            key = REF[t] if m == "REF" else m
            v = np.array(list(PP.get(key, {}).values()))
            if len(v) < 3:
                ax.text(i, 0.5, "not run", rotation=90, ha="center", va="center", fontsize=5.4, color=INK2)
                continue
            vp = ax.violinplot(v, positions=[i], widths=0.8, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(col); b.set_alpha(0.22 if col == GRAY else 0.30); b.set_edgecolor("none")
            ax.scatter(i + rng.uniform(-0.14, 0.14, len(v)), v, s=6, color=col, lw=0, zorder=3,
                       alpha=0.9)
            ax.plot([i - 0.26, i + 0.26], [np.median(v)] * 2, color=INK, lw=1.2, zorder=4)
            ax.text(i, 0.02, f"{int((v < 0.5).sum())}/{len(v)}", ha="center", fontsize=5.4, color=INK2)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([lab for _, lab, _ in models], rotation=45, ha="right", fontsize=6)
        ax.set_title(SHORT[t], fontsize=7, loc="left", color=INK)
        ax.set_ylim(-0.04, 1.04)
    axes[0].set_ylabel("Per-patient AUC", fontsize=7)
    fig.tight_layout(w_pad=0.6)
    _save(fig, "fig_patients")


# ------------------------------------------------------------------ held-out test: ROC + confusion
def _best_fm(task, R):
    D = R["dev"][task]
    return max(("BIOT_pt", "CBraMod_pt"), key=lambda m: D.get(m, {}).get("auc", 0))


def fig_test():
    from sklearn.metrics import roc_curve
    R, _ = _res("clinical_metrics_*")
    fig = plt.figure(figsize=(7.16, 5.1))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.25, 2.1], hspace=0.48)
    top = outer[0].subgridspec(1, 4, wspace=0.42)
    bot = outer[1].subgridspec(2, 4, hspace=0.55, wspace=0.42)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f7f9fc", "#bcd3ea", "#4f7fb0"])
    for j, t in enumerate(TASKS):
        y, g, P = _addendum(t)
        T = R["test"][t]
        ref = REF[t] if REF[t] in P else REF[t] + "_sub"
        fm = _best_fm(t, R)
        curves = [(ref, "AURA", ORANGE, "-"), ("feat_logistic_base", "AURA, BN", VIOLET, "-"),
                  ("BIOT_pt", "BIOT (pt)", BLUE, (0, (3, 1.5))), ("CBraMod_pt", "CBraMod (pt)", BLUE, "-")]
        ax = fig.add_subplot(top[0, j])
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.6, ls=(0, (2, 2)))
        for arm, lab, col, ls in curves:
            if arm not in P or arm not in T:
                continue
            fpr, tpr, _ = roc_curve(y, P[arm])
            ax.plot(fpr, tpr, color=col, lw=1.3 if col != BLUE else 1.0, ls=ls,
                    label=f"{lab} {T[arm]['auc']:.2f}")
            ax.plot(1 - T[arm]["spec"], T[arm]["sens"], "o", ms=3.6, color=col, mec=SURF, mew=0.6, zorder=5)
        ax.set_title(SHORT[t], fontsize=7, loc="left", color=INK)
        ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01); ax.set_aspect("equal")
        ax.set_xlabel("1 $-$ specificity", fontsize=6.5)
        if j == 0:
            ax.set_ylabel("Sensitivity", fontsize=6.5)
        ax.legend(fontsize=5.0, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, handlelength=1.5,
                  columnspacing=0.8, labelspacing=0.3, frameon=False, title="AUC on test patients",
                  title_fontsize=5.0)
        ax.grid(color=GRID, lw=0.4)
        for r_, (arm, lab) in enumerate([("feat_logistic_base", "AURA, BN"),
                                         (fm, pretty(fm).replace(" (pretrained)", " (pt)"))]):
            v = T.get(arm)
            axc = fig.add_subplot(bot[r_, j])
            if not v:
                axc.axis("off"); continue
            M = np.array([[v["tn"], v["fp"]], [v["fn"], v["tp"]]])
            rown = M / M.sum(1, keepdims=True)
            axc.imshow(rown, cmap=cmap, vmin=0, vmax=1)
            for a in range(2):
                for b in range(2):
                    axc.text(b, a, f"{M[a, b]}\n{100 * rown[a, b]:.0f}%", ha="center", va="center",
                             fontsize=5.6, color=SURF if rown[a, b] > 0.6 else INK)
            neg, pos = {"typing": ("focal", "gen."), "detect": ("non-ictal", "ictal")}.get(
                t, ("interictal", "preictal"))
            axc.set_xticks([0, 1]); axc.set_xticklabels([neg, pos], fontsize=5.5)
            axc.set_yticks([0, 1]); axc.set_yticklabels([neg, pos], fontsize=5.5, rotation=90, va="center")
            axc.tick_params(length=0)
            for s_ in axc.spines.values():
                s_.set_visible(False)
            axc.set_title(f"{lab}\nSe {v['sens']:.2f}  Sp {v['spec']:.2f}  F1 {v['f1']:.2f}",
                          fontsize=5.6, color=INK, pad=2)
            if j == 0:
                axc.set_ylabel("True", fontsize=6)
            if r_ == 1:
                axc.set_xlabel("Predicted", fontsize=6)
    _save(fig, "fig_test")


# ------------------------------------------------------------------ baseline normalization
def _align_pair(task):
    """Per-patient AUC without and with baseline normalization (AURA + GBM); earliest-baseline files
    are preferred when both arms exist."""
    from sklearn.metrics import roc_auc_score
    root = REPO / "results" / "_oof_variants"
    for tag, label in (("_early", "earliest third of negatives"), ("", "random third of negatives")):
        fn = sorted(glob.glob(str(root / f"variant_{task}_subject_none_db{tag}_e25_*_align.npz")))
        fb = sorted(glob.glob(str(root / f"variant_{task}_subject_ea_base_db{tag}_e25_*_align.npz")))
        if fn and fb:
            zn, zb = np.load(fn[-1]), np.load(fb[-1])
            if "feat_gbm@s0" in zn.files and "feat_gbm@s0" in zb.files:
                y, g = zn["_y"].astype(int), zn["_groups"].astype(str)
                out = []
                for s in np.unique(g):
                    m = g == s
                    if len(np.unique(y[m])) == 2:
                        out.append((roc_auc_score(y[m], zn["feat_gbm@s0"][m]),
                                    roc_auc_score(y[m], zb["feat_gbm@s0"][m])))
                return np.array(out), label
    return None, None


def _streams():
    """Per-stream AUC and per-window predictions of boosted AURA in continuous EEG: the pre-registered
    replication over every development stream when it exists, else the original six streams."""
    SR, rd = _res("stream_replication_*")
    if SR:
        per = {s: {"none": v["none"], "anchored": v["fixed"], "rolling": v["refreshed"]}
               for s, v in SR["models"]["feat_gbm"]["per_stream_auc"].items()}
        orig = {s for s, v in SR["streams"].items() if v.get("original")}
        with np.load(rd / "preds.npz", allow_pickle=True) as z:
            Y, G, Tt = z["_y"].astype(int), z["_groups"].astype(str), z["_t"].astype(float)
            P = {k.replace("|fixed", "|anchored").replace("|refreshed", "|rolling"): z[k].astype(float)
                 for k in z.files if not k.startswith("_")}
        n_ev = SR["models"]["feat_gbm"]["all"]["n_events"]
        return per, orig, Y, G, Tt, P, n_ev
    SA, sd = _res("stream_anchor_*")
    per = SA["models"]["feat_gbm"]["per_stream_auc"]
    with np.load(sd / "preds.npz", allow_pickle=True) as z:
        Y, G, Tt = z["_y"].astype(int), z["_groups"].astype(str), z["_t"].astype(float)
        P = {k: z[k].astype(float) for k in z.files if not k.startswith("_")}
    return per, set(per), Y, G, Tt, P, SA["n_events"]


def fig_baseline():
    from bsf.eval import clinical as C
    per, orig, Y, G, Tt, P, n_ev = _streams()
    fig = plt.figure(figsize=(7.16, 4.3))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.6, wspace=0.45)
    # (a) per-patient change, development detection
    ax = fig.add_subplot(gs[0, 0])
    pr, lab = _align_pair("detect")
    if pr is not None:
        for a, b in pr:
            ax.plot([0, 1], [a, b], color=VIOLET if b > a else GRAY, lw=0.7, alpha=0.8, marker="o", ms=2.2)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["without BN", "with BN"], fontsize=6)
        ax.set_title(f"Detection, {len(pr)} patients\n({int((pr[:, 1] > pr[:, 0]).sum())} improved)",
                     fontsize=6.5, loc="left", color=INK)
        ax.text(0.5, -0.32, f"baseline: {lab}", transform=ax.transAxes, ha="center", fontsize=5.4,
                color=INK2)
    ax.set_ylabel("Per-patient AUC", fontsize=6.5); ax.set_xlim(-0.25, 1.25)
    ax.axhline(0.5, color=INK2, lw=0.5, ls=(0, (2, 2))); ax.grid(axis="y", color=GRID, lw=0.4)
    _panel_letter(ax, "a")
    # (b) per-stream AUC change from no BN: fixed and refreshed baselines
    ax = fig.add_subplot(gs[0, 1])
    rng = np.random.default_rng(3)
    for x, arm in ((0, "anchored"), (1, "rolling")):
        d = np.array([v[arm] - v["none"] for v in per.values()])
        isorig = np.array([s in orig for s in per])
        jit = rng.uniform(-0.13, 0.13, len(d))
        col = np.where(d > 0, VIOLET, GRAY)
        for o, fc in ((True, None), (False, "white")):
            m = isorig == o
            ax.scatter(x + jit[m], d[m], s=11, c=list(col[m]) if fc is None else fc,
                       edgecolors=list(col[m]), linewidths=0.7, zorder=3)
        med = float(np.median(d))
        ax.plot([x - 0.24, x + 0.24], [med, med], color=INK, lw=1.2, zorder=4)
        ax.text(x + 0.31, med, f"{med:+.2f}", fontsize=5.4, color=INK, va="center",
                bbox=dict(boxstyle="square,pad=0.1", fc=SURF, ec="none"))
    ax.axhline(0, color=INK2, lw=0.6, ls=(0, (2, 2)))
    ax.set_xticks([0, 1]); ax.set_xticklabels(["fixed\n(first 30 min)", "refreshed\n(prior 30 min)"], fontsize=5.6)
    nimp = sum(v["rolling"] > v["none"] for v in per.values())
    ax.set_title(f"Continuous EEG, {len(per)} patients\n(refreshed: {nimp} improved)", fontsize=6.5,
                 loc="left", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.4)
    ax.set_ylabel("ΔAUC per stream (vs no BN)", fontsize=6.5); ax.set_xlim(-0.45, 1.6)
    lo = min(min(v["anchored"], v["rolling"]) - v["none"] for v in per.values())
    ax.set_ylim(lo - 0.2, None)
    ax.text(0.02, 0.03, "filled: six streams used to devise the rule\nopen: pre-registered replication",
            transform=ax.transAxes, fontsize=5.0, color=INK2, va="bottom")
    _panel_letter(ax, "b")
    # (c) event sensitivity against time in warning, pooled streams
    ax = fig.add_subplot(gs[0, 2])
    budgets = np.linspace(0.005, 0.5, 60)
    for arm, col, lab, ls in (("none", ORANGE, "AURA, no BN", "-"), ("anchored", VIOLET, "fixed BN", (0, (3, 1.5))),
                              ("rolling", VIOLET, "refreshed BN", "-")):
        p = P[f"feat_gbm|{arm}"]
        se = [C.sensitivity_at_budget(Y, p, b)["event_sensitivity"] for b in budgets]
        ax.plot(budgets, se, color=col, lw=1.1, ls=ls, label=lab)
    ax.plot([0, 0.5], [0, 0.5], color=INK2, lw=0.6, ls=(0, (2, 2)), label="chance")
    ax.axvline(0.10, color=GRID, lw=0.8)
    ax.set_xlabel("Time in warning", fontsize=6.5); ax.set_ylabel("Seizures detected", fontsize=6.5)
    ax.set_xlim(0, 0.5); ax.set_ylim(0, 1.02); ax.grid(color=GRID, lw=0.4)
    lg = ax.legend(fontsize=5.2, loc="lower right", handlelength=1.6, frameon=True, framealpha=1,
                   edgecolor="none")
    lg.get_frame().set_facecolor(SURF)
    ax.set_title(f"{n_ev} seizures, AURA + GBM", fontsize=6.5, loc="left", color=INK)
    _panel_letter(ax, "c")
    # (d) one stream, 12 h
    gains = {s: v["rolling"] - v["none"] for s, v in per.items() if v["rolling"] >= 0.9 and s in orig}
    s0 = max(gains, key=gains.get) if gains else list(per)[0]
    m = G == s0
    ax = fig.add_subplot(gs[1, :])
    th = {arm: float(np.quantile(P[f"feat_gbm|{arm}"], 0.90)) for arm in ("none", "rolling")}
    h = Tt[m] / 3600.0; yy = Y[m]
    brk = np.r_[0, np.where(np.diff(h) > 0.05)[0] + 1, len(h)]          # gaps between recordings
    on = np.where(np.diff(np.r_[0, yy, 0]) != 0)[0].reshape(-1, 2)
    for a, b in on:
        ax.axvspan(h[a], h[min(b, len(h) - 1)] + 8 / 3600, color=SEIZ, alpha=0.7, lw=0)
    for arm, col, lab, off in (("none", ORANGE, "AURA + GBM, no BN", 1.12), ("rolling", VIOLET,
                                                                              "AURA + GBM, refreshed BN", 0.0)):
        p = P[f"feat_gbm|{arm}"][m]
        k = 15
        for a_, b_ in zip(brk[:-1], brk[1:]):
            seg = slice(a_, b_)
            ps = np.convolve(p[seg], np.ones(k) / k, mode="same") if b_ - a_ > k else p[seg]
            ax.plot(h[seg], p[seg] + off, color=col, lw=0.3, alpha=0.35)
            ax.plot(h[seg], ps + off, color=col, lw=0.9)
        ax.axhline(th[arm] + off, color=col, lw=0.6, ls=(0, (3, 2)))
        ax.text(h[-1] + 0.05, off + 0.5, lab.replace("AURA + GBM, ", ""), fontsize=5.6, color=col, va="center")
    ax.set_yticks([0, 0.5, 1, 1.12, 1.62, 2.12]); ax.set_yticklabels(["0", "0.5", "1", "0", "0.5", "1"], fontsize=5.5)
    ax.set_xlabel("Hours from the start of the recording (gaps between files left blank)", fontsize=6.5)
    ax.set_ylabel("P(seizure)", fontsize=6.5)
    ax.set_xlim(h[0], h[-1] + 1.1)
    ax.set_title(f"Patient {s0.split('_')[-1]}, continuous scalp EEG", fontsize=6.5, loc="left", color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    _panel_letter(ax, "d")
    _save(fig, "fig_baseline")


# ------------------------------------------------------------------ calibration and clinical utility
def _platt(p_dev, y_dev, p):
    from sklearn.linear_model import LogisticRegression
    lg = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))  # noqa: E731
    lr = LogisticRegression(C=1e6).fit(lg(p_dev)[:, None], y_dev)
    return lr.predict_proba(lg(p)[:, None])[:, 1]


def _reliability(y, p, n_bins=6):
    q = np.quantile(p, np.linspace(0, 1, n_bins + 1)); q[-1] += 1e-9
    b = np.clip(np.searchsorted(q, p, side="right") - 1, 0, n_bins - 1)
    return (np.array([p[b == i].mean() for i in range(n_bins) if (b == i).any()]),
            np.array([y[b == i].mean() for i in range(n_bins) if (b == i).any()]))


def _bss(y, p):
    br = np.mean((p - y) ** 2); ref = np.mean((y.mean() - y) ** 2)
    return 1 - br / ref


def fig_utility():
    from run_clinical_metrics import dev_source
    from run_final_stats import gather
    t = "typing"
    y, g, P = _addendum(t)
    yd, gd, A = gather(t)
    dev = {m: (yd, gd, a) for m, a in A.items()}
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.45))
    # (a) reliability, held-out test, raw and recalibrated on development patients
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.6, ls=(0, (2, 2)))
    rec = {}
    for arm, col, lab in (("feat_logistic_base", VIOLET, "AURA, BN"), ("BIOT_pt", BLUE, "BIOT (pt)")):
        ydv, pdv = dev_source(t, arm, dev)
        pr = _platt(pdv, ydv, P[arm]); rec[arm] = pr
        for p_, ls, st in ((P[arm], (0, (3, 1.5)), "raw"), (pr, "-", "recal.")):
            xs, ys = _reliability(y, p_)
            ax.plot(xs, ys, color=col, lw=1.0, ls=ls, marker="o", ms=2.4,
                    label=f"{lab}, {st} (BSS {_bss(y, p_):+.2f})")
    ax.set_xlabel("Predicted probability", fontsize=6.5); ax.set_ylabel("Observed frequency", fontsize=6.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(color=GRID, lw=0.4)
    ax.legend(fontsize=4.9, loc="upper left", handlelength=1.8)
    ax.set_title("IED classification, test patients", fontsize=6.5, loc="left", color=INK)
    _panel_letter(ax, "a")
    # (b) decision curve on the recalibrated outputs
    ax = axes[1]
    ths = np.linspace(0.02, 0.6, 59)
    nb = lambda p, th: ((p >= th) & (y == 1)).mean() - ((p >= th) & (y == 0)).mean() * th / (1 - th)  # noqa: E731
    prev = y.mean()
    ax.plot(ths, [prev - (1 - prev) * th / (1 - th) for th in ths], color=GRAY, lw=0.8, label="flag all")
    ax.axhline(0, color=INK, lw=0.6, ls=(0, (2, 2)), label="flag none")
    ydv, pdv = dev_source(t, REF[t], dev)
    rec["ref"] = _platt(pdv, ydv, P[REF[t]])
    for key, col, lab in (("ref", ORANGE, "AURA"), ("feat_logistic_base", VIOLET, "AURA, BN"),
                          ("BIOT_pt", BLUE, "BIOT (pt)")):
        ax.plot(ths, [nb(rec[key], th) for th in ths], color=col, lw=1.1, label=lab)
    ax.set_ylim(-0.03, prev + 0.02); ax.set_xlim(0, 0.6)
    ax.set_xlabel("Threshold probability", fontsize=6.5); ax.set_ylabel("Net benefit", fontsize=6.5)
    ax.grid(color=GRID, lw=0.4); ax.legend(fontsize=5.2, loc="upper right")
    ax.set_title("Decision curve, test patients", fontsize=6.5, loc="left", color=INK)
    _panel_letter(ax, "b")
    # (c) calibration across the benchmark: informative outputs before and after recalibration
    ax = axes[2]
    N = json.loads((REPO / "manuscript" / "generated" / "numbers.json").read_text())
    cats = [("raw", "none"), ("icpt", "intercept"), ("temp", "temperature"), ("platt", "Platt")]
    vals = [int(N.get(f"cal.{k}", 0)) for k, _ in cats]
    n = int(N.get("cal.n", 1))
    ax.barh(range(len(cats)), vals, color=[GRAY, BLUE, BLUE, BLUE], height=0.62)
    for i, v in enumerate(vals):
        ax.text(v + 1.5, i, f"{v}/{n}", va="center", fontsize=6, color=INK)
    ax.set_yticks(range(len(cats))); ax.set_yticklabels([c for _, c in cats], fontsize=6)
    ax.invert_yaxis(); ax.set_xlim(0, n * 1.15)
    ax.set_xlabel("Model-task outputs with positive\nBrier skill on held-out patients", fontsize=6.3)
    ax.set_title("Recalibration map", fontsize=6.5, loc="left", color=INK)
    _panel_letter(ax, "c")
    fig.tight_layout(w_pad=1.2)
    _save(fig, "fig_utility")


# ------------------------------------------------------------------ what the features carry
FEATS = (["δ", "θ", "α", "β", "γ", "HF"], ["line length", "synchrony", "spectral edge",
                                            "sharp transients", "kurtosis", "θ–γ PAC"])


def fig_features():
    """Two heatmaps over the 18 features x 4 tasks: the share of each feature's variance explained by
    the clinical label within patients, and the share explained by patient identity."""
    R, _ = _res("identity_decomp_*")
    names = [f"{b} power" for b in FEATS[0]] + [f"{b} spread" for b in FEATS[0]] + FEATS[1]
    L = np.array([(R[t].get("matched") or R[t]["first_run"])["per_feature_label"] for t in TASKS]).T
    Pt = np.array([(R[t].get("matched") or R[t]["first_run"])["per_feature_patient"] for t in TASKS]).T
    ida = [(R[t].get("matched") or R[t]["first_run"]) for t in TASKS]
    ramps = (("#fcf6f2", "#ebbfa9", "#a85a39"), ("#f7f9fc", "#b7cfe8", "#3f6f9f"))
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.5))
    cols = ["IED\nclassif.", "iEEG\npred.", "Scalp\npred.", "Scalp\ndetect."]
    for ax, M, ramp, title in ((axes[0], L, ramps[0], "Variance explained by the label, within patients"),
                               (axes[1], Pt, ramps[1], "Variance explained by patient identity")):
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("r", ramp)
        vmax = 0.35 if M is L else 1.0
        ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.2,
                        color=SURF if M[i, j] > 0.62 * vmax else INK)
        ax.set_xticks(range(4)); ax.set_xticklabels(cols, fontsize=5.8)
        ax.xaxis.tick_top()
        ax.set_yticks(range(18)); ax.set_yticklabels(names, fontsize=5.8)
        for y_ in (5.5, 11.5):
            ax.axhline(y_, color=SURF, lw=1.6)
        ax.tick_params(length=0)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.set_title(title, fontsize=6.6, color=INK, pad=18)
    axes[1].set_xlabel("Patient identity decoded from the 18 features: " + ", ".join(
        f"{100 * r['identity_acc']:.0f}%" for r in ida) + "\n(chance " + ", ".join(
        f"{100 * r['identity_chance']:.0f}%" for r in ida) + ")", fontsize=5.6, color=INK2)
    axes[0].set_xlabel("Rows: band power averaged over channels (top), its spatial spread (middle),\n"
                       "and rhythm, synchrony and transient features (bottom)", fontsize=5.6, color=INK2)
    _panel_letter(axes[0], "a"); _panel_letter(axes[1], "b")
    fig.tight_layout(w_pad=2.5)
    _save(fig, "fig_features")


# ------------------------------------------------------------------ supplement: metric heatmap
def fig_metrics_dev():
    R, _ = _res("clinical_metrics_*")
    keys = ("auc", "sens", "spec", "ppv", "npv", "f1", "bal_acc", "mcc")
    lab = ("AUC", "Sens.", "Spec.", "PPV", "NPV", "F1", "Bal. acc.", "MCC")
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f7f9fc", "#b7cfe8", "#3f6f9f"])
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 8.2))
    for ax, t in zip(axes.flat, TASKS):
        D = R["dev"][t]
        ms = [m for m in sorted(D, key=lambda m: -D[m]["auc"])
              if not (t != "typing" and m in ("feat_logistic", "feat_gbm"))][:16]
        M = np.array([[D[m][k] if D[m][k] == D[m][k] else np.nan for k in keys] for m in ms])
        ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, "--" if np.isnan(M[i, j]) else f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=4.8, color=SURF if M[i, j] > 0.62 else INK)
        ax.set_xticks(range(len(keys))); ax.set_xticklabels(lab, fontsize=5.6, rotation=35, ha="right")
        ax.set_yticks(range(len(ms))); ax.set_yticklabels([pretty(m) for m in ms], fontsize=5.4)
        ax.tick_params(length=0)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.set_title(SHORT[t], fontsize=7, loc="left", color=INK)
    fig.tight_layout()
    _save(fig, "fig_metrics_dev")


def fig_onset():
    """Supplementary: each seizure's departure of every AURA feature from the preceding 30 min
    (median robust z over its ictal windows), seizures grouped by patient; diverging, clipped at +-4."""
    from matplotlib.colors import LinearSegmentedColormap
    R, _ = _res("stream_onset_*")
    names = [f"{b} power" for b in FEATS[0]] + [f"{b} spread" for b in FEATS[0]] + FEATS[1]
    Z = np.array([p["z"] for p in R["per_seizure"]])
    subj = [p["subject"].split("_")[-1] for p in R["per_seizure"]]
    cmap = LinearSegmentedColormap.from_list("div", ["#3f78b0", "#f3f1ec", "#b5583a"])
    fig = plt.figure(figsize=(7.16, 3.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[4.2, 0.07, 0.42, 1.0], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(np.clip(Z.T, -4, 4), aspect="auto", cmap=cmap, vmin=-4, vmax=4, interpolation="nearest")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=5.6)
    edges = [i for i in range(1, len(subj)) if subj[i] != subj[i - 1]]
    for e in edges:
        ax.axvline(e - 0.5, color="white", lw=0.8)
    starts, ends = [0] + edges, edges + [len(subj)]
    ax.set_xticks([(a + b - 1) / 2 for a, b in zip(starts, ends)])
    ax.set_xticklabels([subj[a] for a in starts], fontsize=4.4, rotation=90, color=INK2)
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.set_xlabel(f"{len(subj)} seizures in {R['n_streams']} continuous recordings, grouped by patient",
                  fontsize=6.5, labelpad=4)
    for sp in ax.spines.values():
        sp.set_visible(False)
    _panel_letter(ax, "a")
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 1]))
    cb.set_label("median z vs preceding 30 min", fontsize=5.8); cb.ax.tick_params(labelsize=5.4)
    cb.outline.set_visible(False)
    # (b) share of seizures beyond +2 and below -2, per feature
    ax = fig.add_subplot(gs[0, 3])
    F = R["features"]; keys = R["feature_names"]
    up = [F[k]["frac_above2"] for k in keys]; dn = [F[k]["frac_below_m2"] for k in keys]
    yy = np.arange(len(keys))
    ax.barh(yy, up, color="#b5583a", height=0.62, label="z > 2")
    ax.barh(yy, [-v for v in dn], color="#3f78b0", height=0.62, label="z < −2")
    ax.set_yticks([]); ax.set_ylim(len(keys) - 0.5, -0.5)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_xlim(-0.6, 0.6); ax.set_xticks([-0.5, 0, 0.5]); ax.set_xticklabels(["0.5", "0", "0.5"], fontsize=5.4)
    ax.set_xlabel("Share of seizures", fontsize=6.0)
    ax.grid(axis="x", color=GRID, lw=0.4)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    lg = ax.legend(fontsize=5.0, loc="lower left", frameon=True, framealpha=1, edgecolor="none", handlelength=1.0)
    lg.get_frame().set_facecolor(SURF)
    _panel_letter(ax, "b")
    _save(fig, "fig_onset")


def fig_tta():
    """(a) Mean per-stream AUC change of each test-time adaptation over the unadapted model, development
    streams; (b) how well each representation identifies the patient and the label; (c) the refreshed
    baseline's per-stream effect for boosted AURA across the development replication, the held-out test
    patients and the external SeizeIT2 cohort."""
    from matplotlib.colors import LinearSegmentedColormap
    f = REPO / "results" / "stream_tta" / "results.json"
    if not f.exists():
        return
    R = json.loads(f.read_text())
    rows = [("EEGNet", "EEGNet"), ("SCCNet", "SCCNet"), ("BIOT", "BIOT"), ("deep_cnn_raw", "Repo CNN"),
            ("auranet_hybrid", "AURA-Net"), ("feat_gbm", "AURA + GBM")]
    cols = [("adabn_stream", "AdaBN\nstream"), ("adabn_refreshed", "AdaBN\nrefresh."), ("tent", "TENT"),
            ("ea_refreshed", "EA\nrefresh."), ("repr", "Represent.\nrefresh.")]
    M = np.full((len(rows), len(cols)), np.nan); S = np.zeros_like(M, bool)
    for i, (m, _) in enumerate(rows):
        for j, (arm, _) in enumerate(cols):
            a2 = ("refreshed" if m in ("feat_gbm", "auranet_hybrid") else "probe_gbm_refreshed") if arm == "repr" else arm
            d = R["models"].get(m, {}).get(a2, {}).get("delta_vs_none")
            if d:
                M[i, j] = d["mean"]; S[i, j] = (d["p_signflip"] or 1) < 0.05
    fig = plt.figure(figsize=(7.16, 3.15))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.5, 1.35, 1.55], wspace=0.55)
    ax = fig.add_subplot(gs[0, 0])
    cmap = LinearSegmentedColormap.from_list("div", ["#3f78b0", "#f3f1ec", "#b5583a"])
    lim = max(0.1, float(np.nanmax(np.abs(M))))
    ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            if np.isfinite(M[i, j]):
                v = 0.0 if abs(M[i, j]) < 0.005 else M[i, j]
                ax.text(j, i, (f"{v:+.2f}" if v else "0.00").replace("-", "\u2212") + ("*" if S[i, j] else ""),
                        ha="center", va="center", fontsize=5.6, color=INK)
            else:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=5.0, color=INK2)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[1] for c in cols], fontsize=5.3)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[1] for r in rows], fontsize=6)
    ax.text(0.0, -0.2, "Represent.: learned embedding (networks) or features (AURA, AURA-Net); * p < 0.05",
            transform=ax.transAxes, fontsize=4.8, color=INK2, va="top")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"Change in per-stream AUC ({R['n_streams']} streams)", fontsize=6.5, loc="left", color=INK)
    _panel_letter(ax, "a")
    # (b) identity and label decodability
    ax = fig.add_subplot(gs[0, 1])
    agg = {}
    for key, e in R.get("E7", {}).items():
        if e:
            agg.setdefault(key.split("|")[0], []).append(e["net"])
            agg.setdefault("aura", []).append(e["aura"])
    order = [(m, lab) for m, lab in rows[:4] if m in agg] + [("aura", "AURA features")]
    if agg:
        yy = np.arange(len(order))
        idv = [np.mean([x["id_acc"] for x in agg[m]]) for m, _ in order]
        lab = [np.mean([x["label_auc_cv"] for x in agg[m]]) for m, _ in order]
        ch = np.mean([x["chance"] for x in agg["aura"]])
        ax.barh(yy - 0.18, idv, height=0.34, color=VIOLET, label="patient (accuracy)")
        ax.barh(yy + 0.18, lab, height=0.34, color=GRAY, label="seizure (AUC)")
        ax.axvline(ch, color=VIOLET, lw=0.6, ls=(0, (2, 2)))
        ax.axvline(0.5, color=INK2, lw=0.6, ls=(0, (2, 2)))
        ax.set_yticks(yy); ax.set_yticklabels([o[1] for o in order], fontsize=6); ax.invert_yaxis()
        ax.set_xlim(0, 1.0); ax.set_xlabel("Decoded from the representation", fontsize=6)
        ax.grid(axis="x", color=GRID, lw=0.4)
        ax.legend(fontsize=5.0, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=1, frameon=False,
                  handlelength=1.0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title("What the representation encodes", fontsize=6.5, loc="left", color=INK)
    _panel_letter(ax, "b")
    # (c) refreshed - none, boosted AURA, across cohorts
    ax = fig.add_subplot(gs[0, 2])
    cohorts = []
    SR, _ = _res("stream_replication_*")
    if SR:
        per = SR["models"]["feat_gbm"]["per_stream_auc"]
        orig = {s for s, v in SR["streams"].items() if v.get("original")}
        cohorts.append(("Development\nreplication", [v["refreshed"] - v["none"] for s, v in per.items() if s not in orig],
                        SR["models"]["feat_gbm"]["replication"]["delta_refreshed"]["p_signflip"]))
    for pat, lab in (("stream_test_*", "Held-out\npatients"), ("seizeit2_external_*", "SeizeIT2\n(external)")):
        T, _ = _res(pat)
        if T:
            a = T["models"]["feat_gbm|refreshed"]["per_stream_auc"]; b = T["models"]["feat_gbm|none"]["per_stream_auc"]
            cohorts.append((lab, [a[s] - b[s] for s in a], T["primary"]["p_signflip"]))
    rng = np.random.default_rng(1)
    for x, (lab, d, p) in enumerate(cohorts):
        d = np.asarray(d)
        ax.scatter(x + rng.uniform(-0.16, 0.16, len(d)), d, s=8, c=np.where(d > 0, VIOLET, GRAY), alpha=0.8, lw=0, zorder=3)
        med = float(np.median(d))
        ax.plot([x - 0.26, x + 0.26], [med, med], color=INK, lw=1.2, zorder=4)
        ax.text(x, ax.get_ylim()[1] if False else max(d) + 0.05, f"n={len(d)}\np={p:.3f}" if p is not None else f"n={len(d)}",
                ha="center", va="bottom", fontsize=5.0, color=INK2)
    ax.axhline(0, color=INK2, lw=0.6, ls=(0, (2, 2)))
    ax.set_xticks(range(len(cohorts))); ax.set_xticklabels([c[0] for c in cohorts], fontsize=5.4)
    ax.set_xlim(-0.6, len(cohorts) - 0.4)
    top = max(max(c[1]) for c in cohorts) if cohorts else 0.5
    ax.set_ylim(min(-0.3, min(min(c[1]) for c in cohorts) - 0.05) if cohorts else -0.3, top + 0.22)
    ax.set_ylabel("ΔAUC per stream (refreshed − none)", fontsize=6)
    ax.grid(axis="y", color=GRID, lw=0.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("Refreshed baseline, AURA + GBM", fontsize=6.5, loc="left", color=INK)
    _panel_letter(ax, "c")
    _save(fig, "fig_tta")


def main() -> int:
    for fn in (fig_patients, fig_test, fig_baseline, fig_utility, fig_features, fig_metrics_dev, fig_onset, fig_tta):
        try:
            fn(); print(f"{fn.__name__}: ok")
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"{fn.__name__}: FAILED {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
