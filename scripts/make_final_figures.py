#!/usr/bin/env python
"""Figures for the benchmark manuscript, drawn from the same artifacts as the generated numbers.

  fig_contrasts.pdf  the paired AUC difference from the task's analytic arm, with the
                     hierarchical-bootstrap 95% interval, for the twelve models per task closest to it
                     (2x2 small multiples, one shared x-scale); fig_contrasts_all.pdf has every model
  fig_protocol.pdf   AUC of the same models under each fold-drawing protocol (the protocol ladder)

Palette: the dataviz reference palette's first three categorical slots (validated all-pairs on the
light surface; aqua < 3:1 contrast, so every series also carries a distinct marker and a legend), a
neutral gray for the bulk of published architectures, text in ink colours, one axis per panel.
"""

from __future__ import annotations

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
from make_paper_numbers import SPLIT_LABEL, TASK_NAME, latest, pretty  # noqa: E402

OUT = REPO / "manuscript" / "figures"
# muted palette, validated all-pairs for colour-vision deficiency (dataviz validator): AURA terracotta,
# baseline normalization indigo, pretrained foundation models blue; a fourth (ochre) only for task lines
INK, INK2, GRID, SURF = "#2b2a28", "#5e5b55", "#ecebe6", "#ffffff"
BLUE, ORANGE, INDIGO, GRAY = "#5592cb", "#c8734f", "#5a4e96", "#a39f96"
OCHRE, SPINE = "#a08a3c", "#8a867e"
AQUA = INDIGO          # retained name for older callers
plt.rcParams.update({
    "font.size": 7, "font.family": ["Arial", "DejaVu Sans"], "axes.edgecolor": SPINE, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.facecolor": SURF, "figure.facecolor": SURF,
    "pdf.fonttype": 42})


def category(m):
    if m.endswith("_pt") or m.endswith("_probe"):
        return "pretrained foundation model", BLUE, "D"
    if m.startswith("auranet") or m == "deep_cnn_raw":
        if m.startswith("auranet"):
            return "AURA-Net", ORANGE, "s"
        return "published architecture", GRAY, "o"
    if m.startswith("feat"):
        return "AURA, other head", ORANGE, "^"
    if m.endswith("_scratch200"):
        return "same foundation architecture, no pretraining", BLUE, "d_open"
    return "published architecture", GRAY, "o"


def _contrast_panels(R, tasks, top, figsize, fname):
    from matplotlib.ticker import FormatStrFormatter, MultipleLocator
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    keep = {}
    for t in tasks:
        # the first-run-input analytic arms are a fairness artefact reported in the text, not
        # competitors; on typing the input is identical, so there they are the other head
        cs = {m: v for m, v in R[t]["contrasts"].items()
              if t == "typing" or m not in ("feat_logistic", "feat_gbm")}
        order = sorted(cs, key=lambda m: cs[m]["delta"])
        keep[t] = (cs, order[-top:] if top else order)
    # one x-scale for every panel, so the four tasks can be compared at a glance
    lo = min(cs[m]["lo"] for cs, o in keep.values() for m in o)
    hi = max(cs[m]["hi"] for cs, o in keep.values() for m in o)
    lo, hi = np.floor(lo * 10) / 10 - 0.02, np.ceil(hi * 10) / 10 + 0.02
    seen = {}
    for ax, t in zip(np.ravel(axes), tasks):
        cs, order = keep[t]
        ys = np.arange(len(order))
        ax.axvline(0, color=INK2, lw=0.7, zorder=1)
        ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
        for y, m in zip(ys, order):
            c = cs[m]
            lab, col, mk = category(m)
            ax.plot([c["lo"], c["hi"]], [y, y], color=col, lw=1.0, zorder=2, solid_capstyle="round")
            if mk == "d_open":
                h = ax.scatter([c["delta"]], [y], s=16, marker="D", facecolor=SURF, edgecolor=col,
                               lw=0.9, zorder=3)
            else:
                h = ax.scatter([c["delta"]], [y], s=14, marker=mk, color=col, edgecolor=SURF, lw=0.5,
                               zorder=3)
            seen.setdefault(lab, h)
        ax.set_yticks(ys)
        ax.set_yticklabels([pretty(m) + (f" ({cs[m]['n_seeds']} seeds)" if cs[m]["n_seeds"] > 1 else "")
                            for m in order], fontsize=5.6)
        ax.set_ylim(-0.7, len(order) - 0.3)
        ax.set_xlim(lo, hi)
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.set_title(f"{TASK_NAME[t]} vs. {pretty(R[t]['reference_arm'])} ({R[t]['reference_auc']:.3f})",
                     fontsize=7, color=INK, loc="left")
        ax.tick_params(axis="x", labelsize=6)
    for ax in np.ravel(axes)[2:]:
        ax.set_xlabel(r"$\Delta$AUC (model $-$ AURA)", fontsize=6.5)
    fig.legend(list(seen.values()), list(seen.keys()), loc="lower center", ncol=len(seen),
               fontsize=6.3, bbox_to_anchor=(0.5, -0.005), handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.035, 1, 1), w_pad=1.0, h_pad=0.8)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{fname}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{fname}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_contrasts():
    """Main text: the twelve models per task closest to (or above) the analytic arm; supplement: all."""
    d = latest("final_stats_*")
    R = json.loads((d / "results.json").read_text())
    tasks = [t for t in ("typing", "ieeg", "scalp", "detect") if t in R]
    _contrast_panels(R, tasks, 12, (7.16, 3.9), "fig_contrasts")
    n_max = max(len(R[t]["contrasts"]) for t in tasks)
    _contrast_panels(R, tasks, None, (7.16, 0.105 * n_max * 2 + 1.2), "fig_contrasts_all")


def fig_protocol():
    d = latest("protocol_decomp_*")
    P = json.loads((d / "results.json").read_text())
    dv = latest("variant_stats_*")
    V = json.loads((dv / "results.json").read_text()) if dv else {}
    order = ["W-rand", "W-event", "P-seg", "P-event", "X"]
    xpos = {"W-rand": 0.0, "W-event": 1.0, "P-seg": 2.4, "P-event": 3.4, "X": 4.4}
    tasks = [t for t in ("scalp", "ieeg", "detect", "typing") if t in P]
    fig, axes = plt.subplots(1, len(tasks), figsize=(7.16, 2.3), sharey=True)
    handles = {}
    for ax, t in zip(np.atleast_1d(axes), tasks):
        ax.axhline(0.5, color=INK2, lw=0.6, ls=(0, (2, 2)), zorder=1)
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        # networks from the protocol factorial (pooled protocols only), thin and recessive
        fac = (V.get("factorial") or {}).get(t) or {}
        for m, vals in (fac.get("auc_pooled") or {}).items():
            if m.startswith("feat"):
                continue
            xs = [xpos[{"segment": "P-seg", "event": "P-event", "subject": "X"}[p]] for p in vals]
            ys = list(vals.values())
            srt = np.argsort(xs)
            h, = ax.plot(np.array(xs)[srt], np.array(ys)[srt], color=GRAY, lw=0.8, marker="o", ms=2.5,
                         zorder=2)
            handles.setdefault("networks (4 families)", h)
        for m, col, mk in (("feat_logistic", ORANGE, "o"), ("feat_gbm", ORANGE, "s")):
            r = P[t].get(m)
            if not r:
                continue
            for grp in (("W-rand", "W-event"), ("P-seg", "P-event", "X")):
                ks = [k for k in grp if k in r["protocols"]]
                if not ks:
                    continue
                h, = ax.plot([xpos[k] for k in ks], [r["protocols"][k]["pooled_auc"] for k in ks],
                             color=col, lw=1.4, marker=mk, ms=4, zorder=3,
                             ls=(0, (3, 1.6)) if m == "feat_logistic" else "-",
                             mfc=SURF if m == "feat_logistic" else col, mec=col, mew=0.9)
                handles.setdefault("AURA + logistic" if m == "feat_logistic" else "AURA + GBM", h)
        ax.set_xticks([xpos[k] for k in order])
        ax.set_xticklabels([SPLIT_LABEL[k] for k in order], fontsize=6, rotation=35, ha="right")
        ax.axvline(1.7, color=GRID, lw=0.8, zorder=0)
        ax.text(0.5, 0.36, "patient-\nspecific", ha="center", fontsize=5.6, color=INK2, linespacing=1.0)
        ax.text(3.4, 0.36, "patients\npooled", ha="center", fontsize=5.6, color=INK2, linespacing=1.0)
        ax.set_title(TASK_NAME[t], fontsize=7, color=INK, loc="left")
        ax.set_ylim(0.35, 1.0)
    np.atleast_1d(axes)[0].set_ylabel("Pooled AUC", fontsize=7)
    fig.legend(list(handles.values()), list(handles.keys()), loc="lower center", ncol=len(handles),
               fontsize=6.5, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "fig_protocol.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_protocol.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    for fn in (fig_design, fig_contrasts, fig_protocol, fig_scaling):
        try:
            fn(); print(f"{fn.__name__}: ok")
        except Exception as e:  # noqa: BLE001
            print(f"{fn.__name__}: FAILED {type(e).__name__}: {e}")
    return 0




def fig_design():
    """Study design: tasks -> models -> factors varied -> what is reported. Counts from numbers.json."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    N = json.loads((REPO / "manuscript" / "generated" / "numbers.json").read_text())
    g = lambda k, d="?": N.get(k, d)  # noqa: E731
    cols = [
        ("Four cross-patient tasks", [
            f"iEEG forecasting\nKaggle AES + NeuroVista\ncanine; {g('dev.ieeg.subj')} dev / {g('test.ieeg.subj')} test",
            f"Scalp forecasting\nCHB-MIT, Siena\n{g('dev.scalp.subj')} dev / {g('test.scalp.subj')} test",
            f"Scalp seizure detection\nCHB-MIT, Siena\n{g('dev.detect.subj')} dev / {g('test.detect.subj')} test",
            f"Discharge typing (VEPISET)\ngeneralized vs focal\n{g('dev.typing.subj')} dev / {g('test.typing.subj')} test"]),
        ("AURA", [
            "Representation\n18 physiological features\nper EEG window",
            "Patient anchoring\nz-score by the patient's\nown baseline EEG",
            "Classifier heads\nlogistic (19 parameters)\nor boosted trees",
            f"AURA-Net: learns band\nenvelopes and transients,\n{g('par.auranet_pure', '0.2 M').replace(chr(92) + ',', ' ')} parameters"]),
        ("Comparators: one recipe,\nidentical folds, matched input", [
            f"{g('design.n_braindecode')} published architectures\n(incl. BIOT, EEGPT, CBraMod\nfrom scratch)",
            "2 pretrained foundation\nmodels (BIOT, CBraMod),\nscratch controls, probes",
            "Repository CNN",
            "Factors varied\nfold protocol, seed, recipe,\npretraining, anchor placement"]),
        ("Evaluation", [
            "Paired contrasts\npatient + seed bootstrap,\nsign-flip, FDR, open bound",
            "Calibration\nmap fitted on held-out\npatients",
            "Continuous monitoring\n12-h streams at natural\nprevalence",
            f"Sealed test split\nSHA-256 {g('seal.sha')}\nscored once; + addendum"]),
    ]
    fig, ax = plt.subplots(figsize=(7.16, 3.05))
    ax.set_xlim(0, 4); ax.set_ylim(0, 4.55); ax.axis("off")
    fills = ["#eef4fc", "#fdf0ea", "#f2f2ef", "#eaf6f1"]
    edges = [BLUE, ORANGE, GRAY, AQUA]
    for ci, (head, items) in enumerate(cols):
        x0 = ci + 0.06
        ax.text(ci + 0.5, 4.38, head, ha="center", va="center", fontsize=7, color=INK, weight="bold")
        for ri, txt in enumerate(items):
            y0 = 3.2 - ri * 1.02
            ax.add_patch(FancyBboxPatch((x0, y0), 0.88, 0.86, boxstyle="round,pad=0.0,rounding_size=0.06",
                                        fc=fills[ci], ec=edges[ci], lw=0.8))
            ax.text(x0 + 0.44, y0 + 0.43, txt, ha="center", va="center", fontsize=5.6, color=INK,
                    linespacing=1.25)
        if ci < 3:
            ax.add_patch(FancyArrowPatch((ci + 0.955, 2.0), (ci + 1.045, 2.0), arrowstyle="-|>",
                                         mutation_scale=8, color=INK2, lw=0.8))
    fig.savefig(OUT / "fig_design.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_design.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

def fig_scaling():
    """Cross-patient AUC of the analytic vector against the number of training patients."""
    d = latest("patient_scaling_*")
    R = json.loads((d / "results.json").read_text())
    series = [("typing", "feat_gbm", ORANGE, "o"), ("detect", "feat_logistic", INDIGO, "s"),
              ("scalp", "feat_gbm", BLUE, "^"), ("ieeg", "feat_gbm", OCHRE, "D")]
    fig, ax = plt.subplots(figsize=(3.45, 2.4))
    ax.axhline(0.5, color=INK2, lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.grid(color=GRID, lw=0.5, zorder=0)
    for t, m, col, mk in series:
        c = R[t][m]["curve"]
        ks = sorted(int(k) for k in c)
        ys = [c[str(k)]["mean"] for k in ks]
        sd = [c[str(k)]["sd"] for k in ks]
        ax.fill_between(ks, np.array(ys) - np.array(sd), np.array(ys) + np.array(sd), color=col,
                        alpha=0.12, lw=0, zorder=2)
        ax.plot(ks, ys, color=col, lw=1.4, marker=mk, ms=3.5, mec=SURF, mew=0.5, zorder=3,
                label=f"{TASK_NAME[t]} ({'GBM' if m == 'feat_gbm' else 'logistic'})")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32]); ax.set_xticklabels(["2", "4", "8", "16", "32"])
    ax.set_xlabel("Training patients", fontsize=7); ax.set_ylabel("Patient-independent AUC", fontsize=7)
    ax.set_ylim(0.45, 0.9)
    ax.legend(fontsize=5.8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_scaling.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_scaling.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
