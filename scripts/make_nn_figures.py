#!/usr/bin/env python
"""Figures of the Neural Networks manuscript, drawn with the authors' Nature-style toolkit
(vendor/visual-design: sciplots for charts, schematic for diagrams).

One colour per entity in every figure: AURA (physiological features) terracotta; published networks
grey; pretrained foundation models blue; patient-relative adaptation (rolling reference, PBN) teal;
AURA-Ens plum; IEDs sage; annotated seizures a tint of the clinical 'abnormal' red. Every number
comes from the same results files as manuscript/generated/numbers.json.

  python scripts/make_nn_figures.py [fig1 fig2 ...]      -> manuscript/figures/nn_fig*.{pdf,png}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.lines  # noqa: E402,F401
import matplotlib.transforms  # noqa: E402,F401
import matplotlib.cm  # noqa: E402,F401
import matplotlib.ticker  # noqa: E402,F401
import matplotlib.colors  # noqa: E402,F401
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "vendor" / "visual-design" / "scripts"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import schematic as sc  # noqa: E402
import sciplots as sp  # noqa: E402
from figtools import find_text_overlaps  # noqa: E402

OUT = REPO / "manuscript" / "figures"
N = json.loads((REPO / "manuscript" / "generated" / "numbers.json").read_text())

AURA, NET, FM, ADAPT, ENS, IED = sp.TERRA, sp.GREY, sp.BLUE, sp.TEAL, sp.PLUM, sp.SAGE
SEIZ = sp.tint(sp.ABNORMAL, 0.72)


def n(key, default="?"):
    return str(N.get(key, default)).replace("\\,", " ").replace("$-$", "−")


def mbox(ax, x, y, w, h, lines, color=None, style="tint", lw=0.5, bold_first=True, size=0.0, align="center", sub_delta=-0.7):
    """Component box with several lines of text (first line optionally bold), centred vertically."""
    sc.box(ax, x, y, w, h, None, color=color or sp.GREY, style=style, lw=lw)
    k = len(lines); lh = 2.45 + 0.35 * size
    y0 = y + h / 2 + (k - 1) * lh / 2
    for i, s in enumerate(lines):
        sc.text(ax, x + w / 2 if align == "center" else x + 1.6, y0 - i * lh, s,
                size=size if i == 0 else size + sub_delta, weight="bold" if (i == 0 and bold_first) else "normal",
                color=("white" if style == "solid" else (sp.INK if i == 0 else sp.SECOND)), ha=align, zorder=4)


def stage(ax, x, y, w, h, title, sub):
    sc.group(ax, x, y, w, h, title)
    sc.text(ax, x + 2.2, y + h - 6.6, sub, size=-0.8, color=sp.SECOND, va="top")


def _check(fig, name):
    ov = find_text_overlaps(fig)
    sp.save(fig, OUT / name)
    plt.close(fig)
    print(f"{name}: {'ok' if not ov else 'OVERLAPS ' + repr(ov[:6])}")


# ====================================================================================== AURA-Net diagram
# Every shape, count and hyper-parameter below is the scalp configuration of bsf.models.aura_net.AURANet
# (fs 250 Hz, 8-s sub-windows, 18-channel double banana, hybrid with the 18 AURA features); parameter
# counts are read from the module, so the diagram cannot drift from the code.
AN_FS, AN_T, AN_M, AN_R, AN_K, AN_STEP, AN_SUB = 250, 2000, 18, 8, 40, 16, 6


def _an_params():
    from bsf.models.aura_net import AURANet
    m = AURANet(fs=float(AN_FS), d_feat=18, anchor=False, use_core=False, hazard_bias_init=0.0)
    c = lambda mod: sum(p.numel() for p in mod.parameters())  # noqa: E731
    hz = m.hazard
    out = {"filt": c(m.filt) + c(m.trans), "mix": c(m.mixer), "tcn": c(m.chan), "att": c(m.pool),
           "fuse": c(m.feat_proj) + c(m.feat_gate), "head": c(hz.f_eeg) + c(hz.f_feat) + 1 + 1}
    out["total"] = sum(out.values())
    return out, m


def _slabs(ax, x, y, w, h, n, ec, off=0.65, lw=0.4, back_ec=None):
    """A tensor drawn as n stacked feature maps (front face at (x, y), later maps offset up-right)."""
    for i in range(n - 1, -1, -1):
        ax.add_patch(Rectangle((x + i * off, y + i * off), w, h, fc="white",
                               ec=ec if (i == 0 or back_ec is None) else back_ec, lw=lw, zorder=3 + 0.01 * (n - i)))


def _cells(ax, x, y, w, h, vals, cmap, gap=0.1, z=3.6):
    vals = np.atleast_2d(vals); nr, nc = vals.shape; cw, ch = w / nc, h / nr
    for i in range(nr):
        for j in range(nc):
            ax.add_patch(Rectangle((x + j * cw + gap / 2, y + h - (i + 1) * ch + gap / 2), cw - gap, ch - gap,
                                   fc=cmap(float(vals[i, j])), ec="none", zorder=z))


def _node(ax, x, y, r, fc="white", ec=None, lw=0.4, z=5):
    ax.add_patch(plt.Circle((x, y), r, fc=fc, ec=ec or sp.AXIS, lw=lw, zorder=z))


def _card(ax, x, y, w, h, num, title, color):
    sc.box(ax, x, y, w, h, None, color=color, style="outline", lw=0.5)
    sc.badge(ax, x + 2.4, y + h - 2.6, str(num), color=color if color != sp.GREY else sp.AXIS, r=1.25)
    sc.text(ax, x + 4.4, y + h - 2.6, title, size=0.0, weight="bold")


def _card_text(ax, x, y, lines, size=-0.8, lh=2.45, color=None):
    for j, s in enumerate(lines):
        sc.text(ax, x, y - j * lh, s, size=size, color=color or sp.INK, va="center")


def auranet_panel(ax, X0, X1, Y0, Y1):
    """AURA-Net as a tensor-flow diagram (top), the learning signal, and one card per operation (bottom)."""
    P, model = _an_params()
    rng = np.random.default_rng(11)
    sc.group(ax, X0, Y0, X1 - X0, Y1 - Y0,
             f"AURA-Net: physiology-informed network (hybrid configuration, {P['total'] / 1e6:.2f} M parameters)")
    terr_c = sp.seq_cmap(AURA, light=0.95); grey_c = sp.seq_cmap(sp.AXIS, light=0.95)

    # ---- key (top right): block colours and connector types
    kx, ky = X1 - 76.0, Y1 - 3.0
    sc.swatch_key(ax, kx, ky, [((sp.tint(AURA, 0.86), AURA), "physiology-structured"),
                               ((sp.tint(sp.GREY, 0.86), sp.AXIS), "generic, learned"),
                               ((sp.tint(ADAPT, 0.86), ADAPT), "patient-relative (optional)")],
                  ncol=3, col_w=25.5, size=-1.0)
    for i, (c, ls, lab) in enumerate([(sp.AXIS, "-", "forward pass"), (AURA, "-", "AURA features"),
                                      (sp.INK, (0, (2.2, 1.4)), "gradient (learning)")]):
        xx = kx + i * 25.5
        ax.plot([xx, xx + 2.4], [ky - 3.05, ky - 3.05], color=c, lw=0.7, ls=ls, zorder=4)
        sc.text(ax, xx + 3.3, ky - 3.05, lab, size=-1.0)

    yc = Y1 - 21.0                       # trunk centre line
    yg = Y1 - 9.8                        # gradient line
    yb = yc - 12.6                       # physiology (feature) branch
    ylab = yc - 8.6                      # tensor dimension labels
    learn_x = []                         # x of every learned operation (gradient taps)

    def dims(x, s):
        sc.text(ax, x, ylab, s, size=-1.0, color=sp.SECOND, ha="center")

    def above(x, y, s):
        sc.text(ax, x, y, s, size=-1.0, color=sp.SECOND, ha="center")

    def op(x0, x1, num, learned=True):
        sc.arrow(ax, [(x0, yc), (x1, yc)], lw=0.6, head=1.0, width=0.85, zorder=2)
        xm = (x0 + x1) / 2 - 0.3
        sc.badge(ax, xm, yc, str(num), color=sp.AXIS, r=1.25)
        if learned:
            learn_x.append((xm, yc + 1.25))

    # ---- trunk: tensors drawn to their shape (height ~ features, width ~ time, depth = channels)
    x = X0 + 3.0
    w0, h0 = 11.5, 11.0
    _slabs(ax, x, yc - h0 / 2, w0, h0, 3, ec=sp.AXIS, back_ec=sp.FAINT)
    sc.eeg_traces(ax, x + 0.6, yc - h0 / 2 + 0.5, w0 - 1.2, h0 - 1.0, n=5, color=sp.INK, lw=0.3, seed=3)
    for z_ in ax.lines[-5:]:
        z_.set_zorder(3.8)
    above(x + w0 / 2 + 0.65, yc + h0 / 2 + 2.9, f"{AN_SUB} sub-windows, 8 s")
    dims(x + w0 / 2, f"{AN_M} × {AN_T}")
    x_in = x + w0 / 2
    x = x + w0 + 1.6
    op(x, x + 6.0, 1); x += 6.6

    # band-signal tensor: one stripe per band, frequency rising downward
    def band_face(x, y, w, h, nb=6):
        t = np.linspace(0, 1, 200)
        for b in range(nb):
            yy = y + h - (b + 0.5) * h / nb
            f = [2, 4, 7, 12, 20, 32][b]
            ax.plot(x + 0.4 + t * (w - 0.8), yy + 0.32 * h / nb * np.sin(2 * np.pi * f * t + b), color=AURA,
                    lw=0.3, zorder=3.8)
            if b:
                ax.plot([x, x + w], [y + h - b * h / nb] * 2, color=sp.HAIR, lw=0.3, zorder=3.7)

    w1, h1 = 11.0, 9.0
    _slabs(ax, x, yc - h1 / 2, w1, h1, 4, ec=AURA, back_ec=sp.tint(AURA, 0.6))
    band_face(x, yc - h1 / 2, w1, h1)
    dims(x + w1 / 2 + 1.0, f"{AN_M} × {AN_K} × {AN_T}")
    above(x + w1 / 2 + 1.0, yc + h1 / 2 + 3.6, f"{AN_K} bands")
    x = x + w1 + 2.6
    op(x, x + 6.0, 2); x += 6.6
    _slabs(ax, x, yc - h1 / 2, w1, h1, 5, ec=AURA, back_ec=sp.tint(AURA, 0.6))
    for pt in ax.patches[-5:-3]:
        pt.set_edgecolor(sp.AXIS)                  # the two back maps: the virtual channels
    band_face(x, yc - h1 / 2, w1, h1)
    dims(x + w1 / 2 + 1.3, f"{AN_M + AN_R} × {AN_K} × {AN_T}")
    above(x + w1 / 2 + 1.3, yc + h1 / 2 + 4.3, f"+{AN_R} virtual")
    x = x + w1 + 3.2
    op(x, x + 6.0, 3, learned=False); x += 6.6

    # pooled views: coarser in time (125 steps), taller in features (200)
    nT = AN_T // AN_STEP
    w3, h3 = 7.0, 12.0
    _slabs(ax, x, yc - h3 / 2, w3, h3, 5, ec=AURA, back_ec=sp.AXIS)
    v = np.abs(np.sin(np.linspace(0, 3, 10))[:, None] * np.cos(np.linspace(0, 5, 8))[None, :]) * 0.8
    v[:, 5] += 0.5
    _cells(ax, x + 0.35, yc - h3 / 2 + 0.35, w3 - 0.7, h3 - 0.7, np.clip(v, 0, 1), terr_c)
    dims(x + w3 / 2 + 1.3, f"{AN_M + AN_R} × {AN_K * 5} × {nT}")
    above(x + w3 / 2 + 1.3, yc + h3 / 2 + 3.6, "5 views")
    x = x + w3 + 3.2
    op(x, x + 6.0, 4); x += 6.6
    w4, h4 = 7.0, 9.5
    _slabs(ax, x, yc - h4 / 2, w4, h4, 5, ec=sp.AXIS, back_ec=sp.FAINT)
    v = rng.uniform(0, 1, (7, 8)); v = (v + np.roll(v, 1, 1)) / 2
    _cells(ax, x + 0.35, yc - h4 / 2 + 0.35, w4 - 0.7, h4 - 0.7, v, grey_c)
    dims(x + w4 / 2 + 1.3, f"{AN_M + AN_R} × 64 × {nT}")
    above(x + w4 / 2 + 1.3, yc + h4 / 2 + 4.3, "hidden")
    x = x + w4 + 3.2
    sc.arrow(ax, [(x, yc), (x + 4.6, yc)], lw=0.6, head=1.0, width=0.85, zorder=2)
    x += 5.2

    # channel tokens: one 128-d vector per channel (18 electrodes + 8 virtual)
    nc = AN_M + AN_R; ht = 12.5; wt = 2.6
    for i in range(nc):
        yy = yc + ht / 2 - (i + 1) * ht / nc
        ax.add_patch(Rectangle((x, yy + 0.06), wt, ht / nc - 0.12, fc=sp.tint(sp.AXIS, 0.45 if i < AN_M else 0.85),
                               ec="none", zorder=3.6))
    above(x + wt / 2, yc + ht / 2 + 1.6, "tokens")
    dims(x + wt / 2, f"{nc} × 128")
    x_tok = x
    x = x + wt + 1.0
    op(x, x + 6.0, 5); x += 6.6

    def vec(x, h, cmap, n=12, seed=0):
        vv = np.random.default_rng(seed).uniform(0.15, 0.9, (n, 1))
        _cells(ax, x, yc - h / 2, 1.8, h, vv, cmap, gap=0.12)
        ax.add_patch(Rectangle((x, yc - h / 2), 1.8, h, fc="none", ec=sp.AXIS, lw=0.35, zorder=3.7))

    vec(x, 10.0, grey_c, seed=1)
    sc.text(ax, x + 0.9, yc + 6.4, "$z$", size=0.0, ha="center")
    dims(x + 0.9, "128")
    x_z = x + 1.8

    # optional patient-baseline whitening on the embedding
    xb0 = x_z + 1.4
    ax.add_patch(FancyBboxPatch((xb0, yc - 2.1), 6.2, 4.2, boxstyle="round,pad=0,rounding_size=0.6", fc=sp.tint(ADAPT, 0.88),
                                ec=ADAPT, lw=0.5, ls=(0, (2.0, 1.2)), zorder=3))
    sc.text(ax, xb0 + 3.1, yc - 0.1, r"$\Sigma_p^{-1/2}$", size=-0.6, ha="center", zorder=4)
    sc.text(ax, xb0 + 3.1, yc + 3.6, "baseline", size=-1.0, color=ADAPT, ha="center")
    ax.plot([x_z + 0.1, xb0], [yc, yc], color=sp.AXIS, lw=0.6, zorder=2)

    # gated fusion with the physiology branch
    xg = xb0 + 6.2 + 4.2
    sc.arrow(ax, [(xb0 + 6.2, yc), (xg - 1.9, yc)], lw=0.6, head=1.0, width=0.85, zorder=2)
    _node(ax, xg, yc, 1.9, fc=sp.tint(AURA, 0.86), ec=AURA, lw=0.5)
    sc.text(ax, xg, yc, "$g$", size=0.0, ha="center", zorder=6)
    sc.badge(ax, xg, yc + 4.3, "6", color=AURA, r=1.25)
    learn_x.append((xg, yc + 4.3 + 1.25))

    xzt = xg + 1.9 + 3.6
    sc.arrow(ax, [(xg + 1.9, yc), (xzt - 0.1, yc)], lw=0.6, head=1.0, width=0.85, zorder=2)

    # output head drawn as layers of units: 128 -> 64 -> 1, plus the linear skip from f
    nA, nB = 7, 5; hA, hB = 10.0, 7.0
    xa, xh, xs_ = xzt + 0.9, xzt + 7.4, xzt + 13.4
    ya = yc + hA / 2 - np.arange(nA) * hA / (nA - 1)
    yh = yc + hB / 2 - np.arange(nB) * hB / (nB - 1)
    for a_ in ya:
        for h_ in yh:
            ax.plot([xa, xh], [a_, h_], color=sp.FAINT, lw=0.25, zorder=2)
    for h_ in yh:
        ax.plot([xh, xs_], [h_, yc], color=sp.FAINT, lw=0.3, zorder=2)
    mixc = matplotlib.colors.LinearSegmentedColormap.from_list("mix", [sp.tint(sp.AXIS, 0.55), sp.tint(AURA, 0.35)])
    for j, a_ in enumerate(ya):                    # each unit a per-dimension mix of z (grey) and z_f
        _node(ax, xa, a_, 0.55, fc=mixc([0.9, 0.2, 0.6, 0.05, 0.75, 0.35, 0.5][j]), ec="none")
    for h_ in yh:
        _node(ax, xh, h_, 0.55, fc="white", ec=sp.AXIS)
    _node(ax, xs_, yc, 1.5, fc="white", ec=sp.INK, lw=0.5)
    sc.text(ax, xs_, yc - 0.05, r"$\Sigma$", size=-0.6, ha="center", zorder=6)
    sc.text(ax, xa, yc + hA / 2 + 1.6, r"$\tilde z$", size=0.0, ha="center")
    dims(xa, "128"); dims(xh, "64")
    sc.text(ax, xh, yc - hB / 2 - 1.6, "GELU", size=-1.0, color=sp.SECOND, ha="center")
    learn_x.append((xh, yc + hB / 2 + 0.55))

    # sub-window logits: training loss on each, softmax pooling for the clip score at inference
    xl = xs_ + 1.5
    xsplit = xl + 3.0
    sc.text(ax, xl + 1.3, yc + 1.5, r"$\ell_s$", size=-0.4, ha="center")
    ax.plot([xl, xsplit], [yc, yc], color=sp.AXIS, lw=0.6, zorder=2)
    # inference: softmax pool of the six sub-window logits
    xp0 = xsplit + 2.0; wp = 15.0
    sc.arrow(ax, [(xsplit, yc), (xp0 - 0.2, yc)], lw=0.6, head=1.0, width=0.85, zorder=2)
    lv = np.array([-0.6, -0.3, 0.4, 1.6, 1.1, -0.2]); wv = np.exp(lv / 0.6); wv /= wv.sum()
    bw = wp / AN_SUB
    base = yc - 3.6
    for i, (l_, w_) in enumerate(zip(lv, wv)):
        hh = 1.6 + 1.9 * (l_ - lv.min())
        ax.add_patch(Rectangle((xp0 + i * bw + 0.25, base), bw - 0.5, hh, fc=sp.tint(sp.INK, 0.85 - 0.75 * w_ / wv.max()),
                               ec="none", zorder=3))
    ax.plot([xp0, xp0 + wp], [base, base], color=sp.AXIS, lw=0.4, zorder=3)
    sc.text(ax, xp0 + wp / 2 + 5.5, base - 1.6, "inference: softmax pool of the 6", size=-1.0, color=sp.SECOND, ha="center")
    sc.text(ax, xp0 + wp / 2 + 5.5, base - 3.8, r"sub-window logits $\to$ clip probability", size=-1.0, color=sp.SECOND, ha="center")
    xo = xp0 + wp + 4.0
    sc.arrow(ax, [(xp0 + wp + 0.4, yc), (xo - 1.6, yc)], lw=0.6, head=1.0, width=0.85, zorder=2)
    _node(ax, xo, yc, 1.5, fc="white", ec=sp.INK, lw=0.5)
    sc.text(ax, xo, yc - 0.05, r"$\sigma$", size=-0.6, ha="center", zorder=6)
    sc.text(ax, xo + 2.2, yc, "$p$", size=0.4, ha="left", weight="bold")

    # training: loss on every sub-window logit; its gradient flows back through every learned block
    xL0, xL1 = xsplit - 5.0, X1 - 2.0
    sc.arrow(ax, [(xsplit, yc), (xsplit, yg - 1.9)], color=sp.INK, lw=0.6, head=1.0, width=0.85, zorder=2)
    sc.box(ax, xL0, yg - 1.8, xL1 - xL0, 3.6, None, color=sp.INK, style="outline", lw=0.6)
    sc.text(ax, (xL0 + xL1) / 2, yg, r"loss: weighted cross-entropy, every $\ell_s$ vs $y$", size=-1.0, ha="center")
    xgrad0 = min(x_ for x_, _ in learn_x)
    ax.plot([xL0, xgrad0], [yg, yg], color=sp.INK, lw=0.6, ls=(0, (2.2, 1.4)), zorder=2)
    for xx, ytop in learn_x:
        sc.arrow(ax, [(xx, yg), (xx, ytop + 0.35)], color=sp.INK, lw=0.5, head=0.9, width=0.75, dashed=True, zorder=2)
    sc.text(ax, (xgrad0 + learn_x[2][0]) / 2, yg + 1.5, r"gradient $\partial L/\partial\theta$ reaches every learned block",
            size=-1.0, color=sp.INK, ha="center")

    # physiology branch: the 18 AURA features of the same clip, into the gate and straight to the logit
    xf0 = x_in - 4.6
    ax.plot([xf0, xf0], [yc - h0 / 2 - 0.2, yb], color=AURA, lw=0.7, zorder=2)
    fx = xf0 + 16.0
    ax.plot([xf0, xs_], [yb, yb], color=AURA, lw=0.7, zorder=2)
    fv = rng.uniform(0.2, 1.0, 18)
    for i, val in enumerate(fv):
        ax.add_patch(Rectangle((fx + i * 0.75, yb - 0.85), 0.62, 1.7, fc=terr_c(val), ec="none", zorder=3.6))
    ax.add_patch(Rectangle((fx, yb - 0.85), 18 * 0.75 - 0.13, 1.7, fc="none", ec=AURA, lw=0.35, zorder=3.7))
    sc.text(ax, fx, yb - 2.1, r"AURA features $f\in\mathbb{R}^{18}$ (same clip)", size=-1.0, color=AURA, va="center")
    sc.arrow(ax, [(xg, yb), (xg, yc - 1.9)], color=AURA, lw=0.7, head=1.0, width=0.85, zorder=2)
    sc.text(ax, xg + 0.8, yb + 2.4, r"$z_f$", size=-0.6, color=AURA, ha="left")
    sc.arrow(ax, [(xs_, yb), (xs_, yc - 1.5)], color=AURA, lw=0.7, head=1.0, width=0.85, zorder=2)
    sc.text(ax, xs_ + 0.8, yb + 2.4, r"$f^{\mathrm{T}}w_f$", size=-0.6, color=AURA, ha="left")

    # ---- cards: what each numbered operation computes, and what it learns
    cy0, cy1 = Y0 + 1.6, yb - 3.6
    ch = cy1 - cy0
    widths = [27.0, 26.0, 26.0, 28.5, 26.5]
    widths.append((X1 - X0 - 3.2) - sum(widths) - 1.6 * 5)
    cxs = [X0 + 1.6]
    for wd in widths[:-1]:
        cxs.append(cxs[-1] + wd + 1.6)
    gtop = cy1 - 5.4                               # glyph area top
    gbot = cy0 + 14.2                              # glyph area bottom
    tl = cy0 + 9.6                                 # first text line

    # 1. filterbank: magnitude responses of the 32 sinc band-pass filters at initialization
    x, wd = cxs[0], widths[0]
    _card(ax, x, cy0, wd, ch, 1, "Learnable filterbank", AURA)
    k = model.filt.filters().detach().cpu().numpy()[:, 0, :]
    Hm = np.abs(np.fft.rfft(k, 4096, axis=1)); fr = np.fft.rfftfreq(4096, 1 / AN_FS)
    Hm /= Hm.max(1, keepdims=True)
    lo_f, hi_f = 0.7, AN_FS / 2
    sel = (fr >= lo_f) & (fr <= hi_f)
    u = (np.log10(fr[sel]) - np.log10(lo_f)) / (np.log10(hi_f) - np.log10(lo_f))
    gx, gw, gy, gh = x + 2.0, wd - 6.0, gbot + 1.8, gtop - gbot - 3.6
    for i in range(k.shape[0] - 1, -1, -1):
        clin = i < 5
        ax.plot(gx + u * gw, gy + Hm[i, sel] * gh, color=AURA if clin else sp.FAINT, lw=0.55 if clin else 0.3,
                zorder=4 if clin else 3)
    ax.plot([gx, gx + gw], [gy, gy], color=sp.AXIS, lw=0.4, zorder=4)
    for fv_, lab in [(1, "1"), (10, "10"), (100, "100 Hz")]:
        uu = (np.log10(fv_) - np.log10(lo_f)) / (np.log10(hi_f) - np.log10(lo_f))
        ax.plot([gx + uu * gw] * 2, [gy, gy - 0.6], color=sp.AXIS, lw=0.4)
        sc.text(ax, gx + uu * gw, gy - 1.6, lab, size=-1.0, color=sp.SECOND, ha="center")
    for fc_, g_ in [(2.0, r"$\delta$"), (5.7, r"$\theta$"), (10.2, r"$\alpha$"), (20, r"$\beta$"), (46, r"$\gamma$")]:
        uu = (np.log10(fc_) - np.log10(lo_f)) / (np.log10(hi_f) - np.log10(lo_f))
        sc.text(ax, gx + uu * gw, gy + gh + 1.1, g_, size=-0.8, color=AURA, ha="center")
    _card_text(ax, x + 1.6, tl, ["32 sinc band-pass filters,", "2 learned edges each; 8 free", "FIR filters for spike shape"])
    sc.text(ax, x + 1.6, cy0 + 1.8, f"{P['filt']:,} parameters", size=-1.0, color=sp.SECOND)

    # 2. spatial mixer: every electrode feeds every virtual channel with a signed weight
    x, wd = cxs[1], widths[1]
    _card(ax, x, cy0, wd, ch, 2, "Spatial mixer", sp.GREY)
    gx0, gx1 = x + 9.4, x + wd - 7.0
    ye = np.linspace(gtop - 0.6, gbot + 0.2, AN_M); yv = np.linspace(gtop - 2.2, gbot + 1.8, AN_R)
    for a_ in ye:
        for b_ in yv:
            ax.plot([gx0, gx1], [a_, b_], color=sp.HAIR, lw=0.2, zorder=2)
    hi_e = {6: 1.0, 7: -0.8, 11: 0.35, 5: -0.4}
    for e_, wv_ in hi_e.items():
        ax.plot([gx0, gx1], [ye[e_], yv[3]], color=sp.INK, lw=0.25 + 0.7 * abs(wv_), ls="-" if wv_ > 0 else (0, (1.6, 1.0)),
                zorder=3)
    for a_ in ye:
        _node(ax, gx0, a_, 0.3, fc=sp.tint(sp.AXIS, 0.45), ec="none")
    for j, b_ in enumerate(yv):
        _node(ax, gx1, b_, 0.42, fc="white", ec=sp.INK if j == 3 else sp.AXIS, lw=0.6 if j == 3 else 0.35)
    sc.text(ax, gx0 - 0.8, (ye[0] + ye[-1]) / 2, f"{AN_M}\nelectrodes", size=-1.0, color=sp.SECOND, ha="right",
            linespacing=1.0)
    sc.text(ax, gx1 + 0.9, (yv[0] + yv[-1]) / 2, f"{AN_R}\nvirtual", size=-1.0, color=sp.SECOND, ha="left",
            linespacing=1.0)
    _card_text(ax, x + 1.6, tl, [r"$v_r=\Sigma_m\, a_{rm}x_m$, signed", "(solid +, dashed −); weights", "from content, not position"])
    sc.text(ax, x + 1.6, cy0 + 1.8, f"{P['mix']:,} parameters", size=-1.0, color=sp.SECOND)

    # 3. band views: rectify and pool every 64 ms (fixed, no parameters)
    x, wd = cxs[2], widths[2]
    _card(ax, x, cy0, wd, ch, 3, "Band views", AURA)
    t = np.arange(250) / AN_FS
    r3 = np.random.default_rng(5)
    s = 0.55 * np.sin(2 * np.pi * 9 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.8 * t)) + 0.12 * r3.standard_normal(t.size)
    sp_ = (t > 0.52) & (t < 0.58); s[sp_] += -1.9 * np.sin(np.pi * (t[sp_] - 0.52) / 0.06)
    s = np.convolve(s, np.ones(3) / 3, "same")
    gx, gw = x + 2.0, wd - 4.0
    ymid = (gtop + gbot) / 2 + 0.3; sy = (gtop - gbot) / 5.4
    ax.plot(gx + t / t[-1] * gw, ymid + s * sy, color=sp.FAINT, lw=0.35, zorder=3)
    nb = 250 // AN_STEP
    env = np.array([np.abs(s[i * AN_STEP:(i + 1) * AN_STEP]).mean() for i in range(nb)])
    mx = np.array([s[i * AN_STEP:(i + 1) * AN_STEP][np.argmax(np.abs(s[i * AN_STEP:(i + 1) * AN_STEP]))] for i in range(nb)])
    xs_b = gx + (np.arange(nb + 1) * AN_STEP) / 249 * gw
    ax.step(xs_b, np.r_[env, env[-1]] * sy + ymid, where="post", color=AURA, lw=0.65, zorder=4)
    ax.plot((xs_b[:-1] + xs_b[1:]) / 2, ymid + mx * sy, ls="none", marker="o", ms=1.2, mfc=sp.INK, mec="none", zorder=4)
    for i in range(nb + 1):
        ax.plot([xs_b[i]] * 2, [gbot - 0.4, gbot + 0.2], color=sp.FAINT, lw=0.25, zorder=2)
    sc.text(ax, gx + gw, gbot - 1.4, "64-ms steps", size=-1.0, color=sp.SECOND, ha="right")
    _card_text(ax, x + 1.6, tl, ["envelope (line), signed max", "(dots), signed mean, 8-ms", "peak and spread: 5 views"])
    sc.text(ax, x + 1.6, cy0 + 1.8, "fixed, no parameters", size=-1.0, color=sp.SECOND)

    # 4. dilated causal TCN: the dependency fan of the newest output, block by block
    x, wd = cxs[3], widths[3]
    _card(ax, x, cy0, wd, ch, 4, "Dilated causal TCN", sp.GREY)
    gx, gw = x + 5.6, wd - 7.2
    rows = np.linspace(gbot + 0.6, gtop - 0.4, 6)
    tx = lambda tt: gx + tt / (nT - 1) * gw  # noqa: E731
    tend = nT - 1
    for r_ in range(6):
        span = 1 + 6 * (2 ** 5 - 2 ** r_) if r_ < 5 else 1
        lo_t = max(0, tend - span + 1)
        ax.plot([tx(0), tx(tend)], [rows[r_]] * 2, color=sp.HAIR, lw=0.9, solid_capstyle="butt", zorder=2)
        ax.plot([tx(lo_t), tx(tend)], [rows[r_]] * 2, color=sp.tint(sp.AXIS, 0.55), lw=0.9, solid_capstyle="butt", zorder=2.5)
    for b in range(5):
        d = 2 ** b
        for kk in range(7):
            ax.plot([tx(tend), tx(tend - kk * d)], [rows[b + 1], rows[b]], color=sp.INK, lw=0.3 if d > 2 else 0.18, zorder=3)
            _node(ax, tx(tend - kk * d), rows[b], 0.28 if d > 2 else 0.16, fc=sp.INK, ec="none", z=4)
        sc.text(ax, gx - 0.9, (rows[b] + rows[b + 1]) / 2, f"{d}", size=-1.0, color=sp.SECOND, ha="right")
    _node(ax, tx(tend), rows[5], 0.4, fc=sp.INK, ec="none", z=4)
    sc.text(ax, gx - 3.4, (rows[0] + rows[5]) / 2, "dilation", size=-1.0, color=sp.SECOND, ha="center", rotation=90)
    sc.text(ax, gx + gw / 2, gbot - 1.2, f"{nT} steps (8 s); field {model.chan.receptive_field_s(AN_FS):.0f} s",
            size=-1.0, color=sp.SECOND, ha="center")
    _card_text(ax, x + 1.6, tl, ["5 residual blocks, kernel 7,", "shared over channels; readout:", "mean and last step, 128 each"])
    sc.text(ax, x + 1.6, cy0 + 1.8, f"{P['tcn']:,} parameters", size=-1.0, color=sp.SECOND)

    # 5. channel-set attention: 4 learned queries over the 26 channel tokens, plus synchrony
    x, wd = cxs[4], widths[4]
    _card(ax, x, cy0, wd, ch, 5, "Channel-set attention", sp.GREY)
    tx0 = x + 3.0; ttop, tbot = gtop - 0.2, gbot + 0.2
    yt = np.linspace(ttop, tbot, nc)
    qx = x + wd * 0.58; yq = np.linspace(ttop - 1.6, tbot + 3.2, 4)
    att = np.exp(-((np.arange(nc)[None, :] - np.array([[4], [6], [12], [20]])) / 2.2) ** 2)
    for qi, qy in enumerate(yq):
        for ti, ty_ in enumerate(yt):
            a_ = att[qi, ti]
            ax.plot([tx0 + 1.6, qx], [ty_, qy], color=sp.INK if a_ > 0.25 else sp.HAIR,
                    lw=0.15 + 0.45 * a_, alpha=0.25 + 0.6 * a_ if a_ > 0.25 else 1.0, zorder=3 if a_ > 0.25 else 2)
    for ti, ty_ in enumerate(yt):
        ax.add_patch(Rectangle((tx0, ty_ - 0.22), 1.6, 0.44, fc=sp.tint(sp.AXIS, 0.45 if ti < AN_M else 0.85), ec="none", zorder=4))
    for qy in yq:
        _node(ax, qx, qy, 0.6, fc="white", ec=sp.INK, lw=0.5)
    sx_, sy_ = qx, tbot + 0.4
    _node(ax, sx_, sy_, 0.75, fc=sp.tint(sp.AXIS, 0.8), ec=sp.AXIS)
    sc.text(ax, sx_ + 1.3, sy_, "$s$", size=-0.8, ha="left")
    zx = x + wd - 3.4
    for qy in list(yq) + [sy_]:
        ax.plot([qx + 0.6, zx], [qy, (ttop + tbot) / 2], color=sp.FAINT, lw=0.35, zorder=2)
    ax.add_patch(Rectangle((zx, (ttop + tbot) / 2 - 3.0), 1.4, 6.0, fc=sp.tint(sp.AXIS, 0.7), ec=sp.AXIS, lw=0.35, zorder=4))
    sc.text(ax, zx + 0.7, (ttop + tbot) / 2 + 4.2, "$z$", size=-0.4, ha="center")
    _card_text(ax, x + 1.6, tl, ["4 queries attend over the set", r"(any order); $s$ = synchrony;", r"weights $\alpha_m$ = channel map"])
    sc.text(ax, x + 1.6, cy0 + 1.8, f"{P['att']:,} parameters", size=-1.0, color=sp.SECOND)

    # 6. gated fusion, head and training objective (equations)
    x, wd = cxs[5], widths[5]
    _card(ax, x, cy0, wd, ch, 6, "Fusion, output and training", AURA)
    eq = [(r"$z_f=\mathrm{GELU}(W_f\,\mathrm{LN}(f))$", sp.INK),
          (r"$g=\sigma(W_g[z;z_f])$,  $\tilde z=g\odot z+(1-g)\odot z_f$", sp.INK),
          (r"$\ell_s=\mathrm{MLP}(\tilde z)+f^{\mathrm{T}}w_f+b$, per sub-window", sp.INK),
          (r"loss on every $\ell_s$ with the clip label $y$", sp.INK),
          (r"starts as logistic regression on $f$: MLP", sp.SECOND),
          (r"output zeroed, $w_f$ from the logistic fit", sp.SECOND),
          (r"AdamW, 25 epochs, cosine schedule", sp.SECOND),
          (r"optional: $\hat z=\Sigma_p^{-1/2}(z-\mu_p)$, patient baseline", ADAPT)]
    for j, (s_, c_) in enumerate(eq):
        sc.text(ax, x + 1.6, cy1 - 6.2 - j * 3.05, s_, size=-0.8, color=c_)
    sc.text(ax, x + 1.6, cy0 + 1.8, f"{P['fuse']:,} + {P['head']:,} parameters", size=-1.0, color=sp.SECOND)


# ====================================================================================== figure 1
def fig1():
    """Study design (top) and AURA-Net (bottom)."""
    sp.setup("nature")
    W, H = 183, 135
    fig, ax = sc.canvas(W, H)
    top = H - 1; mid = top - 53.5
    xs, w = [1.0, 47.0, 93.0, 139.0], 43.0

    # ---- 1. cohorts and tasks
    x = xs[0]
    stage(ax, x, mid, w, top - mid, "Cohorts and tasks", "development / held-out patients, split once")
    tasks = [("IED classification", "VEPISET, scalp", "typing"), ("iEEG seizure prediction", "Kaggle AES, NeuroVista", "ieeg"),
             ("Scalp seizure prediction", "CHB-MIT, Siena", "scalp"), ("Scalp seizure detection", "CHB-MIT, Siena", "detect")]
    for i, (t, c, k) in enumerate(tasks):
        yy = top - 18.0 - i * 9.4
        sc.box(ax, x + 2.2, yy, w - 4.4, 7.4, None, color=sp.GREY, style="outline")
        sc.text(ax, x + 3.8, yy + 4.9, t, size=-0.2, weight="bold")
        sc.text(ax, x + 3.8, yy + 2.1, c, size=-0.9, color=sp.SECOND)
        sc.text(ax, x + w - 3.6, yy + 3.7, f"{n(f'dev.{k}.subj')} / {n(f'test.{k}.subj')}", size=-0.4, ha="right")

    # ---- 2. representations
    x = xs[1]
    stage(ax, x, mid, w, top - mid, "Representations", "one recipe, identical patient-grouped folds")
    rng = np.random.default_rng(4)
    sc.bars(ax, x + 3.0, top - 19.5, 12.5, 7.0, rng.uniform(0.3, 1.0, 18), colors=[AURA] * 18, gap=0.12)
    mbox(ax, x + 17.0, top - 20.0, w - 19.2, 8.0, ["AURA", "18 physiological features"], color=AURA)
    sc.network(ax, x + 9.2, top - 27.6, 3.4, n=7, color=NET)
    mbox(ax, x + 17.0, top - 31.5, w - 19.2, 8.0, ["Published networks", f"{n('design.n_braindecode')} architectures + CNN"], color=NET)
    sc.eeg_traces(ax, x + 3.0, top - 43.0, 12.5, 7.5, n=4, color=FM, lw=0.35, seed=2)
    mbox(ax, x + 17.0, top - 43.0, w - 19.2, 8.0, ["Foundation models", "BIOT, CBraMod, pretrained"], color=FM)

    # ---- 3. evaluation
    x = xs[2]
    stage(ax, x, mid, w, top - mid, "Evaluation", "always on patients unseen in training")
    gx, gy, cw, ch = x + 3.0, top - 23.0, 1.75, 1.75
    for r in range(5):
        for c in range(7):
            test = c == (r * 3) % 7 or c == (r * 3 + 1) % 7
            ax.add_patch(Rectangle((gx + c * (cw + 0.35), gy + (4 - r) * (ch + 0.35)), cw, ch,
                                   fc=sp.INK if r == 4 else (sp.AXIS if test else sp.tint(sp.GREY, 0.6)), ec="none", zorder=3))
    sc.text(ax, gx + 7.4, gy - 1.4, "patients × time", size=-1.0, color=sp.SECOND, ha="center", va="top")
    sc.swatch_key(ax, gx, gy - 5.6, [(sp.AXIS, "test"), (sp.tint(sp.GREY, 0.6), "training"),
                                     (sp.INK, "patient held out")], ncol=3, col_w=10.5, size=-1.0)
    for i, s in enumerate(["5 data-splitting protocols", "patient-independent folds", "paired tests over patients"]):
        sc.text(ax, x + 20.0, top - 13.6 - i * 3.3, s, size=-0.8)
    chips = [("held-out patients, locked", sp.GREY), ("continuous EEG, 25 recordings", sp.GREY),
             ("external: SeizeIT2, wearable", sp.GREY), ("common 18-channel montage", sp.GREY)]
    for i, (s, c) in enumerate(chips):
        sc.chip(ax, x + 3.0, top - 35.2 - i * 4.4, s, color=c, size=-0.9, h=3.5)

    # ---- 4. patient-relative adaptation (the paper's focus)
    x = xs[3]
    stage(ax, x, mid, w, top - mid, "Patient-relative adaptation", "label-free, from the patient's own EEG")
    tx, ty, tw = x + 3.0, top - 22.0, w - 6.0
    ax.add_patch(Rectangle((tx + 0.10 * tw, ty - 0.6), 0.62 * tw, 8.6, fc=sp.tint(ADAPT, 0.8), ec="none", zorder=1))
    t = np.linspace(0, 1, 500); r2 = np.random.default_rng(7)
    sig = 0.35 * np.sin(2 * np.pi * 38 * t) + 0.25 * r2.standard_normal(t.size)
    sig = np.convolve(sig, np.ones(4) / 4, "same"); sig[t > 0.9] *= 2.6
    ax.plot(tx + t * tw, ty + 3.6 + sig * 1.25, color=sp.INK, lw=0.35, zorder=3)
    ax.add_patch(Rectangle((tx + 0.90 * tw, ty - 0.6), 0.07 * tw, 8.6, fc="none", ec=sp.ABNORMAL, lw=0.6, zorder=4))
    sc.text(ax, tx + 0.41 * tw, ty - 2.0, "reference: 35 to 5 min before", size=-1.0, color=ADAPT, ha="center", va="top")
    sc.text(ax, tx + 0.935 * tw, ty + 9.2, "now", size=-1.0, color=sp.ABNORMAL, ha="center")
    levels = [("features", "rolling ref."), ("embedding", "post hoc"), ("network", "PBN")]
    bw = (w - 6.0 - 2 * 1.2) / 3
    for i, (a, b) in enumerate(levels):
        mbox(ax, x + 3.0 + i * (bw + 1.2), top - 36.5, bw, 7.0, [a, b], color=ADAPT, size=-0.6, sub_delta=0.0)
    mbox(ax, x + 3.0, top - 46.0, w - 6.0, 6.5, ["AURA-Ens and AURA Monitor"], color=ENS, size=-0.2)
    for i in range(3):
        cx = x + 3.0 + i * (bw + 1.2) + bw / 2
        sc.arrow(ax, [(cx, top - 36.6), (cx, top - 39.4)], lw=0.5, head=0.9, width=0.8)

    for x0, x1 in zip(xs[:-1], xs[1:]):
        sc.arrow(ax, [(x0 + w + 0.3, mid + 24), (x1 - 0.3, mid + 24)], lw=0.7, head=1.4, width=1.1)

    # ---- 5. AURA-Net
    auranet_panel(ax, 1.0, W - 1.0, 1.0, mid - 3.0)
    _check(fig, "nn_fig1_overview")


DB18_POS = {"FP1": "Fp1", "FP2": "Fp2", "F7": "F7", "F3": "F3", "FZ": "Fz", "F4": "F4", "F8": "F8", "T7": "T3",
            "C3": "C3", "CZ": "Cz", "C4": "C4", "T8": "T4", "P7": "T5", "P3": "P3", "PZ": "Pz", "P4": "P4",
            "P8": "T6", "O1": "O1", "O2": "O2"}


def bipolar_head(ax, values, cmap, vmax, r=1.0, cx=0.0, cy=0.0, limits=True):
    """10-20 head (nose up, left on the left) centred at (cx, cy), with the 18 double-banana derivations
    drawn as links between their electrodes, shaded and thickened by value. limits=False draws into a
    shared axes (several heads in one panel)."""
    from bsf.monitor.signals import DB18
    ax.add_patch(plt.Circle((cx, cy), r, fill=False, ec=sp.AXIS, lw=0.5))
    ax.plot([cx - 0.1 * r, cx, cx + 0.1 * r], [cy + 0.995 * r, cy + 1.12 * r, cy + 0.995 * r], color=sp.AXIS, lw=0.5)
    for i in np.argsort(values):
        a_, b_ = DB18[i]
        (x1, y1), (x2, y2) = sc.POS[DB18_POS[a_]], sc.POS[DB18_POS[b_]]
        v = max(values[i], 0) / vmax
        ax.plot([cx + x1 * r, cx + x2 * r], [cy + y1 * r, cy + y2 * r], color=cmap(v), lw=0.8 + 2.6 * v,
                solid_capstyle="round", zorder=2 + v)
    for px, py in sc.POS.values():
        ax.add_patch(plt.Circle((cx + px * r, cy + py * r), 0.055 * r, fc="white", ec=sp.AXIS, lw=0.35, zorder=5))
    ax.text(cx - 1.08 * r, cy - 1.0 * r, "L", fontsize=5, color=sp.SECOND, ha="left")
    ax.text(cx + 1.08 * r, cy - 1.0 * r, "R", fontsize=5, color=sp.SECOND, ha="right")
    if limits:
        ax.set_xlim(cx - 1.15 * r, cx + 1.15 * r); ax.set_ylim(cy - 1.1 * r, cy + 1.2 * r)
        ax.set_aspect("equal"); ax.axis("off")


def oline(color, ls="-", lw=0.9, marker="o", ms=3.2, filled=True):
    """Key entry for a line with a filled or open marker (open markers keep a coloured edge)."""
    return matplotlib.lines.Line2D([], [], color=color, lw=lw, ls=ls, marker=marker, ms=ms,
                                   mfc=color if filled else "white", mec=color, mew=0.8)


# ====================================================================================== data helpers
def _res(pattern):
    from make_paper_numbers import latest
    d = latest(pattern)
    return (json.loads((d / "results.json").read_text()), d) if d else (None, None)


TASKS = ("typing", "ieeg", "scalp", "detect")
TASK_LABEL = {"typing": "IED classification", "ieeg": "iEEG seizure prediction",
              "scalp": "Scalp seizure prediction", "detect": "Scalp seizure detection"}
TASK_SHORT = {"typing": "IED\nclassif.", "ieeg": "iEEG\nprediction", "scalp": "scalp\nprediction", "detect": "scalp\ndetection"}
NET_LABEL = {"deep_cnn_raw": "Repository CNN", "auranet_hybrid": "AURA-Net", "feat_gbm": "AURA + GBM",
             "feat_logistic": "AURA + logistic", "ShallowFBCSPNet": "ShallowFBCSP", "aura": "AURA features"}
FEATS = ([f"{b} power" for b in ("δ", "θ", "α", "β", "γ", "HF")]
         + [f"{b} spread" for b in ("δ", "θ", "α", "β", "γ", "HF")]
         + ["line length", "synchrony", "spectral edge", "sharp transients", "kurtosis", "θ–γ PAC"])


def entity_color(m):
    if m.startswith("feat") or m == "aura":
        return AURA
    if m.startswith("auranet"):
        return AURA
    if m.startswith(("BIOT", "CBraMod", "EEGPT")):
        return FM
    if m in ("aura_ens", "ens_mean", "ensemble", "stacked"):
        return ENS
    return NET


def _ci(x, n=4000, rng=0):
    x = np.asarray(x, float)
    r = np.random.default_rng(rng)
    bs = x[r.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def _balance_rows(fig, gs, rows, tol=0.6, iters=40, step=0.6):
    """Give each row of a figure the height its fixed-shape panels need (pipelines drawn in mm, heads,
    squares), changing the figure height to match: a row whose locked panels have spare width grows, one
    whose locked panels have spare height shrinks. Damped steps; the best layout seen is kept.
    rows: one list of locked Axes per gridspec row."""
    W = fig.get_figwidth()

    def deltas():
        fig.canvas.draw(); fig.canvas.draw()          # constrained layout needs two passes to settle
        sl = {id(s_[0]): s_ for s_ in sp._slack(fig)}
        out = []
        for axs in rows:
            ss = [sl[id(a_)] for a_ in axs if id(a_) in sl]
            if not ss:
                out.append(0.0); continue
            sh = max(s_[1] for s_ in ss); sv = min(s_[2] for s_ in ss); r = max(s_[3] for s_ in ss)
            stack = len({round(s_[4].y0, 3) for s_ in ss})
            out.append(sh * r * stack if sh > tol else (-sv * stack if sv > tol else 0.0))
        return out

    best = None
    for _ in range(iters):
        d = deltas(); rat = list(gs.get_height_ratios() or [1] * len(rows)); Hmm = fig.get_figheight() / sp.MM
        score = sum(abs(x) for x in d)
        if best is None or score < best[0]:
            best = (score, rat, Hmm)
        if all(abs(x) < tol for x in d):
            break
        tot = sum(rat); hts = [r_ / tot * Hmm for r_ in rat]
        gs.set_height_ratios([max(8.0, h_ + step * d_) for h_, d_ in zip(hts, d)])
        fig.set_size_inches(W, min(240.0, max(30.0, Hmm + step * sum(d))) * sp.MM)
    gs.set_height_ratios(best[1]); fig.set_size_inches(W, best[2] * sp.MM); fig.canvas.draw(); fig.canvas.draw()


def _letter_finish(fig, panels, name, fit=False, balance=None):
    if balance:                       # letters take layout space: letter first, then balance the rows again
        sp.label_panels(panels); _balance_rows(fig, *balance)
        probs = sp.finish(fig, panels, OUT / name, letters="", fit=False)
    else:
        probs = sp.finish(fig, panels, OUT / name, fit=fit)
    plt.close(fig)
    print(f"{name}: {'ok' if not probs else 'problems ' + repr(probs[:4])}")


# ====================================================================================== figure 2
def fig2():
    """Data-splitting protocol against architecture."""
    from make_paper_numbers import latest
    sp.setup("nature")
    P = json.loads((latest("protocol_decomp_*") / "results.json").read_text())
    dv = latest("variant_stats_*")
    V = json.loads((dv / "results.json").read_text()) if dv else {}
    xpos = {"W-rand": 0.0, "W-event": 1.0, "P-seg": 2.3, "P-event": 3.3, "X": 4.3}
    labels = {"W-rand": "PS\nseg.", "W-event": "PS\nevent", "P-seg": "pool\nseg.", "P-event": "pool\nevent", "X": "PI"}
    fig = sp.figure("double", 104)
    r1, r2 = sp.grid(fig, [[1, 1, 1, 1], [1, 1]], height_ratios=[1.25, 0.8])
    axs = r1 + [r2[0]]
    order = ("scalp", "ieeg", "detect", "typing")
    for ax, t in zip(axs[:4], order):
        ax.axhline(0.5, color=sp.FAINT, lw=0.5, ls=(0, (2, 2)), zorder=0)
        fac = (V.get("factorial") or {}).get(t) or {}
        for m, vals in (fac.get("auc_pooled") or {}).items():
            if m.startswith("feat"):
                continue
            xs = [xpos[{"segment": "P-seg", "event": "P-event", "subject": "X"}[p]] for p in vals]
            o = np.argsort(xs)
            ax.plot(np.array(xs)[o], np.array(list(vals.values()))[o], color=NET, lw=0.6, marker="o", ms=1.8, zorder=2)
        for m, mk in (("feat_logistic", "o"), ("feat_gbm", "s")):
            r = P[t].get(m)
            if not r:
                continue
            for grp in (("W-rand", "W-event"), ("P-seg", "P-event", "X")):
                ks = [k for k in grp if k in r["protocols"]]
                if ks:
                    ax.plot([xpos[k] for k in ks], [r["protocols"][k]["pooled_auc"] for k in ks], color=AURA, lw=1.0,
                            ls=(0, (2.5, 1.4)) if m == "feat_logistic" else "-", marker=mk, ms=2.8,
                            mfc="white" if m == "feat_logistic" else AURA, mec=AURA, mew=0.7, zorder=3)
        ax.axvline(1.65, color=sp.HAIR, lw=0.6, zorder=0)
        ax.set_xticks([xpos[k] for k in xpos], [labels[k] for k in xpos])
        ax.tick_params(axis="x", length=0, labelsize=5.4)
        ax.set_ylim(0.35, 1.0); ax.set_xlim(-0.4, 4.7)
        ax.set_title(TASK_LABEL[t], fontsize=6)
        if t != "scalp":
            ax.tick_params(labelleft=False)
    axs[0].set_ylabel("pooled AUC")
    ent = [(sp.h_line(NET, lw=0.6, marker="o", ms=1.8), "network (one line each)"),
           (oline(AURA, ls=(0, (2.5, 1.4)), lw=1.0, ms=2.8, filled=False), "AURA + logistic"),
           (sp.h_line(AURA, lw=1.0, marker="s", ms=2.8), "AURA + GBM"),
           (sp.h_line(sp.FAINT, lw=0.5, ls=(0, (2, 2))), "chance")]
    fig.legend([e[0] for e in ent], [e[1] for e in ent], loc="outside upper center", ncol=4, borderaxespad=0.2)
    # (e) share of AUC variance explained by protocol, model and residual
    ax = axs[4]
    comp = [("vprot", "protocol", sp.AXIS), ("vmodel", "model", sp.GREY), ("vres", "residual", sp.FAINT)]
    for i, t in enumerate(TASKS):
        left = 0.0
        for k, _lab, c in comp:
            v = float(N.get(f"fac.{t}.{k}", "nan"))
            ax.barh(i, v, left=left, color=c, height=0.62, lw=0)
            if v >= 9:
                ax.text(left + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=5,
                        color="white" if c == sp.AXIS else sp.INK)
            left += v
    ax.set_yticks(range(4), [TASK_LABEL[t].replace(" seizure", "") for t in TASKS]); ax.invert_yaxis()
    ax.set_xlim(0, 100); ax.set_xlabel("variance in AUC (%)"); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    sp.key(ax, [(sp.h_patch(c), lab) for _k, lab, c in comp], ncol=3)
    # (f) AUC inflation of the pooled protocols over patient-independent folds (mean and range over models)
    ax = r2[1]
    for i, t in enumerate(TASKS):
        for j, (prot, mk, lab) in enumerate((("segment", "s", "segment-wise"), ("event", "o", "event-wise"))):
            m_ = N.get(f"fac.{t}.infl.{prot}.mean")
            if m_ is None:
                continue
            f = lambda k: float(N[f"fac.{t}.infl.{prot}.{k}"].replace("$-$", "-"))  # noqa: E731
            y_ = i + (-0.14 if j == 0 else 0.14)
            ax.plot([f("min"), f("max")], [y_, y_], color=sp.AXIS, lw=0.8)
            ax.plot([f("mean")], [y_], mk, ms=3.2, color=sp.AXIS, mfc="white" if j else sp.AXIS, mew=0.7)
    ax.axvline(0, color=sp.FAINT, lw=0.5, ls=(0, (2, 2)))
    ax.set_yticks(range(4), [TASK_LABEL[t].replace(" seizure", "") for t in TASKS]); ax.invert_yaxis()
    ax.set_xlabel("pooled AUC − patient-independent AUC"); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    sp.key(ax, [(oline(sp.AXIS, marker="s", lw=0.8), "segment-wise folds"), (oline(sp.AXIS, lw=0.8, filled=False), "event-wise folds")], ncol=2)
    _letter_finish(fig, axs + [r2[1]], "nn_fig2_protocol")


# ====================================================================================== figure 3
def fig3():
    """What the representations encode: label vs patient variance, and identity decoding."""
    sp.setup("nature")
    R, _ = _res("identity_decomp_*")
    run = {t: (R[t].get("matched") or R[t]["first_run"]) for t in TASKS}
    L = np.array([run[t]["per_feature_label"] for t in TASKS]).T
    Pt = np.array([run[t]["per_feature_patient"] for t in TASKS]).T
    fig = sp.figure("double", 92)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.82, 1.25])
    a, b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    g2 = gs[2].subgridspec(2, 1, height_ratios=[1, 1])
    c, d = fig.add_subplot(g2[0]), fig.add_subplot(g2[1])
    cols = [TASK_SHORT[t] for t in TASKS]
    sp.heatmap(a, L, FEATS, cols, cmap=sp.seq_cmap("#3B4A57"), vmin=0, vmax=0.35, annotate=True, cbar_label="share of variance")
    sp.heatmap(b, Pt, FEATS, cols, cmap=sp.seq_cmap("#3B4A57"), vmin=0, vmax=1.0, annotate=True, cbar_label="share of variance")
    b.tick_params(labelleft=False)
    for ax, ttl in ((a, "clinical label, within patients"), (b, "patient identity")):
        ax.xaxis.tick_top(); ax.tick_params(axis="x", labelsize=5.2)
        for y_ in (5.5, 11.5):
            ax.axhline(y_, color="white", lw=1.4)
        ax.set_title(ttl, fontsize=6, pad=16)
    # (c) identity and label decoding from each representation (held-out patients' clips)
    T = json.loads((REPO / "results" / "stream_tta" / "results.json").read_text())
    agg = {}
    for key, e in T.get("E7", {}).items():
        if e:
            agg.setdefault(key.split("|")[0], []).append(e["net"]); agg.setdefault("aura", []).append(e["aura"])
    rows = [m for m in ("EEGNet", "SCCNet", "deep_cnn_raw", "BIOT") if m in agg] + ["aura"]
    yy = np.arange(len(rows))
    chance = np.mean([x["chance"] for x in agg["aura"]])
    c.axvline(chance, color=sp.FAINT, lw=0.5, ls=(0, (2, 2))); c.axvline(0.5, color=sp.FAINT, lw=0.5, ls=(0, (1, 1.5)))
    for y_, m in zip(yy, rows):
        ida = np.mean([x["id_acc"] for x in agg[m]]); lab = np.mean([x["label_auc_cv"] for x in agg[m]])
        col = entity_color(m)
        c.plot([lab, ida], [y_, y_], color=sp.tint(col, 0.5), lw=1.2, zorder=1)
        c.plot([ida], [y_], "o", ms=4.2, color=col, mec="white", mew=0.5, zorder=3)
        c.plot([lab], [y_], "o", ms=4.2, mfc="white", mec=col, mew=0.9, zorder=3)
    c.set_yticks(yy, [NET_LABEL.get(m, m) for m in rows]); c.invert_yaxis()
    c.set_xlim(0, 1.0); c.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); c.set_xlabel("decoded from the representation")
    c.tick_params(axis="y", length=0); c.spines["left"].set_visible(False)
    sp.key(c, [(sp.h_marker(sp.INK, "o", ms=3.6), "patient identity (accuracy)"),
               (sp.h_marker(sp.INK, "o", ms=3.6, mfc="white", mec=sp.INK), "seizure label (AUC)"),
               (sp.h_line(sp.FAINT, lw=0.5, ls=(0, (2, 2))), f"identity chance ({chance:.2f})"),
               (sp.h_line(sp.FAINT, lw=0.5, ls=(0, (1, 1.5))), "label chance (0.5)")], ncol=1)
    # (d) share of the representation's variance explained by patient and by label (first 20 PCs, median)
    for y_, m in zip(yy, rows):
        ps_ = np.mean([x["patient_share"] for x in agg[m]]); ls_ = np.mean([x["label_share"] for x in agg[m]])
        col = entity_color(m)
        d.barh(y_ - 0.17, ps_, height=0.32, color=col, lw=0)
        d.barh(y_ + 0.17, ls_, height=0.32, color=sp.tint(col, 0.6), lw=0)
    d.set_yticks(yy, [NET_LABEL.get(m, m) for m in rows]); d.invert_yaxis()
    d.set_xlim(0, 0.2); d.set_xlabel("share of variance per component (median)"); d.tick_params(axis="y", length=0); d.spines["left"].set_visible(False)
    sp.key(d, [(sp.h_patch(sp.AXIS), "patient"), (sp.h_patch(sp.tint(sp.AXIS, 0.6)), "seizure label (within patient)")], ncol=1)
    _letter_finish(fig, [a, b, c, d], "nn_fig3_encode")


# ====================================================================================== figure 4
def fig4():
    """Test-time adaptation in continuous EEG."""
    sp.setup("nature")
    R = json.loads((REPO / "results" / "stream_tta" / "results.json").read_text())
    rows = [("EEGNet", "EEGNet"), ("SCCNet", "SCCNet"), ("deep_cnn_raw", "Repository CNN"), ("BIOT", "BIOT"),
            ("auranet_hybrid", "AURA-Net"), ("feat_gbm", "AURA + GBM")]
    cols = [("adabn_stream", "AdaBN,\nrecording"), ("adabn_refreshed", "AdaBN,\nrolling"), ("tent", "TENT"),
            ("ea_refreshed", "input,\nrolling"), ("repr", "represent.,\nrolling")]
    M = np.full((len(rows), len(cols)), np.nan); S = np.zeros_like(M, bool)
    for i, (m, _) in enumerate(rows):
        for j, (arm, _) in enumerate(cols):
            a2 = ("refreshed" if m in ("feat_gbm", "auranet_hybrid") else "probe_gbm_refreshed") if arm == "repr" else arm
            d = R["models"].get(m, {}).get(a2, {}).get("delta_vs_none")
            if d:
                M[i, j] = d["mean"]; S[i, j] = (d["p_signflip"] or 1) < 0.05
    fig = sp.figure("double", 70)
    (a, b), = sp.grid(fig, [[1.2, 1]])
    im, cb = sp.heatmap(a, M, [r[1] for r in rows], [c_[1] for c_ in cols], center=0.0, cmap=sp.div_cmap(sp.ABNORMAL, ADAPT),
                        cbar_label="mean change in per-recording AUC")
    for i in range(len(rows)):
        for j in range(len(cols)):
            if np.isfinite(M[i, j]):
                v = 0.0 if abs(M[i, j]) < 0.005 else M[i, j]
                a.text(j, i, f"{v:+.2f}".replace("-", "−") if v else "0.00", ha="center", va="center", fontsize=5,
                       color=sp.INK, fontweight="bold" if S[i, j] else "normal")
                if S[i, j]:
                    a.add_patch(Rectangle((j - 0.47, i - 0.45), 0.94, 0.9, fill=False, ec=sp.INK, lw=0.6))
            else:
                a.text(j, i, "not applicable", ha="center", va="center", fontsize=4.6, color=sp.SECOND)
    a.tick_params(axis="x", labelsize=5.2)
    sp.key(a, [(sp.h_patch("white", sp.INK), "sign-flip $P$ < 0.05 (bold)")], ncol=1)
    # (b) refreshed representation across cohorts: AURA features and BIOT's embedding
    SR, _ = _res("stream_replication_*")
    TT, _ = _res("stream_test_2*")
    SZ, _ = _res("seizeit2_external_*")
    SE, _ = _res("seizeit2_embedding_*")
    groups = []
    per = SR["models"]["feat_gbm"]["per_stream_auc"]; orig = {s for s, v in SR["streams"].items() if v.get("original")}
    groups.append(("feat", 0, [v["refreshed"] - v["none"] for s, v in per.items() if s not in orig],
                   SR["models"]["feat_gbm"]["replication"]["delta_refreshed"]["p_signflip"]))
    pb = R["models"]["BIOT"]["probe_gbm_refreshed"]; pn = R["models"]["BIOT"]["probe_gbm_none"]
    groups.append(("biot", 0, [pb["per_stream_auc"][s] - pn["per_stream_auc"][s] for s in pb["per_stream_auc"]],
                   pb.get("delta_vs_none", {}).get("p_signflip")))
    for coh, D, fa, fn, ba, bn in ((1, TT, "feat_gbm|refreshed", "feat_gbm|none", "BIOT|probe_gbm_refreshed", "BIOT|probe_gbm_none"),):
        A, B = D["models"][fa]["per_stream_auc"], D["models"][fn]["per_stream_auc"]
        groups.append(("feat", coh, [A[s] - B[s] for s in A], D["primary"]["p_signflip"]))
        A, B = D["models"][ba]["per_stream_auc"], D["models"][bn]["per_stream_auc"]
        groups.append(("biot", coh, [A[s] - B[s] for s in A], D["models"][ba].get("delta_vs_none", {}).get("p_signflip")))
    A, B = SZ["models"]["feat_gbm|refreshed"]["per_stream_auc"], SZ["models"]["feat_gbm|none"]["per_stream_auc"]
    groups.append(("feat", 2, [A[s] - B[s] for s in A], SZ["primary"]["p_signflip"]))
    e = SE["models"]["BIOT"]
    groups.append(("biot", 2, [e["refreshed"]["per_stream_auc"][s] - e["none"]["per_stream_auc"][s] for s in e["refreshed"]["per_stream_auc"]],
                   e["refreshed"].get("delta_vs_none", {}).get("p_signflip")))
    rng = np.random.default_rng(2)
    b.axhline(0, color=sp.FAINT, lw=0.5, ls=(0, (2, 2)), zorder=0)
    for kind, coh, d, p in groups:
        d = np.asarray(d); off = -0.18 if kind == "feat" else 0.18; x = coh + off
        col = AURA if kind == "feat" else FM
        b.scatter(x + rng.uniform(-0.07, 0.07, len(d)), d, s=3.5, color=sp.tint(col, 0.35), lw=0, zorder=2)
        m, lo, hi = _ci(d)
        b.plot([x + 0.11, x + 0.11], [lo, hi], color=col, lw=0.9, zorder=3)
        b.plot([x + 0.11], [m], "o", ms=3.2, color=col, mec="white", mew=0.4, zorder=4)
        if p is not None:
            b.text(x, -0.62, sp.fmt_p(p).replace("$P$ = ", ""), ha="center", fontsize=4.8, color=sp.SECOND)
    b.set_xticks([0, 1, 2], ["development\n(replication)", "held-out\npatients", "SeizeIT2\n(external)"])
    b.tick_params(axis="x", length=0); b.set_xlim(-0.55, 2.55); b.set_ylim(-0.68, 0.82)
    b.text(-0.53, -0.56, "$P$:", fontsize=4.8, color=sp.SECOND)
    b.set_ylabel("ΔAUC per recording (rolling − none)")
    sp.key(b, [(sp.h_marker(AURA, "o", ms=3.0), "AURA features"), (sp.h_marker(FM, "o", ms=3.0), "BIOT embedding (post hoc)"),
               (sp.h_marker(sp.GREY, "o", ms=2.0), "one recording"), (sp.h_line(sp.INK, lw=0.9, marker="o", ms=2.8), "mean, 95% bootstrap CI")], ncol=2)
    _letter_finish(fig, [a, b], "nn_fig4_tta")


# ====================================================================================== figure 5
def fig5():
    """Patient baseline in clips and continuous EEG, with one example recording."""
    from bsf.eval import clinical as C
    from make_clinical_figures import _align_pair, _streams
    sp.setup("nature")
    per, orig, Y, G, Tt, P, n_ev = _streams()
    fig = sp.figure("double", 100)
    rows = sp.grid(fig, [[1, 1, 1.1], [1]], height_ratios=[1.05, 0.85])
    a, b, c = rows[0]; d = rows[1][0]
    pr, _lab = _align_pair("detect")
    sp.paired_slopes(a, pr[:, 0], pr[:, 1], labels=("no reference", "patient\nreference"), color=ADAPT, unit_name="patient")
    a.set_ylabel("per-patient AUC"); a.set_ylim(0, 1.02)
    rng = np.random.default_rng(3)
    b.axhline(0, color=sp.FAINT, lw=0.5, ls=(0, (2, 2)))
    for x, arm in ((0, "anchored"), (1, "rolling")):
        dd = np.array([v[arm] - v["none"] for v in per.values()]); isorig = np.array([s in orig for s in per])
        jit = rng.uniform(-0.12, 0.12, len(dd))
        b.scatter(x + jit[isorig], dd[isorig], s=6, color=ADAPT, lw=0, zorder=3)
        b.scatter(x + jit[~isorig], dd[~isorig], s=6, facecolor="white", edgecolor=ADAPT, lw=0.6, zorder=3)
        m, lo, hi = _ci(dd)
        b.plot([x + 0.26] * 2, [lo, hi], color=sp.INK, lw=0.8); b.plot([x + 0.26], [m], "o", ms=3, color=sp.INK)
    b.set_xticks([0, 1], ["fixed\n(first 30 min)", "rolling\n(35–5 min before)"]); b.tick_params(axis="x", length=0)
    b.set_xlim(-0.45, 1.5); b.set_ylabel("ΔAUC per recording (vs none)")
    sp.key(b, [(oline(ADAPT, lw=0, ms=2.6), "devising set (6)"), (oline(ADAPT, lw=0, ms=2.6, filled=False), "replication"),
               (sp.h_line(sp.INK, lw=0.8, marker="o", ms=2.6), "mean, 95% CI")], ncol=2)
    budgets = np.linspace(0.005, 0.5, 60)
    for arm, col, ls, lab in (("none", AURA, "-", "no reference"), ("anchored", ADAPT, (0, (2.5, 1.4)), "fixed reference"),
                              ("rolling", ADAPT, "-", "rolling reference")):
        se = [C.sensitivity_at_budget(Y, P[f"feat_gbm|{arm}"], bb)["event_sensitivity"] for bb in budgets]
        c.plot(budgets, se, color=col, lw=1.0, ls=ls)
    c.plot([0, 0.5], [0, 0.5], color=sp.FAINT, lw=0.5, ls=(0, (2, 2)))
    c.set_xlim(0, 0.5); c.set_ylim(0, 1.02); c.set_xlabel("time in warning (fraction)"); c.set_ylabel("seizures detected (fraction)")
    sp.key(c, [(sp.h_line(AURA, lw=1.0), "no reference"), (sp.h_line(ADAPT, lw=1.0, ls=(0, (2.5, 1.4))), "fixed"),
               (sp.h_line(ADAPT, lw=1.0), "rolling"), (sp.h_line(sp.FAINT, lw=0.5, ls=(0, (2, 2))), "chance")], ncol=2)
    # (d) one recording
    gains = {s: v["rolling"] - v["none"] for s, v in per.items() if v["rolling"] >= 0.9 and s in orig}
    s0 = max(gains, key=gains.get) if gains else list(per)[0]
    m = G == s0
    h = Tt[m] / 3600.0; yy = Y[m]
    brk = np.r_[0, np.where(np.diff(h) > 0.05)[0] + 1, len(h)]
    on = np.where(np.diff(np.r_[0, yy, 0]) != 0)[0].reshape(-1, 2)
    for a_, b_ in on:
        d.axvspan(h[a_], h[min(b_, len(h) - 1)] + 8 / 3600, color=SEIZ, lw=0, zorder=0)
    for arm, col, off in (("none", AURA, 1.18), ("rolling", ADAPT, 0.0)):
        p = P[f"feat_gbm|{arm}"][m]; th = float(np.quantile(P[f"feat_gbm|{arm}"], 0.90))
        for a_, b_ in zip(brk[:-1], brk[1:]):
            seg = slice(a_, b_)
            ps = np.convolve(p[seg], np.ones(15) / 15, mode="same") if b_ - a_ > 15 else p[seg]
            d.plot(h[seg], p[seg] + off, color=sp.tint(col, 0.6), lw=0.3, zorder=1)
            d.plot(h[seg], ps + off, color=col, lw=0.8, zorder=2)
        d.plot([h[0], h[-1]], [th + off] * 2, color=col, lw=0.5, ls=(0, (3, 2)))
    d.set_yticks([0, 0.5, 1, 1.18, 1.68, 2.18], ["0", "0.5", "1", "0", "0.5", "1"])
    d.set_ylim(-0.05, 2.25); d.set_xlim(h[0], h[-1])
    d.set_xlabel("time from the start of the recording (h); gaps between files left blank")
    d.set_ylabel("seizure probability")
    gaps = np.diff(h); gi = int(np.argmax(gaps)); xg = (h[gi] + h[gi + 1]) / 2 if gaps[gi] > 0.5 else h[0] + 0.5
    d.text(xg, 1.18 + 0.5, "no reference", ha="center", va="center", fontsize=5.5, color=AURA)
    d.text(xg, 0.5, "rolling reference", ha="center", va="center", fontsize=5.5, color=ADAPT)
    sp.key(d, [(sp.h_line(sp.INK, lw=0.3), "per window"), (sp.h_line(sp.INK, lw=0.8), "2-min average"),
               (sp.h_line(sp.INK, lw=0.5, ls=(0, (3, 2))), "alarm threshold (10% time in warning)"),
               (sp.h_patch(SEIZ), "annotated seizure")], ncol=4)
    _letter_finish(fig, [a, b, c, d], "nn_fig5_baseline")


# ====================================================================================== figure 6
def _inkey(ax, entries, loc, **kw):
    """Compact key inside an empty corner of a panel (ROC below the diagonal, calibration above it)."""
    hs, ls = zip(*entries)
    return ax.legend(hs, ls, loc=loc, frameon=False, fontsize=5.2, handlelength=1.4, borderaxespad=0.3,
                     labelspacing=0.25, **kw)


def fig6():
    """Held-out test patients: ROC and confusion matrices at development thresholds."""
    from sklearn.metrics import roc_curve

    from make_clinical_figures import REF, _addendum, _best_fm
    sp.setup("nature")
    R, _ = _res("clinical_metrics_*")
    fig = sp.figure("double", 150)
    gs = fig.add_gridspec(3, 1)
    r1, r2, r3 = ([fig.add_subplot(c_) for c_ in gs[i].subgridspec(1, 4)] for i in range(3))
    neg_pos = {"typing": ("focal", "gen."), "detect": ("non-ictal", "ictal")}
    seen = {}
    curves = [("ref", "AURA", AURA, "-"), ("feat_logistic_base", "AURA, patient reference", ADAPT, "-"),
              ("BIOT_pt", "BIOT (pretrained)", FM, (0, (3, 1.5))), ("CBraMod_pt", "CBraMod (pretrained)", sp.tint(FM, 0.35), "-")]
    for j, t in enumerate(TASKS):
        y, g, P = _addendum(t); T = R["test"][t]
        ref = REF[t] if REF[t] in P else REF[t] + "_sub"
        ax = r1[j]
        ax.plot([0, 1], [0, 1], color=sp.FAINT, lw=0.5, ls=(0, (2, 2)))
        aucs = []
        for arm, lab, col, ls in curves:
            a2 = ref if arm == "ref" else arm
            if a2 not in P or a2 not in T:
                continue
            fpr, tpr, _ = roc_curve(y, P[a2])
            ax.plot(fpr, tpr, color=col, lw=1.0, ls=ls)
            ax.plot(1 - T[a2]["spec"], T[a2]["sens"], "o", ms=3.0, color=col, mec="white", mew=0.4, zorder=5)
            aucs.append((col, T[a2]["auc"]))
            seen[lab] = (sp.h_line(col, lw=1.0, ls=ls), lab)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xlabel("1 − specificity")
        if j == 0:
            ax.set_ylabel("sensitivity")
        ax.set_title(TASK_LABEL[t], fontsize=5.8, loc="left", pad=2)
        for k, (col, v) in enumerate(aucs):              # AUC values in the empty lower-right corner
            ax.text(0.97, 0.04 + 0.075 * (len(aucs) - 1 - k), f"{v:.2f}", color=col, fontsize=5.4, ha="right",
                    va="bottom", transform=ax.transAxes, fontweight="bold")
        ax.text(0.97, 0.04 + 0.075 * len(aucs), "AUC", color=sp.SECOND, fontsize=5.2, ha="right", va="bottom",
                transform=ax.transAxes)
        neg, pos = neg_pos.get(t, ("interictal", "preictal"))
        fm = _best_fm(t, R)
        for axc, (arm, lab, col) in ((r2[j], ("feat_logistic_base", "AURA, patient reference", ADAPT)),
                                     (r3[j], (fm, "BIOT (pretrained)" if fm.startswith("BIOT") else "CBraMod (pretrained)", FM))):
            v = T.get(arm)
            if not v:
                axc.axis("off"); continue
            sp.confusion(axc, np.array([[v["tn"], v["fp"]], [v["fn"], v["tp"]]]), [neg, pos], color=col, key_loc=None)
            axc.set_title(f"{lab}\nsens. {v['sens']:.2f}, spec. {v['spec']:.2f}, F1 {v['f1']:.2f}", fontsize=5.4)
            if j:
                axc.set_ylabel("")
    fig.legend([h for h, _ in seen.values()], list(seen), loc="outside upper center", ncol=len(seen), frameon=False,
               fontsize=5.6, handlelength=1.6, columnspacing=1.6)
    rows = [r1, [a_ for a_ in r2 if a_.axison], [a_ for a_ in r3 if a_.axison]]
    _balance_rows(fig, gs, rows)
    _letter_finish(fig, r1 + r2 + r3, "nn_fig6_test", balance=(gs, rows))


# ====================================================================================== figure 7
def fig_montage():
    """Networks against the matched physiological reference in the common montage (3 seeds), and the
    change from file-order channels to the double banana."""
    sp.setup("nature")
    C, _ = _res("montage_contrast_*")
    fig = sp.figure("double", 60)
    a, b, c = sp.grid(fig, [[1, 1, 0.58]])[0]
    for ax, task in ((a, "detect"), (b, "scalp")):
        r = C["tasks"][task]; nets = r["networks"]
        order = sorted(nets, key=lambda m: nets[m]["delta"])
        yy = np.arange(len(order))
        ax.axvline(0, color=sp.FAINT, lw=0.5, ls=(0, (2, 2)))
        for y_, m in zip(yy, order):
            v = nets[m]; col = entity_color(m)
            ax.plot([v["lo"], v["hi"]], [y_, y_], color=col, lw=0.9, solid_capstyle="butt")
            ax.plot([v["delta"]], [y_], "o", ms=3.4, color=col, mfc=col if v["significantly_better"] else "white", mec=col, mew=0.8)
            ax.text(1.03, y_, sp.fmt_ci(v["delta"], v["lo"], v["hi"], 3),
                    transform=matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData),
                    va="center", fontsize=5, color=sp.INK)
        ns = max(v.get("n_seeds") or 1 for v in nets.values())
        ax.set_yticks(yy, [NET_LABEL.get(m, m) for m in order]); ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False); ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_xlabel("network − AURA, same clips (AUC)"); ax.set_xlim(-0.21, 0.15)
        sp.key(ax, [(oline(sp.GREY), "better (both tests, BH)"), (oline(sp.GREY, filled=False), "not better")], ncol=2,
               title=f"{TASK_LABEL[task]}: AURA {r['reference_auc']:.3f}, {ns} seed{'s' if ns > 1 else ''}",
               title_fontsize=5.4, alignment="left")
    nets = [m for m in ("ContraWR", "BIOT", "ATCNet", "EEGNet", "SCCNet", "deep_cnn_raw") if f"mon.detect.{m}" in N]
    ends = {}
    for m in nets + ["feat_gbm"]:
        new_ = float(N[f"mon.detect.{m}"]); old = new_ - float(N[f"mon.detect.{m}.delta"].replace("$-$", "-"))
        col = entity_color(m)
        c.plot([0, 1], [old, new_], color=sp.tint(col, 0.3), lw=0.8 if m != "feat_gbm" else 1.2, marker="o", ms=2.6,
               mfc=col, mec="white", mew=0.3, zorder=3 if m in ("BIOT", "feat_gbm") else 2)
        ends[m] = new_
    net_end = np.mean([ends[m] for m in nets if m != "BIOT"])
    for y_, lab, col in ((net_end, "networks", NET), (ends.get("BIOT"), "BIOT", FM), (ends["feat_gbm"], "AURA + GBM", AURA)):
        if y_ is not None:
            c.text(1.12, y_, lab, color=col if col != NET else sp.SECOND, fontsize=5.2, va="center", ha="left")
    c.set_xticks([0, 1], ["file\norder", "double\nbanana"]); c.tick_params(axis="x", length=0); c.set_xlim(-0.2, 1.08)
    c.set_ylabel("detection AUC (development, seed 0)")
    _letter_finish(fig, [a, b, c], "nn_fig_montage")


def fig_ens():
    """AURA-Ens on the 25 development recordings in the common montage."""
    sp.setup("nature")
    E = json.loads((REPO / "results" / "aura_ens_db18" / "results.json").read_text())
    Mo = E["models"]
    fig = sp.figure("double", 64)
    a, d, e = sp.grid(fig, [[0.62, 1, 1.3]])[0]
    # (a) per-recording AUC: rolling-reference AURA + GBM against AURA-Ens
    A = Mo["feat_gbm"]["refreshed"]["per_stream_auc"]; B = Mo["aura_ens"]["post"]["per_stream_auc"]
    ks = sorted(A)
    xa, xb = np.array([A[k] for k in ks]), np.array([B[k] for k in ks])
    for u, v in zip(xa, xb):
        a.plot([0, 1], [u, v], color=ENS if v > u else sp.GREY, lw=0.5, alpha=0.75, zorder=2)
    a.plot(np.zeros_like(xa), xa, "o", ms=2.4, color=AURA, mec="white", mew=0.3, zorder=3)
    a.plot(np.ones_like(xb), xb, "o", ms=2.4, color=ENS, mec="white", mew=0.3, zorder=3)
    a.set_xticks([0, 1], ["AURA + GBM,\nrolling ref.", "AURA-Ens"]); a.tick_params(axis="x", length=0); a.set_xlim(-0.35, 1.35)
    a.set_ylabel("AUC per recording"); a.set_ylim(min(xa.min(), xb.min()) - 0.03, 1.0)
    dv = E["models"]["aura_ens"]["post"]["delta_vs_aura_refreshed"]
    sp.key(a, [(sp.h_line(ENS, lw=0.8), "higher with AURA-Ens"), (sp.h_line(sp.GREY, lw=0.8), "lower")], ncol=1,
           title=f"{dv['n_better']} of {len(ks)} higher; {sp.fmt_p(dv['p_signflip'])}", title_fontsize=5.4, alignment="left")
    # (b) operating points: seizures detected against false alarms per 24 h at the three budgets
    best_net = max((m for m in Mo if m not in ("feat_gbm", "feat_logistic", "auranet_hybrid", "ens_mean", "aura_ens")
                    and "probe_gbm_refreshed" in Mo[m]), key=lambda m: Mo[m]["probe_gbm_refreshed"]["auc"])
    series = [("feat_gbm", "refreshed", AURA, "AURA + GBM, rolling ref."),
              (best_net, "probe_gbm_refreshed", entity_color(best_net), f"{NET_LABEL.get(best_net, best_net)} embedding, rolling ref."),
              ("aura_ens", "post", ENS, "AURA-Ens"), ("aura_ens", "pbn", sp.tint(ENS, 0.45), "AURA-Ens with PBN members")]
    series = [s_ for s_ in series if s_[1] in Mo.get(s_[0], {})]
    ymax = 0
    for m, arm, col, lab in series:
        sz = Mo[m][arm]["szcore"]; bs = sorted(sz, key=float)
        ys = [sz[b_]["sensitivity"] for b_ in bs]; ymax = max(ymax, max(ys))
        d.plot([sz[b_]["fa_per_24h"] for b_ in bs], ys, color=col, lw=1.0, marker="o", ms=3, mec="white", mew=0.4)
    d.set_xscale("symlog", linthresh=1.0, linscale=0.6)      # a budget can be met with zero false alarms
    d.set_xticks([0, 1, 2, 6, 24], ["0", "1", "2", "6", "24"]); d.set_xlim(-0.1, 34)
    d.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    d.set_ylim(0, np.ceil(ymax * 10 + 0.5) / 10)
    d.set_xlabel("false alarms per 24 h"); d.set_ylabel("share of seizures detected")
    sp.key(d, [(sp.h_line(col, lw=1.0, marker="o", ms=3), lab) for m, arm, col, lab in series], ncol=2)
    # (c) per network: no adaptation, post-hoc embedding normalization, PBN (median per-recording AUC)
    nets = [m for m in Mo if all(k in Mo[m] for k in ("none", "probe_gbm_refreshed", "pbn"))]
    xs = np.arange(len(nets)); allv = []
    for i, m in enumerate(nets):
        vals = [Mo[m][k]["median_stream_auc"] for k in ("none", "probe_gbm_refreshed", "pbn")]; allv += vals
        e.plot([i - 0.22, i, i + 0.22], vals, color=sp.FAINT, lw=0.6, zorder=1)
        for dx, v, col, mk in zip((-0.22, 0, 0.22), vals, (sp.GREY, sp.tint(ADAPT, 0.4), ADAPT), ("o", "o", "s")):
            e.plot([i + dx], [v], mk, ms=3.6, color=col, mec="white", mew=0.3, zorder=3)
    labs = []
    for m in nets:
        dp = Mo[m]["pbn"].get("delta_vs_posthoc")
        labs.append(f"{NET_LABEL.get(m, m)}\n{sp.fmt_p(dp['p_signflip'] or 1) if dp else ''}")
    e.set_xticks(xs, labs, fontsize=5.2); e.tick_params(axis="x", length=0); e.set_xlim(-0.5, len(nets) - 0.5)
    e.set_ylim(np.floor(min(allv) * 20) / 20 - 0.01, np.ceil(max(allv) * 20) / 20 + 0.01)
    e.set_ylabel("median AUC per recording")
    sp.key(e, [(sp.h_marker(sp.GREY, "o", ms=3), "no adaptation"), (sp.h_marker(sp.tint(ADAPT, 0.4), "o", ms=3), "embedding, post hoc"),
               (sp.h_marker(ADAPT, "s", ms=3), "PBN")], ncol=3, title="$P$: PBN against post hoc (sign-flip test)",
           title_fontsize=5.4, alignment="left")
    _letter_finish(fig, [a, d, e], "nn_fig_ens")


# ====================================================================================== figure 8
def fig8():
    """AURA Monitor: pipeline, one patient's timeline, seizure fingerprints, IED detection and risk calibration."""
    from sklearn.metrics import roc_curve
    sp.setup("nature")
    rep = json.loads((REPO / "reports" / "chbmit_chb01" / "report.json").read_text())
    PD, pdir = _res("seizure_patterns_*")
    F = json.loads((pdir / "fingerprints.json").read_text())
    IR, idir = _res("ied_monitor_*")
    RK, _ = _res("risk_monitor_*")
    fig = sp.figure("double", 170)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.52, 1.0, 0.9])
    # (a) pipeline, drawn in millimetres on its own axes
    a = fig.add_subplot(gs[0]); a.set_xlim(0, 183); a.set_ylim(0, 40); a.set_aspect("equal"); a.axis("off")
    sc.group(a, 0.5, 0.5, 182, 39, None)
    y0 = 12.0
    mbox(a, 3, y0, 22, 15, ["Scalp EEG", "any montage"], color=sp.GREY, style="outline")
    mbox(a, 29, y0, 24, 15, ["Common montage", "18 derivations,", "signal quality"], color=sp.GREY)
    mbox(a, 57, y0, 25, 15, ["Patient reference", "35 to 5 min", "before each window"], color=ADAPT)
    mods = [("AURA-Ens", "seizure probability", ENS), ("IED detection", "rate, type, map", IED),
            ("Fingerprints", "onset map, types", AURA), ("Next-hour risk", "calibrated", sp.GREY)]
    for i, (t1, t2, col) in enumerate(mods):
        mbox(a, 90, 31.0 - i * 8.6, 33, 7.0, [t1, t2], color=col, size=-0.4)
        sc.arrow(a, [(82.2, y0 + 7.5), (86.0, y0 + 7.5), (86.0, 34.5 - i * 8.6), (89.8, 34.5 - i * 8.6)], lw=0.5, head=1.0, width=0.8)
        sc.arrow(a, [(123.2, 34.5 - i * 8.6), (126.5, 34.5 - i * 8.6), (126.5, y0 + 7.5), (129.8, y0 + 7.5)], lw=0.5, head=1.0, width=0.8)
    mbox(a, 130, y0, 23, 15, ["Alarms", "evidence cards,", "report"], color=sp.GREY, style="outline")
    mbox(a, 157, y0, 23, 15, ["Clinician review", "confirm or reject"], color=ADAPT, style="tint", lw=1.0)
    for x0, x1 in ((25, 29), (53, 57), (153, 157)):
        sc.arrow(a, [(x0 + 0.2, y0 + 7.5), (x1 - 0.2, y0 + 7.5)], lw=0.6, head=1.2, width=1.0)
    sc.arrow(a, [(168.5, y0 - 0.2), (168.5, 4.0), (106.5, 4.0), (106.5, 6.4)], lw=0.6, head=1.1, width=0.9, dashed=True, color=ADAPT)
    sc.text(a, 137.5, 2.0, "confirmed seizures train a patient head; rejected alarms raise the threshold", size=-1.0, color=ADAPT, ha="center")
    # (b) timeline of one patient, (c) onset maps of four of its seizures
    gbc = gs[1].subgridspec(1, 2, width_ratios=[2.05, 1], wspace=0.04)
    gb = gbc[0].subgridspec(3, 1, height_ratios=[1, 0.62, 0.5], hspace=0.1)
    b1, b2, b3 = (fig.add_subplot(gb[i]) for i in range(3))
    T = rep["timeline"]; t = np.asarray(T["t"], float); p = np.asarray(T["prob"], float)
    sz = rep["annotated_seizures"]
    lo = max(t[0], sz[0][0] - 4 * 3600) if sz else t[0]; hi = min(t[-1], lo + 12 * 3600)
    hh = lambda v: (np.asarray(v, float) - lo) / 3600  # noqa: E731
    m = (t >= lo) & (t <= hi)
    for ax in (b1, b2, b3):
        for s0, s1 in sz:
            if lo <= s0 <= hi:
                ax.axvspan(hh(s0), hh(s1) + 0.04, color=SEIZ, lw=0, zorder=0)
        ax.set_xlim(0, hh(hi))
    b1.plot(hh(t[m]), p[m], color=ENS, lw=0.6)
    b1.axhline(rep["detection"]["threshold"], color=sp.INK, lw=0.5, ls=(0, (3, 2)))
    b1.set_ylim(0, 1.0); b1.set_yticks([0, 0.5, 1]); b1.set_ylabel("seizure\nprobability"); b1.tick_params(labelbottom=False)
    tb = np.asarray(T["ied_bins_t"], float); nb = np.asarray(T["ied_bins_n"], float); mb = (tb >= lo) & (tb <= hi)
    b2.bar(hh(tb[mb]) + 1 / 12, nb[mb], width=1 / 6 * 0.85, color=IED, lw=0)
    b2.set_ylabel("IEDs per\n10 min"); b2.tick_params(labelbottom=False)
    rt = np.asarray(T["risk_t"], float); rp = np.asarray(T["risk_p"], float); mr = (rt >= lo) & (rt <= hi)
    b3.plot(hh(rt[mr]), rp[mr], color=sp.AXIS, lw=0.9); b3.set_ylabel("risk,\nnext hour"); b3.set_xlabel("time (h)")
    b3.set_ylim(0, np.ceil(rp[mr].max() * 10) / 10)
    for ax in (b1, b2, b3):
        ax.yaxis.set_label_coords(-0.075, 0.5)          # one column of y labels
    sp.key(b1, [(sp.h_line(ENS, lw=0.6), "AURA-Ens"), (sp.h_line(sp.INK, lw=0.5, ls=(0, (3, 2))), "alarm threshold"),
                (sp.h_patch(IED), "IEDs"), (sp.h_line(sp.AXIS, lw=0.9), "risk"), (sp.h_patch(SEIZ), "annotated seizure")], ncol=5)
    c_ = fig.add_subplot(gbc[1])
    own = [f for f in F if f["patient"] == "chb01"][:4]
    vmax = max(max(f["onset_map"]) for f in own)
    cmap = sp.seq_cmap(sp.ABNORMAL, light=0.9)
    dx, dy = 2.55, 2.75                                # head pitch (radius 1), room for each title
    for k, f in enumerate(own):
        hx, hy = (k % 2) * dx, -(k // 2) * dy
        bipolar_head(c_, f["onset_map"], cmap, vmax, cx=hx, cy=hy, limits=False)
        c_.text(hx, hy + 1.24, f"seizure {f['seizure'] + 1}, type {chr(65 + f.get('type', 0))}", fontsize=5.4,
                ha="center", va="bottom", color=sp.INK)
    c_.set_xlim(-1.15, dx + 1.15); c_.set_ylim(-dy - 1.12, 1.62); c_.set_aspect("equal"); c_.axis("off")
    cax = c_.inset_axes([1.03, 0.08, 0.045, 0.84])
    sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, vmax), cmap=cmap)
    cb = fig.colorbar(sm, cax=cax)
    cb.outline.set_visible(False); cb.set_label("ictal evidence, first 16 s (z)"); cb.ax.tick_params(length=1.5, width=0.4)
    cmaps = [c_]
    # (d) fingerprint similarity, (e) IED detection ROC, (f) risk calibration: three square panels
    gd = gs[2].subgridspec(1, 3)
    d, e, f_ = (fig.add_subplot(gd[i]) for i in range(3))
    from bsf.monitor.patterns import vector
    V = np.stack([vector(x) for x in F]); Pp = np.array([x["patient"] for x in F]); S = V @ V.T
    iu = np.triu_indices(len(F), 1); same = Pp[iu[0]] == Pp[iu[1]]
    bins = np.linspace(-0.2, 1, 31)
    hd = d.hist(S[iu][~same], bins=bins, density=True, color=sp.tint(sp.GREY, 0.3), lw=0)[0]
    hs = d.hist(S[iu][same], bins=bins, density=True, histtype="step", color=AURA, lw=1.0)[0]
    ctr = (bins[:-1] + bins[1:]) / 2
    top = max(hd.max(), hs.max())
    d.text(ctr[np.argmax(hd)], hd.max() + 0.06 * top, "different\npatients", ha="center", va="bottom", fontsize=5.2, color=sp.SECOND)
    d.text(ctr[np.argmax(hs)] - 0.07, hs.max(), "same\npatient", ha="right", va="top", fontsize=5.2, color=AURA)
    d.text(-0.18, top * 1.28, f"{PD['n_seizures']} seizures, {PD['n_patients']} patients\n{sp.fmt_p(PD['stereotypy']['p_perm'])}",
           ha="left", va="top", fontsize=5.2, color=sp.INK)
    d.set_ylim(0, top * 1.3); d.set_xlim(-0.2, 1.0)
    d.set_xlabel("fingerprint similarity (cosine)"); d.set_ylabel("density"); d.set_box_aspect(1)
    pr = np.load(idir / "preds.npz", allow_pickle=True)
    yv = pr["y"]
    e.plot([0, 1], [0, 1], color=sp.FAINT, lw=0.5, ls=(0, (2, 2)))
    ent = []
    for k, lab, col, ls in (("gbm_raw", "descriptors", IED, "-"), ("gbm_patz", "patient-referenced", sp.tint(IED, 0.45), "-"),
                            ("cnn", "repository CNN", NET, (0, (3, 1.5)))):
        fpr, tpr, _ = roc_curve(yv, pr[k]); e.plot(fpr, tpr, color=col, lw=1.0, ls=ls)
        ent.append((sp.h_line(col, lw=1.0, ls=ls), f"{lab}, {IR['detection'][k]['auc']:.3f}"))
    e.set_xlim(0, 1); e.set_ylim(0, 1); e.set_box_aspect(1)
    e.set_xlabel("1 − specificity"); e.set_ylabel("sensitivity")
    _inkey(e, ent, "lower right", title=f"IEDs, {IR['n_patients']} patients (AUC)", title_fontsize=5.2, alignment="left")
    rel = RK["horizons"]["60"]["aura_risk"]["reliability"]
    pm = [r["p_mean"] for r in rel]; ob = [r["observed"] for r in rel]; nn_ = [r["n"] for r in rel]
    from sciplots import wilson
    ci = [wilson(round(o * n_), n_) for o, n_ in zip(ob, nn_)]
    mx = max(max(pm), max(h for _, h in ci)) * 1.08
    f_.plot([0, mx], [0, mx], color=sp.FAINT, lw=0.5, ls=(0, (2, 2)))
    f_.vlines(pm, [l_ for l_, _ in ci], [h_ for _, h_ in ci], color=sp.AXIS, lw=0.7)
    f_.plot(pm, ob, "-o", color=sp.AXIS, lw=0.9, ms=2.8, mec="white", mew=0.4)
    f_.set_xlim(0, mx); f_.set_ylim(0, mx); f_.set_box_aspect(1)
    f_.set_xlabel("predicted risk, next hour"); f_.set_ylabel("observed frequency")
    _inkey(f_, [(sp.h_line(sp.AXIS, lw=0.9, marker="o", ms=2.8), "risk model (quantile bins)"),
                (matplotlib.lines.Line2D([], [], color=sp.AXIS, lw=0, marker="|", ms=5), "95% CI (Wilson)"),
                (sp.h_line(sp.FAINT, lw=0.5, ls=(0, (2, 2))), "perfect calibration")], "upper left")
    rows = [[a], cmaps, [d, e, f_]]
    _balance_rows(fig, gs, rows)
    _letter_finish(fig, [a, b1, cmaps[0], d, e, f_], "nn_fig8_monitor", balance=(gs, rows))


# ====================================================================================== main
FIGS = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5, "fig6": fig6, "montage": fig_montage, "ens": fig_ens, "fig8": fig8}


def main() -> int:
    want = sys.argv[1:] or list(FIGS)
    for k in want:
        try:
            FIGS[k]()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"{k}: FAILED {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
