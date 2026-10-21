r"""Nature-style schematic diagrams (framework, pipeline, architecture figures), drawn in millimetres.

    import sys; sys.path.insert(0, r"C:\Users\user\.claude\skills\visual-design\scripts")
    import sciplots as sp, schematic as sc
    sp.setup("nature")
    fig, ax = sc.canvas(183, 60)             # coordinates are millimetres, origin bottom-left
    sc.group(ax, 2, 4, 40, 52, "Clinical EEG")
    sc.box(ax, 6, 30, 32, 9, "19 channels, 256 Hz", color=sp.BLUE)
    sc.arrow(ax, [(42, 30), (50, 30)])
    sp.save(fig, "fig1_framework")

Look: flat vector shapes, soft tinted fills with 0.5 pt keylines (or none), small corner radii,
6-7 pt sentence-case text, thin grey connectors with small filled heads, one accent colour for the
new component, glyphs (traces, heads, bars) instead of clip-art, math set as math ($x_{ij}$).
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, Rectangle

import sciplots as sp

MM = sp.MM

#: 10-20 positions on a unit head, nose up (from D:\NeuroIPS\eegbudget\viz.py).
POS = {"Fp1": (-.31, .95), "Fp2": (.31, .95),
       "F7": (-.81, .59), "F3": (-.42, .55), "Fz": (0, .5), "F4": (.42, .55), "F8": (.81, .59),
       "T3": (-.93, 0), "C3": (-.47, 0), "Cz": (0, 0), "C4": (.47, 0), "T4": (.93, 0),
       "T5": (-.81, -.59), "P3": (-.42, -.55), "Pz": (0, -.5), "P4": (.42, -.55), "T6": (.81, -.59),
       "O1": (-.31, -.95), "O2": (.31, -.95)}


def _fs(delta=0.0):
    return plt.rcParams["font.size"] + delta


def canvas(width_mm=183, height_mm=60):
    fig = plt.figure(figsize=(width_mm * MM, height_mm * MM))
    fig.set_layout_engine(None)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_mm); ax.set_ylim(0, height_mm); ax.set_aspect("equal"); ax.axis("off")
    return fig, ax


def text(ax, x, y, s, size=0.0, weight="normal", color=None, ha="left", va="center", **kw):
    return ax.text(x, y, s, fontsize=_fs(size), fontweight=weight, color=color or sp.INK, ha=ha, va=va, **kw)


def group(ax, x, y, w, h, title=None, fill=None, edge=None, r=2.0, title_size=0.5):
    """A region of the figure: very light fill, no border by default, title top-left in bold."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fill or sp.PANEL, ec=edge or "none", lw=0.5 if edge else 0, zorder=0))
    if title:
        text(ax, x + 2.2, y + h - 2.6, title, size=title_size, weight="bold", va="top")


def box(ax, x, y, w, h, title=None, sub=None, color=None, style="tint", r=1.0, lw=0.5, size=0.0, zorder=2):
    """A component. style: 'tint' (pale fill + keyline), 'outline' (white + keyline),
    'solid' (filled, white text; reserve for the one component the figure is about)."""
    c = color or sp.BLUE
    fc = {"tint": sp.tint(c, 0.86), "outline": "white", "solid": c}[style]
    ec = "none" if style == "solid" else c
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=zorder))
    tc = "white" if style == "solid" else sp.INK
    cx, cy = x + w / 2, y + h / 2
    if title and sub:
        text(ax, cx, cy + 1.3, title, size=size, weight="bold", color=tc, ha="center", zorder=zorder + 1)
        text(ax, cx, cy - 1.6, sub, size=size - 0.8, color=tc if style == "solid" else sp.SECOND, ha="center",
             zorder=zorder + 1)
    elif title:
        text(ax, cx, cy, title, size=size, color=tc, ha="center", zorder=zorder + 1)


def chip(ax, x, y, s, color=None, size=-0.6, h=3.6, pad=1.4):
    """A small rounded tag (task, condition, arm). Width follows the text."""
    c = color or sp.BLUE
    t = text(ax, x + pad, y + h / 2, s, size=size, zorder=4)
    ax.figure.canvas.draw()
    bb = t.get_window_extent().transformed(ax.transData.inverted())
    w = bb.width + 2 * pad
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={h / 2}",
                                fc=sp.tint(c, 0.84), ec="none", zorder=3))
    return x + w


def arrow(ax, pts, color=None, lw=0.6, head=1.3, width=1.0, dashed=False, zorder=1):
    """Polyline connector with a small filled head at the end. pts in mm."""
    c = color or sp.AXIS; pts = np.asarray(pts, float)
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    d = np.array([x1 - x0, y1 - y0]); d = d / (np.linalg.norm(d) + 1e-9); n = np.array([-d[1], d[0]])
    tip = np.array([x1, y1]); base = tip - d * head
    line = np.vstack([pts[:-1], base])
    ax.plot(line[:, 0], line[:, 1], color=c, lw=lw, solid_capstyle="butt", zorder=zorder,
            ls=(0, (2.2, 1.4)) if dashed else "-")
    ax.add_patch(Polygon([tip, base + n * width / 2, base - n * width / 2], closed=True, fc=c, ec="none",
                         zorder=zorder))


def swatch_key(ax, x, y, entries, size=-1.6, sw=2.2, row_h=3.0, col_w=None, ncol=1):
    """Key for a glyph inside a diagram: entries are (colour or (fill, edge), label), laid out in
    ncol columns from (x, y) downward. Every colour a glyph uses must appear here."""
    for i, (c, lab) in enumerate(entries):
        r_, c_ = divmod(i, ncol)
        xx = x + c_ * (col_w or 0); yy = y - r_ * row_h
        pair = isinstance(c, (tuple, list)) and len(c) == 2
        fc, ec = (c if pair else (c, "none"))
        ax.add_patch(Rectangle((xx, yy - sw / 2), sw, sw * 0.8, fc=fc, ec=ec, lw=0.4, zorder=4))
        text(ax, xx + sw + 0.9, yy - sw * 0.1, lab, size=size, va="center", zorder=4)


def badge(ax, x, y, s, color=None, r=1.7):
    c = color or sp.INK
    ax.add_patch(Circle((x, y), r, fc=c, ec="none", zorder=5))
    text(ax, x, y - 0.05, s, size=-0.5, weight="bold", color="white", ha="center", zorder=6)


# ---------------------------------------------------------------- glyphs
def eeg_traces(ax, x, y, w, h, n=5, color=None, lw=0.4, seed=0):
    """Stacked synthetic EEG-like traces filling the box (x, y, w, h)."""
    r = np.random.default_rng(seed); t = np.linspace(0, 1, 300); c = color or sp.INK
    for i in range(n):
        f = r.uniform(7, 11); sig = (0.5 * np.sin(2 * np.pi * f * t * 2 + r.uniform(0, 6))
                                     + 0.25 * np.sin(2 * np.pi * 3 * t + r.uniform(0, 6))
                                     + 0.18 * r.standard_normal(t.size))
        sig = np.convolve(sig, np.ones(5) / 5, "same")
        yc = y + h * (n - i - 0.5) / n
        ax.plot(x + t * w, yc + sig / np.abs(sig).max() * h / n * 0.42, color=c, lw=lw, zorder=3)


def head(ax, cx, cy, r, carried=None, color=None, lw=0.45, ghost=None, bipolar=None, labels=False):
    """10-20 head: carried electrodes filled, the rest as faint rings. bipolar: (ch1, ch2) drawn as a link."""
    c = color or sp.INK; g = ghost or sp.FAINT
    ax.add_patch(Circle((cx, cy), r, fill=False, ec=sp.AXIS, lw=lw, zorder=2))
    ax.add_patch(Polygon([(cx - .1 * r, cy + .995 * r), (cx, cy + 1.13 * r), (cx + .1 * r, cy + .995 * r)],
                         closed=False, fill=False, ec=sp.AXIS, lw=lw, zorder=2))
    carried = set(POS) if carried is None else set(carried)
    er = 0.085 * r
    if bipolar:
        (x1, y1), (x2, y2) = POS[bipolar[0]], POS[bipolar[1]]
        ax.plot([cx + x1 * r, cx + x2 * r], [cy + y1 * r, cy + y2 * r], color=sp.TERRA, lw=0.8, zorder=3)
    for ch, (px, py) in POS.items():
        on = ch in carried
        ax.add_patch(Circle((cx + px * r, cy + py * r), er, fc=c if on else "white", ec=c if on else g,
                            lw=0.35, zorder=4))
        if labels and on:
            text(ax, cx + px * r, cy + py * r - er - 0.4, ch, size=-2.2, ha="center", va="top", zorder=5)


def bars(ax, x, y, w, h, values, colors=None, baseline=True, gap=0.3):
    v = np.asarray(values, float); n = v.size; bw = w / n
    vmax = np.abs(v).max() or 1; zero = y + (h / 2 if (v < 0).any() else 0)
    scale = (h / 2 if (v < 0).any() else h) / vmax
    cs = colors or [sp.BLUE] * n
    for i, (val, c) in enumerate(zip(v, cs)):
        ax.add_patch(Rectangle((x + i * bw + gap / 2, min(zero, zero + val * scale)), bw - gap, abs(val) * scale,
                               fc=c, ec="none", zorder=3))
    if baseline:
        ax.plot([x, x + w], [zero, zero], color=sp.AXIS, lw=0.4, zorder=4)


def hbars(ax, x, y, w, h, values, colors=None, gap=0.35, center=None):
    """Horizontal bars (e.g. a per-feature deviation profile). center: x of the zero line."""
    v = np.asarray(values, float); n = v.size; bh = h / n
    x0 = center if center is not None else x + w / 2
    scale = (w / 2) / (np.abs(v).max() or 1); cs = colors or [sp.BLUE] * n
    for i, (val, c) in enumerate(zip(v, cs)):
        yy = y + h - (i + 1) * bh
        ax.add_patch(Rectangle((min(x0, x0 + val * scale), yy + gap / 2), abs(val) * scale, bh - gap,
                               fc=c, ec="none", zorder=3))
    ax.plot([x0, x0], [y, y + h], color=sp.AXIS, lw=0.4, zorder=4)


def curve(ax, x, y, w, h, fx, color=None, lw=0.7, band=None):
    t = np.linspace(0, 1, 100); c = color or sp.BLUE
    if band is not None:
        lo, hi = band(t)
        ax.fill_between(x + t * w, y + lo * h, y + hi * h, color=sp.tint(c, 0.7), lw=0, zorder=2)
    ax.plot(x + t * w, y + fx(t) * h, color=c, lw=lw, zorder=3)


def matrix(ax, x, y, w, h, M, cmap=None):
    M = np.asarray(M, float); nr, nc = M.shape; cmap = cmap or sp.seq_cmap(sp.BLUE)
    Mn = (M - M.min()) / (np.ptp(M) or 1)
    cw, ch = w / nc, h / nr
    for i in range(nr):
        for j in range(nc):
            ax.add_patch(Rectangle((x + j * cw, y + h - (i + 1) * ch), cw * 0.92, ch * 0.92,
                                   fc=cmap(Mn[i, j]), ec="none", zorder=3))


def network(ax, cx, cy, r, n=6, color=None, seed=1):
    rr = np.random.default_rng(seed); c = color or sp.BLUE
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + 0.3
    P = np.c_[cx + r * np.cos(ang), cy + r * np.sin(ang)]
    for i in range(n):
        for j in range(i + 1, n):
            if rr.random() < 0.45:
                ax.plot(*P[[i, j]].T, color=sp.FAINT, lw=0.4, zorder=2)
    for px, py in P:
        ax.add_patch(Circle((px, py), 0.12 * r, fc=c, ec="white", lw=0.3, zorder=3))
