r"""Nature-style scientific figure toolkit (matplotlib).

    import sys; sys.path.insert(0, r"C:\Users\user\.claude\skills\visual-design\scripts")
    import sciplots as sp
    sp.setup("nature")                       # or "ieee", "slides"
    fig = sp.figure("double", 70)            # 183 mm wide, 70 mm tall
    ...                                      # one sp.<chart>(ax, ...) call per panel
    sp.label_panels([ax1, ax2, ...])         # a, b, c aligned per row
    sp.save(fig, "fig2_results")             # vector PDF + 600 dpi PNG

Rules built in (Nature figure guide): Arial/Helvetica, 5-7 pt text, 8 pt bold lowercase panel
letters, 89 / 120 / 183 mm widths, maximum height 170 mm, units in parentheses, no gridlines,
no drop shadows, fonts embedded as TrueType (type 42).

Keys: every chart draws a key that names each colour, symbol, line and band it uses, placed
outside the data area (above the axes by default; key_loc="right" or "bottom" also sit outside;
key_loc=None turns it off when one shared figure-level key is drawn instead). Nothing is drawn
over the data except the data.

Statistics: intervals are 95% unless stated; bootstrap intervals are percentile intervals;
ROC AUC counts ties as one half (Mann-Whitney); classifier CIs use a class-stratified bootstrap;
calibration uses quantile bins with Wilson intervals; net benefit follows Vickers and Elkin (2006);
the critical-difference diagram uses the Nemenyi test after a Friedman test (Demsar 2006). Each
function's docstring states what its marks mean, so the caption can say the same.
"""
from __future__ import annotations

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import ScaledTranslation, blended_transform_factory

# ---------------------------------------------------------------- geometry
MM = 1 / 25.4
WIDTHS = {  # mm
    "nature": {"single": 89, "onehalf": 120, "double": 183, "max_h": 170},
    "ieee": {"single": 88.9, "double": 181.9, "max_h": 230},
    "slides": {"single": 150, "double": 300, "max_h": 160},
}

# ---------------------------------------------------------------- colour
INK = "#22282D"        # all text and key marks
AXIS = "#5C656B"       # spines and ticks: visible at 0.5 pt, softer than black
SECOND = "#6B7780"     # secondary annotation text
FAINT = "#C8CFD3"      # light reference lines
HAIR = "#E8EBEA"       # bands, separators
GREY = "#A7ADB1"       # comparator / context series
PANEL = "#F4F6F5"      # subtle panel fill (diagrams, highlight bands)

#: House palette, refined from D:\NeuroIPS\eegbudget\viz.py and model_paper\figstyle.py: the same
#: desaturated hues with lightness tuned so neighbours separate under colour-vision deficiency.
HOUSE = ["#319287", "#AC704E", "#326893", "#809D62", "#835A8C"]
TEAL, TERRA, BLUE, SAGE, PLUM = HOUSE
#: Semantic clinical status colours. Never reuse them as categorical hues.
NORMAL, BORDERLINE, ABNORMAL = "#5F8A6B", "#C29049", "#A85549"

_STATE = {"journal": "nature", "cycle": HOUSE}


def tint(c, t=0.82):
    """Blend colour c toward white by t (0 = c, 1 = white)."""
    r = np.array(to_rgb(c)); return tuple(r + (1 - r) * t)


def seq_cmap(c, light=0.94, name=None):
    """Sequential map from a near-white tint of c to c (lightness changes monotonically)."""
    return LinearSegmentedColormap.from_list(name or f"seq_{c}", [tint(c, light), c])


def div_cmap(low=TERRA, high=BLUE, mid="#F3F2EF"):
    """Diverging map for quantities with a meaningful zero."""
    return LinearSegmentedColormap.from_list("div_house", [low, mid, high])


def colors(n=None, name=None):
    """Series colours: the house palette, or a named set from assets/palettes.json."""
    cyc = HOUSE
    if name:
        from figtools import palette
        cyc = palette(name)["data"]
    return cyc if n is None else [cyc[i % len(cyc)] for i in range(n)]


def _fs(d=0.0):
    return mpl.rcParams["font.size"] + d


def _minus(s):
    return s.replace("-", "\u2212")


# ---------------------------------------------------------------- style and layout
def setup(journal="nature", palette=None):
    """Apply the journal preset. Sizes are printed sizes when the figure is placed at 100%."""
    base = {"nature": 6.0, "ieee": 7.5, "slides": 12.0}[journal]
    lw = {"nature": 0.5, "ieee": 0.6, "slides": 1.0}[journal]
    _STATE["journal"] = journal
    _STATE["cycle"] = colors(name=palette) if palette else HOUSE
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "font.size": base, "axes.labelsize": base, "axes.titlesize": base + 0.5,
        "xtick.labelsize": base - 0.5, "ytick.labelsize": base - 0.5, "legend.fontsize": base - 0.5,
        "axes.titleweight": "normal", "axes.titlelocation": "left", "axes.titlepad": 3,
        "text.color": INK, "axes.labelcolor": INK, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "axes.edgecolor": AXIS, "xtick.color": AXIS, "ytick.color": AXIS,
        "axes.linewidth": lw, "xtick.major.width": lw, "ytick.major.width": lw,
        "xtick.minor.width": lw * 0.8, "ytick.minor.width": lw * 0.8,
        "xtick.major.size": 2.2 * base / 6, "ytick.major.size": 2.2 * base / 6,
        "xtick.minor.size": 1.2 * base / 6, "ytick.minor.size": 1.2 * base / 6,
        "xtick.major.pad": 1.8, "ytick.major.pad": 1.8, "axes.labelpad": 2.5,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "axes.prop_cycle": mpl.cycler(color=_STATE["cycle"]),
        "lines.linewidth": 2 * lw, "lines.markersize": 3.0 * base / 6,
        "lines.solid_capstyle": "round", "patch.linewidth": lw, "errorbar.capsize": 0,
        "legend.frameon": False, "legend.handlelength": 1.3, "legend.handleheight": 0.7,
        "legend.handletextpad": 0.45, "legend.borderaxespad": 0.2, "legend.labelspacing": 0.3,
        "legend.columnspacing": 1.1, "legend.borderpad": 0.1,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "figure.dpi": 200, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.w_pad": 1.5 * MM, "figure.constrained_layout.h_pad": 1.5 * MM,
        "figure.constrained_layout.wspace": 0.05, "figure.constrained_layout.hspace": 0.07,
        "axes.unicode_minus": True,
    })


def figure(width="double", height_mm=60, **subplots_kw):
    """New figure at the journal's printed width. `width` is a preset name or millimetres."""
    j = WIDTHS[_STATE["journal"]]
    w = j[width] if isinstance(width, str) else float(width)
    if height_mm > j["max_h"]:
        raise ValueError(f"height {height_mm} mm exceeds the {_STATE['journal']} maximum of {j['max_h']} mm")
    if subplots_kw:
        return plt.subplots(figsize=(w * MM, height_mm * MM), **subplots_kw)
    return plt.figure(figsize=(w * MM, height_mm * MM))


def grid(fig, rows, height_ratios=None):
    """Lay out panels row by row. rows: one list of width ratios per row, e.g. [[1.2, 1], [1, 1, 1]].

    Each row is its own sub-grid, so labels or text outside one panel (a forest plot's number
    column, a long y label) never widen the margins of panels in other rows. Returns a list of
    rows, each a list of Axes. For panels stacked inside one cell, call .subgridspec on the
    returned axes' subplotspec or build that row by hand.
    """
    gs = fig.add_gridspec(len(rows), 1, height_ratios=height_ratios)
    out = []
    for i, wr in enumerate(rows):
        sg = gs[i].subgridspec(1, len(wr), width_ratios=wr)
        out.append([fig.add_subplot(sg[j]) for j in range(len(wr))])
    return out


def label_panels(axes, letters="abcdefghijklmnop", pad_mm=0.8, align_rows=True):
    """Letter each panel at the top-left of everything it draws (tick labels, axis label, key).

    Letters of panels in the same row share one baseline. Call after drawing, before saving.
    """
    fig = axes[0].figure
    fig.canvas.draw(); r = fig.canvas.get_renderer(); dpi = fig.dpi
    info = [(ax, ax.get_tightbbox(r), ax.get_window_extent(r)) for ax in axes]
    tol = 4 * MM * dpi
    tops = {}
    if align_rows:
        for ax, tb, ab in info:
            k = next((k for k in tops if abs(k - ab.y1) < tol), ab.y1)
            tops[k] = max(tops.get(k, -1e9), tb.y1)
    size = {"nature": 8, "ieee": 9, "slides": 16}[_STATE["journal"]]
    out = []
    for (ax, tb, ab), letter in zip(info, letters):
        k = next((k for k in tops if abs(k - ab.y1) < tol), None)
        top = tops[k] if k is not None else tb.y1
        margin = (ab.x0 - tb.x0) / dpi                     # inches of tick labels / axis label left of the axes
        if margin * 25.4 >= size * 0.3528 * 0.8 + 1.0:     # room for the letter above the labels
            off = ScaledTranslation(-(margin + 0.2 * MM), (top - ab.y1) / dpi + pad_mm * MM, fig.dpi_scale_trans)
            ha = "left"
        else:                                             # no margin: sit in the gutter, clear of the key
            off = ScaledTranslation(-1.0 * MM, (top - ab.y1) / dpi + pad_mm * MM, fig.dpi_scale_trans)
            ha = "right"
        out.append(ax.text(0, 1, letter, transform=ax.transAxes + off, fontsize=size, fontweight="bold",
                           va="top", ha=ha, color=INK))
    return out


def despine(ax, left=True, bottom=True):
    ax.spines["left"].set_visible(left); ax.spines["bottom"].set_visible(bottom)
    if not left:
        ax.tick_params(axis="y", length=0)
    if not bottom:
        ax.tick_params(axis="x", length=0)
    return ax


# ---------------------------------------------------------------- keys (always outside the data)
def h_line(color=INK, lw=1.0, ls="-", marker=None, ms=3.0, mfc=None):
    return Line2D([], [], color=color, lw=lw, ls=ls, marker=marker, ms=ms, mfc=mfc or color,
                  mec="white" if marker in ("o", "s", "D") else color, mew=0.4)


def h_marker(color=INK, marker="o", ms=3.0, mfc=None, mec=None):
    return Line2D([], [], ls="", marker=marker, ms=ms, color=color, mfc=mfc or color, mec=mec or mfc or color,
                  mew=0.5)


def h_patch(fc, ec="none", alpha=1.0):
    return Patch(fc=fc, ec=ec, lw=0.5, alpha=alpha)


def _auto_ncol(ax, labels, fontsize, avail_mm=None):
    """Columns that fit the key within the panel width (rows wrap instead of widening the panel)."""
    fig = ax.figure
    width_mm = avail_mm or ax.get_position().width * fig.get_figwidth() / MM
    em = fontsize * 0.3528                       # 1 em in mm (pt -> mm)
    entry = [(mpl.rcParams["legend.handlelength"] + mpl.rcParams["legend.handletextpad"]) * em
             + 0.56 * em * len(str(l)) for l in labels]
    for ncol in range(len(labels), 0, -1):
        cols = [max(entry[i::ncol]) for i in range(ncol)]   # widest entry per column (approximation)
        if sum(cols) + (ncol - 1) * mpl.rcParams["legend.columnspacing"] * em <= 0.85 * width_mm:
            return ncol
    return 1


def key(ax, entries, loc="top", ncol=None, **kw):
    """Key outside the data area. entries: list of (handle, label). loc: 'top' (rows above the axes),
    'right' (column beside the axes) or 'bottom' (rows below the x label). Unless ncol is given,
    entries wrap into as many rows as the panel width needs, so a key never widens its panel."""
    if not entries or loc is None:
        return None
    hs, ls = zip(*entries)
    fs = kw.get("fontsize", mpl.rcParams["legend.fontsize"])
    fs = float(fs) if not isinstance(fs, str) else mpl.rcParams["font.size"]
    if loc == "right":
        kw = {"loc": "upper left", "bbox_to_anchor": (1.02, 1), "ncol": ncol or 1, **kw}
    else:
        n = ncol or _auto_ncol(ax, ls, fs)
        anchor = (0, 1.02) if loc == "top" else (0, -0.22)
        kw = {"loc": "lower left" if loc == "top" else "upper left", "bbox_to_anchor": anchor, "ncol": n, **kw}
    kw.setdefault("borderaxespad", 0)
    ncol_ = kw["ncol"]
    if loc != "right" and 1 < ncol_ < len(hs):
        # matplotlib fills keys column by column; reorder (and pad) so the key reads left to right
        nrow = math.ceil(len(hs) / ncol_)
        grid = [[None] * ncol_ for _ in range(nrow)]
        for i, e in enumerate(zip(hs, ls)):
            grid[i // ncol_][i % ncol_] = e
        blank = (Line2D([], [], lw=0, alpha=0), "")
        order = [grid[r][c] or blank for c in range(ncol_) for r in range(nrow)]
        hs, ls = zip(*order)
    return ax.legend(hs, ls, **kw)


def fmt_p(p):
    """Exact P value: P = 0.03, P = 0.003, P = 4.2 x 10^-6. (Never stars alone.)"""
    if p >= 0.001:
        s = f"{p:.2f}" if p >= 0.01 else f"{p:.3f}"
        return f"$P$ = {s}"
    e = int(math.floor(math.log10(p))); m = p / 10 ** e
    if round(m, 1) >= 10:
        m, e = m / 10, e + 1
    return rf"$P$ = {m:.1f} $\times$ 10$^{{{e}}}$"


def fmt_ci(est, lo, hi, d=2):
    return _minus(f"{est:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]")


def save(fig, path, formats=("pdf", "png")):
    """Write vector PDF (for the paper) and PNG (for checking); return the PDF path."""
    from pathlib import Path
    p = Path(path)
    for f in formats:
        fig.savefig(p.with_suffix("." + f))
    return p.with_suffix(".pdf")


# ---------------------------------------------------------------- composition: no wasted space
def _is_cbar(ax):
    return ax.get_label() == "<colorbar>" or hasattr(ax, "_colorbar")


def _locked(ax):
    """Axes whose shape is fixed (equal aspect, images, box aspect): they cannot stretch to fill a cell."""
    return ax.get_visible() and not _is_cbar(ax) and (ax.get_aspect() != "auto" or ax.get_box_aspect() is not None)


def _slack(fig):
    """Per locked axes: (axes, horizontal slack mm, vertical slack mm, height/width) inside its layout cell."""
    fig.canvas.draw()
    W, H = fig.get_size_inches(); out = []
    for ax in fig.axes:
        if not _locked(ax):
            continue
        o = ax.get_position(original=True); a = ax.get_position(original=False)
        ratio = (a.height * H) / max(a.width * W, 1e-9)
        out.append((ax, (o.width - a.width) * W / MM, (o.height - a.height) * H / MM, ratio, o))
    return out


def _refresh_physical(fig, tol_mm=0.5):
    """Re-draw panels that lay themselves out in millimetres (e.g. critical_difference) at their final width."""
    fig.canvas.draw(); changed = False
    for ax in fig.axes:
        spec = getattr(ax, "_sp_physical", None)
        if not spec:
            continue
        func, args, kw, w_used = spec
        w_now = ax.get_position(original=True).width * fig.get_figwidth() / MM
        if abs(w_now - w_used) > tol_mm:
            ax.cla(); func(ax, *args, _w_mm=w_now, **kw); changed = True
    return changed


def _need_mm(fig, tol_mm):
    """How much taller (+) or shorter (-) the figure should be so fixed-shape panels fill their cells."""
    s = _slack(fig)
    if not s:
        return 0.0
    H = fig.get_size_inches()[1]; rows = {}
    for ax, sh, sv, ratio, o in s:
        rows.setdefault(round((o.y0 + o.y1) / 2 * H / MM / 4), []).append((sh, sv, ratio))
    need = 0.0
    for items in rows.values():
        if any(sh > tol_mm for sh, sv, r in items):
            need += max(sh * r for sh, sv, r in items if sh > tol_mm)
        elif all(sv > tol_mm for sh, sv, r in items):
            need -= min(sv for sh, sv, r in items)
    return need


def fit_height(fig, tol_mm=0.6, max_iter=16, refresh=True):
    """Resize the figure height so fixed-shape panels (squares, images, size-aware diagrams) fill
    their cells: rows with spare height shrink, rows limited by height grow. Width stays at the
    journal width; stretchable panels follow. Uses a bracketed bisection on the height, which
    converges even though the layout responds non-linearly. Returns the final height in mm."""
    j = WIDTHS[_STATE["journal"]]; W = fig.get_size_inches()[0]

    def need_at(h):
        fig.set_size_inches(W, h * MM)
        if refresh:
            _refresh_physical(fig)
        return _need_mm(fig, tol_mm)

    h = fig.get_size_inches()[1] / MM; n = need_at(h)
    if abs(n) < tol_mm:
        return h
    lo, hi, step = (h, None, max(abs(n), 2.0)) if n > 0 else (None, h, max(abs(n), 2.0))
    best = (abs(n), h)
    for _ in range(max_iter):                       # bracket the root
        if lo is not None and hi is not None:
            break
        h2 = min(max((h + step) if lo is not None else (h - step), 25.0), j["max_h"])
        n2 = need_at(h2); best = min(best, (abs(n2), h2))
        if n2 > 0:
            lo = h2
        else:
            hi = h2
        h = h2; step *= 1.6
        if h2 in (25.0, j["max_h"]):
            break
    if lo is not None and hi is not None:
        for _ in range(max_iter):                   # bisection
            mid = (lo + hi) / 2; nm = need_at(mid); best = min(best, (abs(nm), mid))
            if abs(nm) < tol_mm or hi - lo < 0.2:
                break
            if nm > 0:
                lo = mid
            else:
                hi = mid
    need_at(best[1])
    fig.canvas.draw()
    return best[1]


def blank_report(fig, min_band_mm=7.0, min_slack_mm=2.5, fill_min=0.7):
    """List layout problems that leave visible empty space:
    - fixed-shape panels with spare room in their cell (run fit_height, or re-compose the row),
    - axis-free panels (diagrams) whose drawing fills too little of the panel,
    - empty horizontal or vertical bands across the whole figure wider than min_band_mm."""
    problems = []
    for ax, sh, sv, ratio, o in _slack(fig):
        if sh > min_slack_mm or sv > min_slack_mm:
            problems.append(f"panel at ({o.x0:.2f}, {o.y0:.2f}) leaves {sh:.1f} mm spare width, {sv:.1f} mm spare height")
    r = fig.canvas.get_renderer()
    for ax in fig.axes:
        if ax.axison or _is_cbar(ax) or not ax.get_visible():
            continue
        ab = ax.get_window_extent(r)
        arts = [a for a in ax.get_children() if a.get_visible() and hasattr(a, "get_window_extent")
                and a is not ax.patch and not isinstance(a, mpl.spines.Spine)]
        bbs = []
        for a in arts:
            try:
                b = a.get_window_extent(r)
                if b.width > 0 and b.height > 0 and np.isfinite(b.bounds).all():
                    bbs.append(b)
            except Exception:
                pass
        if bbs:
            u = mpl.transforms.Bbox.union(bbs)
            fx = min(u.width, ab.width) / ab.width; fy = min(u.height, ab.height) / ab.height
            if fx < fill_min or fy < fill_min:
                problems.append(f"axis-free panel at ({ax.get_position().x0:.2f}, {ax.get_position().y0:.2f}) "
                                f"fills {100 * fx:.0f}% of its width and {100 * fy:.0f}% of its height")
    # empty bands across the rendered figure (inside the content box)
    img = np.asarray(r.buffer_rgba())[..., :3].min(axis=2) < 245
    ys, xs = np.nonzero(img)
    if ys.size:
        img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        px_mm = fig.dpi * MM
        for axis_, name in ((1, "horizontal"), (0, "vertical")):
            empty = ~img.any(axis=axis_); run = best = 0
            for e in empty:
                run = run + 1 if e else 0; best = max(best, run)
            if best / px_mm > min_band_mm:
                problems.append(f"empty {name} band of {best / px_mm:.1f} mm across the figure")
    return problems


def finish(fig, panels, path=None, letters="abcdefghijklmnop", fit=True, check=True):
    """One call to finish a multi-panel figure: fit the height to fixed-shape panels, letter the
    panels (aligned per row), check for text overlaps and empty space, and save PDF + PNG.
    Returns a list of problems (empty when the figure is clean)."""
    if fit:
        fit_height(fig)
    label_panels(panels, letters)
    if fit:
        fit_height(fig, refresh=False)   # letters take layout space; refit without re-drawing lettered panels
    problems = []
    if check:
        from figtools import find_text_overlaps
        problems += [f"text overlap: {a!r} / {b!r}" for a, b in find_text_overlaps(fig)]
        problems += blank_report(fig)
    if path:
        save(fig, path)
    for p in problems:
        print("LAYOUT:", p)
    return problems


# ---------------------------------------------------------------- statistics helpers
def _kde(x, grid, bw=None):
    """Gaussian KDE with Silverman's bandwidth, 0.9 min(SD, IQR/1.34) n^(-1/5)."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    s = np.std(x, ddof=1) if x.size > 1 else 1.0
    iqr = np.subtract(*np.percentile(x, [75, 25])) if x.size > 3 else s
    h = bw or 0.9 * min(s, iqr / 1.34 if iqr > 0 else s) * x.size ** (-0.2)
    h = h if h > 0 else 1e-3
    z = (grid[:, None] - x[None, :]) / h
    return np.exp(-0.5 * z ** 2).sum(1) / (x.size * h * math.sqrt(2 * math.pi))


def boot_ci(x, stat=np.mean, n=5000, alpha=0.05, rng=0):
    """Point estimate and percentile bootstrap CI."""
    r = np.random.default_rng(rng); x = np.asarray(x, float)
    bs = np.array([stat(r.choice(x, x.size)) for _ in range(n)])
    return stat(x), *np.quantile(bs, [alpha / 2, 1 - alpha / 2])


def boot_diff(a, b, paired=False, n=5000, rng=0):
    """Mean difference b - a and its bootstrap distribution (groups resampled independently,
    or subject-wise differences resampled when paired)."""
    r = np.random.default_rng(rng); a = np.asarray(a, float); b = np.asarray(b, float)
    if paired:
        d = b - a
        return d.mean(), np.array([r.choice(d, d.size).mean() for _ in range(n)])
    return b.mean() - a.mean(), np.array([r.choice(b, b.size).mean() - r.choice(a, a.size).mean()
                                          for _ in range(n)])


def wilson(k, n, z=1.959963984540054):
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return np.nan, np.nan
    p = k / n; den = 1 + z ** 2 / n; cen = (p + z ** 2 / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return cen - half, cen + half


def average_ranks(scores, higher_is_better=True):
    """Average rank of each model across datasets (ties get their mean rank) and the Friedman test.

    scores: (n_models, n_datasets). Returns (ranks, friedman_statistic, P). Report the Friedman P
    before reading a critical-difference diagram.
    """
    from scipy.stats import rankdata, friedmanchisquare
    s = np.asarray(scores, float)
    R = np.column_stack([rankdata(-s[:, j] if higher_is_better else s[:, j]) for j in range(s.shape[1])])
    chi, p = friedmanchisquare(*s)
    return R.mean(1), chi, p


def _jitter(n, width, rng):
    return (rng.random(n) - 0.5) * width


# ---------------------------------------------------------------- distributions and groups
def raincloud(ax, groups, colors_=None, width=0.8, point_size=4, key_loc="top", rng=0):
    """Rows of: kernel density (cloud), interquartile range (thick bar), median (open circle),
    and every observation (jittered dots). groups: dict name -> array."""
    r = np.random.default_rng(rng); names = list(groups); cs = colors_ or colors(len(names))
    for i, (nm, c) in enumerate(zip(names, cs)):
        x = np.asarray(groups[nm], float); x = x[np.isfinite(x)]
        lo, hi = x.min(), x.max(); pad = 0.08 * (hi - lo + 1e-9)
        g = np.linspace(lo - pad, hi + pad, 200); d = _kde(x, g); d = d / d.max() * width * 0.45
        q1, med, q3 = np.percentile(x, [25, 50, 75])
        ax.fill_between(g, i, i + d, color=tint(c, 0.55), lw=0, zorder=1)
        ax.plot(g, i + d, color=c, lw=0.6, zorder=2)
        ax.plot([q1, q3], [i, i], color=c, lw=2.2, solid_capstyle="butt", zorder=3)
        ax.plot([med], [i], "o", ms=2.6, mfc="white", mec=c, mew=0.7, zorder=4)
        ax.scatter(x, i - 0.14 - np.abs(_jitter(x.size, 0.22, r)), s=point_size, color=c, alpha=0.55, lw=0, zorder=2)
    ax.set_yticks(range(len(names)), names); ax.set_ylim(-0.6, len(names) - 0.2); despine(ax, left=False)
    key(ax, [(h_patch(tint(GREY, 0.55), GREY), "density"), (h_line(GREY, lw=2.2), "IQR"),
             (h_marker(GREY, "o", mfc="white", mec=GREY), "median"), (h_marker(GREY, "o", ms=2), "observations")],
        loc=key_loc)
    return ax


def bars_points(ax, groups, colors_=None, point_size=5, key_loc="top", rng=0):
    """Pale bar = mean; black line = 95% bootstrap CI of the mean; dots = individual values."""
    r = np.random.default_rng(rng); names = list(groups); cs = colors_ or colors(len(names))
    for i, (nm, c) in enumerate(zip(names, cs)):
        x = np.asarray(groups[nm], float); m, lo, hi = boot_ci(x)
        ax.bar(i, m, width=0.62, color=tint(c, 0.78), edgecolor=c, lw=0.6, zorder=1)
        ax.scatter(i - 0.06 + _jitter(x.size, 0.3, r), x, s=point_size, color=c, alpha=0.8, lw=0, zorder=3)
        ax.plot([i + 0.2, i + 0.2], [lo, hi], color=INK, lw=0.8, zorder=4, solid_capstyle="butt")
    ax.set_xticks(range(len(names)), names); ax.tick_params(axis="x", length=0)
    key(ax, [(h_patch(tint(GREY, 0.78), GREY), "mean"), (h_line(INK, lw=0.8), "95% CI"),
             (h_marker(GREY, "o", ms=2.2), "individual values")], loc=key_loc)
    return ax


def paired_slopes(ax, before, after, labels=("before", "after"), color=None, unit_name="subject", key_loc="top"):
    """Grey line per unit (subject, seed, site), coloured line = mean change."""
    c = color or BLUE; b = np.asarray(before, float); a = np.asarray(after, float)
    for y0, y1 in zip(b, a):
        ax.plot([0, 1], [y0, y1], color=FAINT, lw=0.5, zorder=1)
    ax.scatter(np.zeros_like(b), b, s=5, color=GREY, lw=0, zorder=2)
    ax.scatter(np.ones_like(a), a, s=5, color=GREY, lw=0, zorder=2)
    ax.plot([0, 1], [b.mean(), a.mean()], color=c, lw=1.4, marker="o", ms=3.5, mec="white", mew=0.5, zorder=3)
    ax.set_xticks([0, 1], labels); ax.set_xlim(-0.35, 1.35); ax.tick_params(axis="x", length=0)
    despine(ax, bottom=False)
    key(ax, [(h_line(FAINT, lw=0.6, marker="o", ms=2, mfc=GREY), f"one {unit_name}"),
             (h_line(c, lw=1.4, marker="o", ms=3), "mean")], loc=key_loc)
    return ax


def estimation(ax_raw, ax_diff, groups, ref=None, paired=False, colors_=None, key_loc="top", rng=0):
    """Cumming estimation plot. Top: every observation, with mean (dot) and +/- 1 SD (line) beside
    each group. Bottom: mean difference from the reference group (dot), its 95% percentile bootstrap
    CI (line) and the bootstrap distribution (shaded curve)."""
    r = np.random.default_rng(rng); names = list(groups); ref = ref or names[0]
    cs = colors_ or colors(len(names))
    for i, (nm, c) in enumerate(zip(names, cs)):
        x = np.asarray(groups[nm], float)
        ax_raw.scatter(i + _jitter(x.size, 0.38, r), x, s=5, color=c, alpha=0.7, lw=0, zorder=2)
        m, sd = x.mean(), x.std(ddof=1)
        ax_raw.plot([i + 0.3] * 2, [m - sd, m + sd], color=INK, lw=0.6, zorder=3)
        ax_raw.plot([i + 0.3], [m], "o", ms=2.4, color=INK, zorder=3)
    ax_raw.set_xticks(range(len(names))); ax_raw.tick_params(axis="x", length=0, labelbottom=False)
    ax_raw.set_xlim(-0.6, len(names) - 0.4); despine(ax_raw, bottom=False)
    a = np.asarray(groups[ref], float)
    ax_diff.axhline(0, color=FAINT, lw=0.5, zorder=0)
    for i, (nm, c) in enumerate(zip(names, cs)):
        if nm == ref:
            ax_diff.plot([i], [0], "o", ms=2.4, mfc="white", mec=INK, mew=0.6, zorder=4)
            continue
        est, bs = boot_diff(a, groups[nm], paired=paired, rng=rng)
        lo, hi = np.quantile(bs, [0.025, 0.975])
        g = np.linspace(bs.min(), bs.max(), 200); d = _kde(bs, g); d = d / d.max() * 0.35
        ax_diff.fill_betweenx(g, i, i + d, color=tint(c, 0.55), lw=0, zorder=1)
        ax_diff.plot([i, i], [lo, hi], color=INK, lw=0.8, zorder=3)
        ax_diff.plot([i], [est], "o", ms=3, color=INK, zorder=4)
    ax_diff.set_xticks(range(len(names)), names); ax_diff.tick_params(axis="x", length=0)
    key(ax_raw, [(h_marker(GREY, "o", ms=2.2), "observations"),
                 (h_line(INK, lw=0.6, marker="o", ms=2.4), "mean \u00b1 SD")], loc=key_loc)
    key(ax_diff, [(h_patch(tint(GREY, 0.55)), "bootstrap distribution"),
                  (h_line(INK, lw=0.8, marker="o", ms=3), "difference, 95% CI"),
                  (h_marker(INK, "o", mfc="white", mec=INK, ms=2.4), "reference")], loc=key_loc)
    return ax_raw, ax_diff


def ridgeline(ax, groups, colors_=None, overlap=0.6):
    """Kernel densities of each group, stacked; each ridge is named by its y tick."""
    names = list(groups); cs = colors_ or colors(len(names))
    allx = np.concatenate([np.asarray(v, float) for v in groups.values()])
    g = np.linspace(np.nanmin(allx), np.nanmax(allx), 300)
    for i, (nm, c) in enumerate(zip(names[::-1], cs[::-1])):
        d = _kde(groups[nm], g); d = d / d.max() * (1 + overlap)
        ax.fill_between(g, i, i + d, color=tint(c, 0.45), lw=0, zorder=2 * i)
        ax.plot(g, i + d, color=c, lw=0.7, zorder=2 * i + 1)
    ax.set_yticks(np.arange(len(names)) + 0.25, names[::-1]); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    return ax


# ---------------------------------------------------------------- benchmarks
def forest(ax, names, est, lo, hi, ref=None, ref_label="reference", highlight=None, value_col=True, d=3,
           color=None, key_loc="top", value_label="estimate [95% CI]"):
    """Estimate (marker) and 95% CI (line) per row; dashed line = reference value; the numbers
    are repeated in a column right of the axes."""
    y = np.arange(len(names))[::-1]; c = color or BLUE
    hl_set = set(np.atleast_1d(highlight)) if highlight is not None else set()
    for yi, nm, e, l, h in zip(y, names, est, lo, hi):
        hl = nm in hl_set; col = c if hl else GREY
        ax.plot([l, h], [yi, yi], color=col, lw=1.0 if hl else 0.8, zorder=2)
        ax.plot([e], [yi], "s" if hl else "o", ms=3.2 if hl else 2.6, color=col, mec="white", mew=0.4, zorder=3)
    if ref is not None:
        ax.axvline(ref, color=AXIS, lw=0.5, ls=(0, (2, 2)), zorder=0)
    ax.set_yticks(y, names); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.7, len(names) - 0.3)
    if value_col:
        tr = blended_transform_factory(ax.transAxes, ax.transData)
        for yi, e, l, h in zip(y, est, lo, hi):
            ax.text(1.02, yi, fmt_ci(e, l, h, d), transform=tr, va="center", ha="left", fontsize=_fs(-0.5))
        ax.text(1.02, len(names) - 0.3, value_label, transform=tr, va="bottom", ha="left", fontsize=_fs(-0.5),
                color=SECOND)
    ent = []
    if hl_set:
        ent.append((h_line(c, lw=1.0, marker="s", ms=3.2), "proposed, 95% CI"))
    ent.append((h_line(GREY, lw=0.8, marker="o", ms=2.6), "comparator, 95% CI"))
    if ref is not None:
        ent.append((h_line(AXIS, lw=0.5, ls=(0, (2, 2))), ref_label))
    key(ax, ent, loc=key_loc)
    return ax


def benchmark_dots(ax, models, datasets, est, lo=None, hi=None, highlight=None, colors_=None, dodge=0.18,
                   key_loc="top"):
    """One row per model; per dataset a dot (estimate) and line (95% CI), dodged in key order.
    est, lo, hi: arrays (n_models, n_datasets). A pale band marks the highlighted model."""
    est = np.asarray(est, float); nd = len(datasets); cs = colors_ or colors(nd)
    y = np.arange(len(models))[::-1]; offs = -(np.arange(nd) - (nd - 1) / 2) * dodge
    for i, yi in enumerate(y):
        if highlight is not None and models[i] in np.atleast_1d(highlight):
            ax.axhspan(yi - 0.45, yi + 0.45, color=PANEL, lw=0, zorder=0)
        for j in range(nd):
            if lo is not None:
                ax.plot([lo[i][j], hi[i][j]], [yi + offs[j]] * 2, color=cs[j], lw=0.7, zorder=2)
            ax.plot([est[i, j]], [yi + offs[j]], "o", ms=2.8, color=cs[j], mec="white", mew=0.35, zorder=3)
    ax.set_yticks(y, models); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, len(models) - 0.4)
    ent = [(h_line(c, lw=0.7, marker="o", ms=2.8), dn) for c, dn in zip(cs, datasets)]
    if highlight is not None:
        ent.append((h_patch(PANEL, FAINT), "proposed"))
    key(ax, ent, loc=key_loc)
    return ax


_NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
                11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391}   # q_0.05(k, inf) / sqrt(2)


def critical_difference(ax, avg_ranks, n_datasets, highlight=None, friedman_p=None, _w_mm=None):
    """Demsar critical-difference diagram (Nemenyi, alpha = 0.05; valid only after a significant
    Friedman test, see average_ranks). The best rank is on the right. A thick bar joins models whose
    mean ranks differ by less than the CD, i.e. that are not significantly different.
    avg_ranks: dict model -> mean rank (1 = best). Returns the CD.

    Drawn in millimetres: the rank axis spans the panel width between the two label columns, and
    the panel's height is set to what the diagram needs (anchored to the top of its cell), so
    fit_height/finish can close any leftover space in the row.
    """
    names = sorted(avg_ranks, key=avg_ranks.get)
    ranks = np.array([avg_ranks[n] for n in names]); k = len(names)
    cd = _NEMENYI_Q05[k] * math.sqrt(k * (k + 1) / (6 * n_datasets))
    fs = _fs(-0.5); em = fs * 0.3528; cw = 0.5 * em             # mm per character (Arial average, measured ~0.46 em)
    fig = ax.figure
    W = _w_mm or ax.get_position(original=True).width * fig.get_figwidth() / MM
    ax._sp_physical = (critical_difference, (avg_ranks, n_datasets), dict(highlight=highlight, friedman_p=friedman_p), W)
    half = math.ceil(k / 2)
    right = names[:half]; left = names[half:][::-1]
    lab_r = [f"{n} ({avg_ranks[n]:.2f})" for n in right]; lab_l = [f"({avg_ranks[n]:.2f}) {n}" for n in left]
    Lw = max(len(s) for s in lab_l) * cw + 3.0 if lab_l else 3.0
    Rw = max(len(s) for s in lab_r) * cw + 3.0
    xa, xb = Lw + 1.5, W - Rw - 1.5                              # rank axis from rank k (left) to rank 1 (right)
    X = lambda rk: xa + (k - rk) / (k - 1) * (xb - xa)
    cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and ranks[j + 1] - ranks[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    row, cl = 3.1, 1.9                                           # label row pitch, clique bar pitch (mm)
    Hh = 7.5 + 1.5 + cl * max(len(cliques), 1) + 1.5 + row * half + 2.5 + (10.4 if friedman_p is not None else 7.0)
    ax.set_axis_off(); ax.set_xlim(0, W); ax.set_ylim(0, Hh)
    ax.set_box_aspect(Hh / W); ax.set_anchor("N")
    y_ax = Hh - 7.5
    ax.plot([X(k), X(1)], [y_ax, y_ax], color=AXIS, lw=0.6)
    for rk in range(1, k + 1):
        ax.plot([X(rk)] * 2, [y_ax, y_ax + 0.9], color=AXIS, lw=0.6)
        ax.text(X(rk), y_ax + 1.3, str(rk), ha="center", va="bottom", fontsize=fs)
    ax.text((X(k) + X(1)) / 2, Hh - 0.2, "mean rank (1 = best)", ha="center", va="top", fontsize=fs)
    for n_, (i, j) in enumerate(cliques):
        yb = y_ax - 1.5 - cl * n_ - cl / 2
        ax.plot([X(ranks[j]) - 0.8, X(ranks[i]) + 0.8], [yb, yb], color=INK, lw=1.6, solid_capstyle="round")
    y0 = y_ax - 1.5 - cl * max(len(cliques), 1) - 1.5 - row / 2
    for side, group, labs in (("right", right, lab_r), ("left", left, lab_l)):
        for n_, (nm, lab) in enumerate(zip(group, labs)):
            rk = avg_ranks[nm]; yl = y0 - n_ * row
            hl = nm == highlight; col = BLUE if hl else INK
            xe = xb + 1.2 if side == "right" else xa - 1.2
            ax.plot([X(rk), X(rk), xe], [y_ax, yl, yl], color=col, lw=0.9 if hl else 0.6)
            ax.text(xe + (0.6 if side == "right" else -0.6), yl, lab, ha="left" if side == "right" else "right",
                    va="center", fontsize=fs, color=col, fontweight="bold" if hl else "normal")
    # key under the diagram, one item per line: what a bar means, the CD scale, and the omnibus test
    base = 3.4 if friedman_p is not None else 0.0
    ax.plot([xa, xa + 5], [base + 4.9, base + 4.9], color=INK, lw=1.6, solid_capstyle="round")
    ax.text(xa + 6.2, base + 4.9, "joined models: not significantly different", va="center", fontsize=fs)
    x1 = xa + cd / (k - 1) * (xb - xa)
    ax.plot([xa, x1], [base + 1.4, base + 1.4], color=INK, lw=0.6)
    for xx in (xa, x1):
        ax.plot([xx, xx], [base + 0.8, base + 2.0], color=INK, lw=0.6)
    ax.text(x1 + 1.2, base + 1.4, f"critical difference (CD) = {cd:.2f}; Nemenyi, α = 0.05, N = {n_datasets}",
            va="center", fontsize=fs)
    if friedman_p is not None:
        ax.text(xa, 1.4, f"Friedman test, {fmt_p(friedman_p)}", va="center", fontsize=fs)
    return cd


def rank_heatmap(ax, values, models, datasets, higher_is_better=True, fmt="{:.3f}", color=None, key_loc="bottom"):
    """Cell shade = rank within each dataset (darker = better); text = value; bold = best (ties included)."""
    from scipy.stats import rankdata
    v = np.asarray(values, float); c = color or TEAL
    rank = np.column_stack([rankdata(-v[:, j] if higher_is_better else v[:, j], method="min")
                            for j in range(v.shape[1])]) - 1
    shade = 1 - rank / max(v.shape[0] - 1, 1)
    cm = seq_cmap(c, light=0.97); norm = mpl.colors.Normalize(-0.15, 1.35)
    ax.imshow(shade, cmap=cm, norm=norm, aspect="auto")
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            dark = shade[i, j] > 0.8
            ax.text(j, i, _minus(fmt.format(v[i, j])), ha="center", va="center", fontsize=_fs(-0.5),
                    fontweight="bold" if rank[i, j] == 0 else "normal", color="white" if dark else INK)
    ax.set_xticks(range(len(datasets)), datasets); ax.set_yticks(range(len(models)), models)
    ax.tick_params(length=0); ax.xaxis.set_ticks_position("top")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(datasets)), minor=True); ax.set_yticks(np.arange(-0.5, len(models)), minor=True)
    ax.grid(which="minor", color="white", lw=1.2); ax.tick_params(which="minor", length=0)
    ent = [(h_patch(cm(norm(1.0))), "rank 1 in dataset"), (h_patch(cm(norm(0.0))), f"rank {v.shape[0]}"),
           (h_patch("none"), "bold = best (ties included)")]
    key(ax, ent, loc=key_loc, bbox_to_anchor=(0, -0.02))
    return ax


def pareto(ax, cost, perf, names, highlight=None, logx=True, higher_is_better=True, label=True, key_loc="top"):
    """Performance against cost. Pareto-optimal models in ink joined by the frontier (step line);
    dominated models in grey; the highlighted model as a coloured diamond."""
    cost = np.asarray(cost, float); perf = np.asarray(perf, float)
    idx = np.argsort(cost); best = -np.inf if higher_is_better else np.inf; front = []
    for i in idx:
        if (perf[i] > best) if higher_is_better else (perf[i] < best):
            best = perf[i]; front.append(i)
    ax.step(cost[front], perf[front], where="post", color=FAINT, lw=0.8, zorder=1)
    hl_set = set(np.atleast_1d(highlight)) if highlight is not None else set()
    for i, nm in enumerate(names):
        hl = nm in hl_set; onf = i in front
        ax.scatter(cost[i], perf[i], s=26 if hl else 14, color=BLUE if hl else (INK if onf else GREY),
                   edgecolor="white", lw=0.5, zorder=3, marker="D" if hl else "o")
        if label:
            ax.annotate(nm, (cost[i], perf[i]), xytext=(3, 2), textcoords="offset points", fontsize=_fs(-1),
                        color=INK if (hl or onf) else SECOND)
    if logx:
        ax.set_xscale("log")
    ax.margins(x=0.12, y=0.1)
    ent = [(h_line(FAINT, lw=0.8), "Pareto frontier"), (h_marker(INK, "o", ms=2.8), "Pareto-optimal"),
           (h_marker(GREY, "o", ms=2.8), "dominated")]
    if hl_set:
        ent.insert(0, (h_marker(BLUE, "D", ms=3.2), "proposed"))
    key(ax, ent, loc=key_loc, ncol=2)
    return front


# ---------------------------------------------------------------- classifier results
def _roc_points(y, s):
    """ROC vertices at distinct thresholds (tied scores share one step, so AUC counts ties as 1/2)."""
    o = np.argsort(-s, kind="mergesort"); y = y[o]; s = s[o]
    last = np.r_[np.nonzero(np.diff(s))[0], y.size - 1]
    tp = np.cumsum(y)[last]; fp = np.cumsum(1 - y)[last]
    return np.r_[0, fp / fp[-1]], np.r_[0, tp / tp[-1]]


def _auc(fpr, tpr):
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


def bootstrap_roc(y, s, n=2000, grid=None, rng=0):
    """ROC with a 95% pointwise band (class-stratified bootstrap; TPR interpolated on a fixed FPR
    grid) and AUC with its 95% percentile bootstrap CI."""
    r = np.random.default_rng(rng); y = np.asarray(y).astype(int); s = np.asarray(s, float)
    grid = np.linspace(0, 1, 201) if grid is None else grid
    pos, neg = np.nonzero(y == 1)[0], np.nonzero(y == 0)[0]
    f0, t0 = _roc_points(y, s)
    ts, aucs = [], []
    for _ in range(n):
        i = np.r_[r.choice(pos, pos.size), r.choice(neg, neg.size)]
        f, t = _roc_points(y[i], s[i]); ts.append(np.interp(grid, f, t)); aucs.append(_auc(f, t))
    ts = np.array(ts)
    return dict(fpr_pts=f0, tpr_pts=t0, fpr=grid, lo=np.quantile(ts, 0.025, 0), hi=np.quantile(ts, 0.975, 0),
                auc=_auc(f0, t0), auc_lo=np.quantile(aucs, 0.025), auc_hi=np.quantile(aucs, 0.975))


def roc(ax, curves, colors_=None, key_loc="top"):
    """curves: dict name -> bootstrap_roc(...). Line = empirical ROC (straight segments between
    threshold vertices, the trapezoid used for AUC); band = 95% pointwise bootstrap band;
    dashed diagonal = chance. The key gives AUC [95% CI]."""
    cs = colors_ or colors(len(curves)); ent = []
    ax.plot([0, 1], [0, 1], color=AXIS, lw=0.5, ls=(0, (2, 2)), zorder=0)
    for (nm, c_), c in zip(curves.items(), cs):
        ax.fill_between(c_["fpr"], c_["lo"], c_["hi"], color=c, alpha=0.16, lw=0)
        ax.plot(c_["fpr_pts"], c_["tpr_pts"], color=c, lw=1.0)
        ent.append((h_line(c, lw=1.0), f"{nm}, AUC {fmt_ci(c_['auc'], c_['auc_lo'], c_['auc_hi'])}"))
    ent += [(h_patch(GREY, alpha=0.3), "95% bootstrap band"), (h_line(AXIS, lw=0.5, ls=(0, (2, 2))), "chance")]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.set_anchor("N")
    ax.set_xlabel("1 \u2212 specificity"); ax.set_ylabel("sensitivity")
    key(ax, ent, loc=key_loc, ncol=1)
    return ax


def calibration(ax, y, p, bins=10, color=None, ax_hist=None, key_loc="top"):
    """Reliability diagram: observed event rate (dot) against mean predicted probability in quantile
    bins, with Wilson 95% CIs (lines); dashed diagonal = perfect calibration."""
    c = color or BLUE; y = np.asarray(y, float); p = np.asarray(p, float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, edges.size - 2)
    mp, obs, lo, hi = [], [], [], []
    for b in range(edges.size - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        k, n = y[m].sum(), m.sum(); l, h = wilson(k, n)
        mp.append(p[m].mean()); obs.append(k / n); lo.append(l); hi.append(h)
    ax.plot([0, 1], [0, 1], color=AXIS, lw=0.5, ls=(0, (2, 2)))
    ax.vlines(mp, lo, hi, color=c, lw=0.7)
    ax.plot(mp, obs, "-o", color=c, lw=0.9, ms=2.8, mec="white", mew=0.4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.set_anchor("N")
    ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency")
    ent = [(h_line(c, lw=0.9, marker="o", ms=2.8), f"observed ({len(mp)} quantile bins)"),
           (Line2D([], [], color=c, lw=0, marker="|", ms=5, mew=0.7), "95% CI (Wilson)"),
           (h_line(AXIS, lw=0.5, ls=(0, (2, 2))), "perfect calibration")]
    if ax_hist is not None:
        ax_hist.hist(p[y == 0], bins=30, range=(0, 1), color=GREY, alpha=0.8, lw=0)
        ax_hist.hist(p[y == 1], bins=30, range=(0, 1), color=c, alpha=0.8, lw=0)
        ax_hist.set_yticks([]); despine(ax_hist, left=False); ax_hist.set_xlim(0, 1)
        ent += [(h_patch(GREY, alpha=0.8), "negatives"), (h_patch(c, alpha=0.8), "positives")]
    key(ax, ent, loc=key_loc, ncol=1)
    return ax


def confusion(ax, cm, labels, color=None, key_loc="top"):
    """Row-normalised confusion matrix: shade and % = share of each true class; (n) = count.
    The diagonal (bold) is per-class recall."""
    cm = np.asarray(cm, float); c = color or BLUE
    pct = cm / cm.sum(1, keepdims=True)
    ax.imshow(pct, cmap=seq_cmap(c, light=0.97), vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            dark = pct[i, j] > 0.55
            ax.text(j, i - 0.12, f"{100 * pct[i, j]:.0f}%", ha="center", va="center", fontsize=_fs(-0.5),
                    color="white" if dark else INK, fontweight="bold" if i == j else "normal")
            ax.text(j, i + 0.22, f"({int(cm[i, j])})", ha="center", va="center", fontsize=_fs(-1.2),
                    color="white" if dark else SECOND)
    ax.set_xticks(range(len(labels)), labels); ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted class"); ax.set_ylabel("true class"); ax.tick_params(length=0); ax.set_anchor("N")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(labels)), minor=True); ax.set_yticks(np.arange(-0.5, len(labels)), minor=True)
    ax.grid(which="minor", color="white", lw=1.5); ax.tick_params(which="minor", length=0)
    key(ax, [(h_patch("none"), "% of true class (row); (n) = count\nbold diagonal = recall")], loc=key_loc,
        handlelength=0, handletextpad=0)
    return ax


def decision_curve(ax, y, models, thresholds=None, colors_=None, key_loc="top"):
    """Net benefit = TP/N - FP/N x p_t/(1 - p_t) across threshold probabilities p_t, with the
    treat-all and treat-none strategies as references."""
    y = np.asarray(y, float); th = np.linspace(0.02, 0.6, 59) if thresholds is None else np.asarray(thresholds)
    prev = y.mean(); cs = colors_ or colors(len(models)); N = y.size
    ax.plot(th, prev - (1 - prev) * th / (1 - th), color=GREY, lw=0.8)
    ax.axhline(0, color=AXIS, lw=0.5, ls=(0, (2, 2)))
    ent = []
    for (nm, p), c in zip(models.items(), cs):
        p = np.asarray(p, float)
        nb = [((p >= t) & (y == 1)).sum() / N - ((p >= t) & (y == 0)).sum() / N * t / (1 - t) for t in th]
        ax.plot(th, nb, color=c, lw=1.0); ent.append((h_line(c, lw=1.0), nm))
    ent += [(h_line(GREY, lw=0.8), "treat all"), (h_line(AXIS, lw=0.5, ls=(0, (2, 2))), "treat none")]
    ax.set_ylim(-0.05, max(prev, 0.05) * 1.1)
    ax.set_box_aspect(1); ax.set_anchor("N")
    ax.set_xlabel("threshold probability"); ax.set_ylabel("net benefit")
    key(ax, ent, loc=key_loc, ncol=2)
    return ax


# ---------------------------------------------------------------- relationships and agreement
def scatter_fit(ax, x, y, color=None, stat="pearson", n_boot=2000, point_size=6, key_loc="top", rng=0):
    """Observations (dots), ordinary least-squares line, 95% pairs-bootstrap band for the line,
    and the correlation with its P value in the key."""
    from scipy import stats
    c = color or BLUE; x = np.asarray(x, float); y = np.asarray(y, float)
    ax.scatter(x, y, s=point_size, color=c, alpha=0.6, lw=0, zorder=2)
    gx = np.linspace(x.min(), x.max(), 100); b1, b0 = np.polyfit(x, y, 1)
    r = np.random.default_rng(rng); fits = []
    for _ in range(n_boot):
        i = r.integers(0, x.size, x.size); bb1, bb0 = np.polyfit(x[i], y[i], 1); fits.append(bb0 + bb1 * gx)
    ax.fill_between(gx, *np.quantile(fits, [0.025, 0.975], 0), color=c, alpha=0.18, lw=0, zorder=1)
    ax.plot(gx, b0 + b1 * gx, color=INK, lw=0.8, zorder=3)
    rr, pp = stats.spearmanr(x, y) if stat == "spearman" else stats.pearsonr(x, y)
    sym = "Spearman's \u03c1" if stat == "spearman" else "Pearson's $r$"
    key(ax, [(h_marker(c, "o", ms=2.4), f"observations (n = {x.size})"), (h_line(INK, lw=0.8), "least-squares fit"),
             (h_patch(c, alpha=0.25), "95% CI of fit"),
             (h_patch("none"), f"{sym} = {_minus(f'{rr:.2f}')}, {fmt_p(pp)}")], loc=key_loc, ncol=2)
    return ax


def bland_altman(ax, a, b, color=None, units="", names=("A", "B"), key_loc="top"):
    """Agreement between two methods: difference (A - B) against their mean; solid line = bias
    (mean difference); dashed lines = 95% limits of agreement, bias +/- 1.96 SD. Values of the
    three lines are printed right of the axes."""
    c = color or BLUE; a = np.asarray(a, float); b = np.asarray(b, float)
    m = (a + b) / 2; d = a - b; bias = d.mean(); sd = d.std(ddof=1); lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
    ax.scatter(m, d, s=6, color=c, alpha=0.6, lw=0)
    ax.axhline(bias, color=INK, lw=0.7)
    for v in (lo, hi):
        ax.axhline(v, color=AXIS, lw=0.5, ls=(0, (3, 2)))
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    for v in (bias, lo, hi):
        ax.text(1.01, v, _minus(f"{v:.2f}"), transform=tr, va="center", ha="left", fontsize=_fs(-1), color=SECOND)
    ax.set_xlabel(f"mean of {names[0]} and {names[1]}{units}"); ax.set_ylabel(f"{names[0]} \u2212 {names[1]}{units}")
    key(ax, [(h_marker(c, "o", ms=2.4), f"pairs (n = {a.size})"), (h_line(INK, lw=0.7), "bias"),
             (h_line(AXIS, lw=0.5, ls=(0, (3, 2))), "95% limits of agreement")], loc=key_loc, ncol=2)
    return ax


def line_ci(ax, x, series, colors_=None, direct_labels=True, markers=True, band_label="95% CI", key_loc="top"):
    """series: dict name -> (y, lo, hi). Lines with markers at measured points, shaded bands = CI.
    With direct_labels, names sit at the line ends (spread so they never overlap) and the key
    explains only the line and band."""
    cs = colors_ or colors(len(series)); ends = []
    for (nm, (yv, lo, hi)), c in zip(series.items(), cs):
        if lo is not None:
            ax.fill_between(x, lo, hi, color=c, alpha=0.15, lw=0)
        ax.plot(x, yv, color=c, lw=1.0, marker="o" if markers else None, ms=2.6, mec="white", mew=0.4)
        ends.append(np.asarray(yv)[-1])
    if direct_labels:
        from figtools import spread_labels
        spread_labels(ax, ends, list(series), cs, x=np.asarray(x)[-1], dx_pts=4, fontsize=_fs(-0.5))
        ent = [(h_line(GREY, lw=1.0, marker="o" if markers else None, ms=2.6), "mean"),
               (h_patch(GREY, alpha=0.3), band_label)]
    else:
        ent = [(h_line(c, lw=1.0), nm) for c, nm in zip(cs, series)] + [(h_patch(GREY, alpha=0.3), band_label)]
    key(ax, ent, loc=key_loc)
    return ax


def heatmap(ax, M, rows, cols, cmap=None, center=None, vmin=None, vmax=None, annotate=False, fmt="{:.2f}",
            cbar_label=None):
    """Heatmap with a thin colour bar (the bar is its key). center=0 gives a diverging map around zero."""
    M = np.asarray(M, float)
    if center is not None:
        lim = np.nanmax(np.abs(M - center)); vmin, vmax = center - lim, center + lim; cmap = cmap or div_cmap()
    im = ax.imshow(M, cmap=cmap or seq_cmap(BLUE), vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)), cols); ax.set_yticks(range(len(rows)), rows); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    if annotate:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = im.norm(M[i, j]); dark = abs(v - 0.5) > 0.32 if center is not None else v > 0.6
                ax.text(j, i, _minus(fmt.format(M[i, j])), ha="center", va="center", fontsize=_fs(-1),
                        color="white" if dark else INK)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.05, pad=0.03, aspect=22)
    cb.outline.set_visible(False); cb.ax.tick_params(length=1.5, width=0.4)
    if cbar_label:
        cb.set_label(cbar_label)
    return im, cb


def sig_bracket(ax, x1, x2, y, text, h=0.02):
    """Thin bracket with exact text (e.g. fmt_p(p)); for the one comparison the panel is about."""
    yr = ax.get_ylim(); hh = h * (yr[1] - yr[0])
    ax.plot([x1, x1, x2, x2], [y, y + hh, y + hh, y], color=INK, lw=0.5)
    ax.text((x1 + x2) / 2, y + hh * 1.3, text, ha="center", va="bottom", fontsize=_fs(-0.5))
