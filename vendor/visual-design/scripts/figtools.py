r"""Small helpers for clean matplotlib figures: collision-free direct labels and a text-overlap check.

    import sys; sys.path.insert(0, r"C:\Users\user\.claude\skills\visual-design\scripts")
    from figtools import spread_labels, find_text_overlaps
"""
import itertools
import json
import os

_PALETTES = os.path.join(os.path.dirname(__file__), "..", "assets", "palettes.json")


def palette(name):
    """Return one palette set from assets/palettes.json as a dict with keys
    data (validated series colours, in slot order), soft, tint, ink, names, use, scatter_safe."""
    with open(_PALETTES, encoding="utf-8") as f:
        lib = json.load(f)
    if name not in lib or name.startswith("_"):
        raise KeyError(f"unknown palette {name!r}; choose from {[k for k in lib if not k.startswith('_')]}")
    return lib[name]


def use_palette(name):
    """Set matplotlib's colour cycle to a palette's data colours and return the palette dict."""
    import matplotlib as mpl
    p = palette(name)
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=p["data"])
    return p


def spread_labels(ax, ys, labels, colors, x, min_gap_frac=0.055, dx_pts=0, **text_kw):
    """Place end-of-line labels at data x (shifted right by dx_pts points), nudging them apart
    vertically so none overlap.

    min_gap_frac is the minimum spacing as a fraction of the y-axis span. Set the final
    y-limits before calling.
    """
    from matplotlib.transforms import offset_copy
    if dx_pts:
        text_kw.setdefault("transform", offset_copy(ax.transData, fig=ax.figure, x=dx_pts, y=0, units="points"))
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    pos = [ys[i] for i in order]
    for k in range(1, len(pos)):
        pos[k] = max(pos[k], pos[k - 1] + gap)
    overflow = pos[-1] - hi if pos else 0
    if overflow > 0:
        pos = [p - overflow for p in pos]
        for k in range(len(pos) - 2, -1, -1):
            pos[k] = min(pos[k], pos[k + 1] - gap)
    kw = dict(va="center", ha="left", fontsize=6.5, clip_on=False)
    kw.update(text_kw)
    for k, i in enumerate(order):
        ax.text(x, pos[k], labels[i], color=colors[i], **kw)


def find_text_overlaps(fig, pad_px=1.0):
    """Return pairs of visible text objects whose boxes overlap (ticks, titles, labels, legends, annotations).

    Call once the layout is final. An empty list means no text collides with other text.
    It does not check text against data marks; look at the rendered figure for that.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    hidden = set()  # tick labels whose tick lies outside the view limits are never drawn
    for ax in fig.axes:
        for axis, (lo, hi) in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = min(lo, hi), max(lo, hi)
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                if not (lo - 1e-9 <= tk.get_loc() <= hi + 1e-9):
                    hidden.update({id(tk.label1), id(tk.label2)})
    seen, texts = set(), []
    for t in fig.findobj(match=lambda o: hasattr(o, "get_text") and hasattr(o, "get_window_extent")):
        if id(t) in seen or id(t) in hidden or not t.get_visible() or not str(t.get_text()).strip():
            continue
        seen.add(id(t)); texts.append(t)
    boxes = []
    for t in texts:
        try:
            boxes.append((t, t.get_window_extent(r).padded(-pad_px)))
        except Exception:
            pass
    def same(a, ba, b, bb):  # one label reachable twice through the artist tree, not a collision
        return a.get_text() == b.get_text() and all(abs(u - v) < 0.5 for u, v in zip(ba.bounds, bb.bounds))
    return [(a.get_text(), b.get_text())
            for (a, ba), (b, bb) in itertools.combinations(boxes, 2) if ba.overlaps(bb) and not same(a, ba, b, bb)]
