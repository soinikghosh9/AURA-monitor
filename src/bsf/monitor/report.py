"""A self-contained HTML report for one patient: timeline, seizure catalogue, alarm review, interictal
summary and the validation behind every number. No external scripts; inline SVG; light and dark."""
from __future__ import annotations

import html
import json

import numpy as np

from bsf.monitor import signals as S

CSS = """
:root{--bg:#F6F7F6;--panel:#FFFFFF;--ink:#22282D;--ink2:#5C656B;--muted:#6B7780;--grid:#E8EBEA;--line:rgba(34,40,45,.12);
--accent:#319287;--accent-ink:#23766D;--prob:#835A8C;--thr:#22282D;--ied:#809D62;--risk:#5C656B;--sz:#A85549;
--szfill:rgba(168,85,73,.14);--ok:#5F8A6B;--warn:#C29049;--bad:#A85549;--q:#C8CFD3}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#151A1C;--panel:#1C2225;--ink:#E6E9E8;--ink2:#AAB3B8;
--muted:#8D979D;--grid:#2A3236;--line:rgba(230,233,232,.12);--accent:#6FC2B6;--accent-ink:#6FC2B6;--prob:#B892C2;--thr:#E6E9E8;
--ied:#A9C48C;--risk:#AAB3B8;--sz:#D9877B;--szfill:rgba(217,135,123,.18);--ok:#8DBB98;--warn:#D9AE6B;--bad:#D9877B;--q:#3A4246}}
:root[data-theme=dark]{--bg:#151A1C;--panel:#1C2225;--ink:#E6E9E8;--ink2:#AAB3B8;--muted:#8D979D;--grid:#2A3236;
--line:rgba(230,233,232,.12);--accent:#6FC2B6;--accent-ink:#6FC2B6;--prob:#B892C2;--thr:#E6E9E8;--ied:#A9C48C;--risk:#AAB3B8;
--sz:#D9877B;--szfill:rgba(217,135,123,.18);--ok:#8DBB98;--warn:#D9AE6B;--bad:#D9877B;--q:#3A4246}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 "Geist","IBM Plex Sans","Source Sans 3",system-ui,sans-serif}
.mono,.tile .v,td.num{font-family:"Geist Mono","IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}
main{max-width:1200px;margin:0 auto;padding:28px 20px 72px}
h1{font-size:26px;font-weight:600;letter-spacing:-.03em;margin:0 0 4px;text-wrap:balance}
h2{font-size:17px;font-weight:600;letter-spacing:-.01em;margin:36px 0 12px}h3{font-size:14px;font-weight:600;margin:0 0 6px}
.sub{color:var(--ink2)}.muted{color:var(--muted)}.note{font-size:12.5px;color:var(--muted);max-width:75ch}
.banner{display:flex;gap:10px;align-items:baseline;border:1px solid var(--line);background:var(--panel);border-radius:14px;
padding:10px 14px;margin:16px 0;font-size:13px}
.banner::before{content:"";flex:none;width:8px;height:8px;border-radius:50%;background:var(--warn);transform:translateY(-1px)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px 14px}
.tile .v{font-size:22px;font-weight:500}.tile .k{font-size:12px;color:var(--ink2);margin-top:2px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px;overflow-x:auto}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px}
.chip{display:inline-flex;align-items:center;gap:5px;font-size:12px;padding:1px 8px;border-radius:999px;
border:1px solid var(--line);margin:0 4px 4px 0;color:var(--ink2)}
.chip.ok,.chip.bad,.chip.warn{border-color:transparent;padding-left:0}
.chip.ok::before,.chip.bad::before,.chip.warn::before{content:"";width:7px;height:7px;border-radius:50%;background:currentColor}
.chip.ok{color:var(--ok)}.chip.bad{color:var(--bad)}.chip.warn{color:var(--warn)}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid);vertical-align:top}
th{color:var(--ink2);font-weight:500}ul{margin:4px 0 0 18px;padding:0}svg text{fill:var(--ink2);font-size:11px}
.legend{font-size:12px;color:var(--ink2);margin:4px 0 8px}.legend span{margin-right:14px;white-space:nowrap}
.sw{display:inline-block;width:12px;height:3px;vertical-align:middle;margin-right:5px;border-radius:1px}
details summary{cursor:pointer;color:var(--ink2)}
a{color:var(--accent-ink)}
@media (prefers-reduced-motion:no-preference){details[open] summary~*{animation:fade .3s cubic-bezier(.16,1,.3,1)}}
@keyframes fade{from{opacity:0;transform:translateY(-2px)}to{opacity:1;transform:none}}
"""


def _e(x) -> str:
    return html.escape(str(x))


def _clock(t: float) -> str:
    d, s = divmod(int(t), 86400)
    return f"day {d + 1} {s // 3600:02d}:{(s % 3600) // 60:02d}"


def _path(xs, ys) -> str:
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))


def timeline_svg(R: dict, width: int = 1140) -> str:
    T = R["timeline"]
    t = np.asarray(T["t"], float); t0, t1 = float(t.min()), float(t.max()) + 8
    L, Rm, top = 64, 12, 6
    W = width - L - Rm
    X = lambda v: L + (np.asarray(v, float) - t0) / (t1 - t0) * W  # noqa: E731
    tracks = [("Seizure probability", 120), ("IEDs / 10 min", 70), ("Risk, next hour", 60), ("Signal quality", 14)]
    H = top + sum(h + 26 for _, h in tracks) + 22
    o = [f'<svg viewBox="0 0 {width} {H}" width="100%" role="img" aria-label="Monitoring timeline">']
    y = top
    # seizure bands across all tracks
    for a, b in R.get("annotated_seizures", []):
        o.append(f'<rect x="{X(a):.1f}" y="{top}" width="{max(3.0, X(b) - X(a)):.1f}" height="{H - top - 22}" fill="var(--szfill)"/>')
        o.append(f'<path d="M{X(a) - 5:.1f},{H - 24} L{X(a) + 5:.1f},{H - 24} L{X(a):.1f},{H - 32} Z" fill="var(--sz)">'
                 f'<title>annotated seizure {_e(_clock(a))}, {b - a:.0f} s</title></path>')
    for name, h in tracks:
        o.append(f'<text x="{L}" y="{y + 11}" style="font-weight:600">{_e(name)}</text>')
        y0 = y + 16
        o.append(f'<line x1="{L}" x2="{L + W}" y1="{y0 + h}" y2="{y0 + h}" stroke="var(--line)"/>')
        if name.startswith("Seizure"):
            p = np.asarray(T["prob"], float)
            k = max(1, len(p) // 2400)
            xs, ys = X(t[::k]), y0 + h - p[::k] * h
            o.append(f'<path d="{_path(xs, ys)}" fill="none" stroke="var(--prob)" stroke-width="1.2"/>')
            thr = R["detection"]["threshold"]
            o.append(f'<line x1="{L}" x2="{L + W}" y1="{y0 + h - thr * h:.1f}" y2="{y0 + h - thr * h:.1f}" stroke="var(--thr)" stroke-dasharray="4 3"/>')
            for v in (0, 0.5, 1):
                o.append(f'<text x="{L - 6}" y="{y0 + h - v * h + 4:.1f}" text-anchor="end">{v:g}</text>')
            for ev in R["alarms"]:
                c = "var(--sz)" if ev.get("confirmed") else "var(--muted)"
                fill = c if ev.get("confirmed") else "none"
                o.append(f'<circle cx="{X(ev["t_on"]):.1f}" cy="{y0 + 6}" r="4.5" fill="{fill}" stroke="{c}" stroke-width="1.6">'
                         f'<title>{_e(_clock(ev["t_on"]))}: {"confirmed" if ev.get("confirmed") else "not confirmed"}</title></circle>')
        elif name.startswith("IED"):
            tb = np.asarray(T["ied_bins_t"], float); n = np.asarray(T["ied_bins_n"], float)
            mx = max(1.0, float(n.max()))
            bw = max(1.0, X(t0 + 600) - X(t0) - 1)
            for a, v in zip(tb, n):
                if v > 0:
                    o.append(f'<rect x="{X(a):.1f}" y="{y0 + h - v / mx * h:.1f}" width="{bw:.1f}" height="{v / mx * h:.1f}" rx="1" fill="var(--ied)">'
                             f'<title>{_e(_clock(a))}: {int(v)} IEDs</title></rect>')
            o.append(f'<text x="{L - 6}" y="{y0 + 4}" text-anchor="end">{int(mx)}</text>')
        elif name.startswith("Risk"):
            tb = np.asarray(T["risk_t"], float); r = np.asarray(T["risk_p"], float)
            mx = max(0.05, float(r.max()) * 1.1)
            if len(tb):
                o.append(f'<path d="{_path(X(tb), y0 + h - r / mx * h)}" fill="none" stroke="var(--risk)" stroke-width="1.6"/>')
            o.append(f'<text x="{L - 6}" y="{y0 + 4}" text-anchor="end">{mx:.2f}</text>')
        else:
            q = np.asarray(T["quality"], float)
            k = max(1, len(q) // 1200)
            for a, v in zip(t[::k], q[::k]):
                if v < 0.8:
                    o.append(f'<rect x="{X(a):.1f}" y="{y0}" width="{max(1.0, X(a + 8 * k) - X(a)):.1f}" height="{h}" fill="var(--warn)" opacity="{min(1, 1.2 - v):.2f}"/>')
        y = y0 + h + 10
    # hour ticks
    span_h = (t1 - t0) / 3600
    step = 1 if span_h <= 12 else 2 if span_h <= 24 else 4 if span_h <= 48 else 12
    first = np.ceil(t0 / 3600 / step) * step * 3600
    for tk in np.arange(first, t1, step * 3600):
        o.append(f'<line x1="{X(tk):.1f}" x2="{X(tk):.1f}" y1="{top}" y2="{H - 20}" stroke="var(--grid)" stroke-width="0.6"/>')
        o.append(f'<text x="{X(tk):.1f}" y="{H - 6}" text-anchor="middle">{_e(_clock(tk))}</text>')
    o.append("</svg>")
    return "".join(o)


HEAD_POS = {"FP1": (-.31, .95), "FP2": (.31, .95), "F7": (-.81, .59), "F3": (-.42, .55), "FZ": (0, .5),
            "F4": (.42, .55), "F8": (.81, .59), "T7": (-.93, 0), "C3": (-.47, 0), "CZ": (0, 0), "C4": (.47, 0),
            "T8": (.93, 0), "P7": (-.81, -.59), "P3": (-.42, -.55), "PZ": (0, -.5), "P4": (.42, -.55),
            "P8": (.81, -.59), "O1": (-.31, -.95), "O2": (.31, -.95)}


def channel_map_svg(cmap, size: int = 150, title: str = "") -> str:
    """10-20 head (nose up, left on the left); each double-banana derivation is a link between its two
    electrodes, drawn darker and thicker the stronger its evidence (hover for the value)."""
    v = np.clip(np.asarray(cmap, float), 0, None)
    mx = max(3.0, float(v.max()))
    r = size * 0.40; cx = size / 2; cy = size * 0.50
    P = lambda e: (cx + HEAD_POS[e][0] * r, cy - HEAD_POS[e][1] * r)  # noqa: E731
    o = [f'<svg viewBox="0 0 {size} {size}" width="{size}" role="img" aria-label="{_e(title or "channel map")}">',
         f'<circle cx="{cx}" cy="{cy}" r="{r:.1f}" fill="none" stroke="var(--ink2)" stroke-width="0.8"/>',
         f'<path d="M{cx - 0.1 * r:.1f},{cy - 0.995 * r:.1f} L{cx:.1f},{cy - 1.13 * r:.1f} L{cx + 0.1 * r:.1f},{cy - 0.995 * r:.1f}" '
         f'fill="none" stroke="var(--ink2)" stroke-width="0.8"/>']
    for i in np.argsort(v):
        (x1, y1), (x2, y2) = P(S.DB18[i][0]), P(S.DB18[i][1])
        a = v[i] / mx
        o.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="var(--sz)" '
                 f'stroke-opacity="{0.15 + 0.85 * a:.2f}" stroke-width="{1.2 + 4.5 * a:.1f}" stroke-linecap="round">'
                 f'<title>{_e(S.DB18_NAMES[i])}: {v[i]:.1f}</title></line>')
    for e in HEAD_POS:
        x, y = P(e)
        o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{size * 0.022:.1f}" fill="var(--panel)" stroke="var(--ink2)" stroke-width="0.6"/>')
    o.append(f'<text x="{cx - r - 2:.1f}" y="{size - 4}" text-anchor="start">L</text>'
             f'<text x="{cx + r + 2:.1f}" y="{size - 4}" text-anchor="end">R</text></svg>')
    return "".join(o)


def eeg_svg(snippet: dict, width: int = 520, height: int = 300) -> str:
    """18 traces of the double banana around onset (bipolar, per-channel scaled), onset marked."""
    x = np.asarray(snippet["x"], float); t = np.asarray(snippet["t_rel"], float)
    n = x.shape[0]; L = 54
    W = width - L - 6; rowh = (height - 18) / n
    X = lambda v: L + (v - t[0]) / (t[-1] - t[0]) * W  # noqa: E731
    o = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="EEG around onset">']
    sc = np.median(np.abs(x - np.median(x, 1, keepdims=True)), 1) * 1.4826 + 1e-9
    for i in range(n):
        yc = 6 + rowh * (i + 0.5)
        ys = yc - np.clip(x[i] / sc[i], -6, 6) * rowh / 8
        o.append(f'<text x="{L - 4}" y="{yc + 3:.1f}" text-anchor="end" style="font-size:8.5px">{_e(S.DB18_NAMES[i])}</text>')
        o.append(f'<path d="{_path(X(t), ys)}" fill="none" stroke="var(--ink2)" stroke-width="0.6"/>')
    o.append(f'<line x1="{X(0):.1f}" x2="{X(0):.1f}" y1="0" y2="{height - 14}" stroke="var(--sz)" stroke-dasharray="3 2"/>')
    for v in range(int(np.ceil(t[0] / 10)) * 10, int(t[-1]) + 1, 10):
        o.append(f'<text x="{X(v):.1f}" y="{height - 2}" text-anchor="middle">{v:+d} s</text>')
    o.append("</svg>")
    return "".join(o)


def spread_svg(sc: dict, width: int = 160, height: int = 44) -> str:
    t = np.asarray(sc["t"], float); n = np.asarray(sc["n_channels"], float)
    if len(t) < 2:
        return ""
    X = lambda v: 4 + (v - t[0]) / (t[-1] - t[0]) * (width - 8)  # noqa: E731
    Y = lambda v: height - 4 - v / 18 * (height - 10)  # noqa: E731
    return (f'<svg viewBox="0 0 {width} {height}" width="{width}" role="img" aria-label="spread"><line x1="{X(0):.1f}" x2="{X(0):.1f}" y1="2" '
            f'y2="{height - 2}" stroke="var(--sz)" stroke-dasharray="2 2"/><path d="{_path(X(t), Y(n))}" fill="none" stroke="var(--prob)" stroke-width="1.5"/></svg>')


def render(R: dict) -> str:
    v = R["validation"]
    out = [f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
           f"<title>AURA Monitor report</title><style>{CSS}</style></head><body><main>"]
    out.append(f"<h1>AURA Monitor: {_e(R['patient'])}</h1><div class=sub>{_e(R['cohort_label'])} &middot; {R['hours']:.1f} h monitored "
               f"({_e(_clock(R['t_start']))} to {_e(_clock(R['t_end']))}) &middot; 18-channel double banana &middot; false-alarm budget "
               f"{R['detection']['budget_per_24h']:g} per 24 h</div>")
    out.append("<div class=banner><b>Research software, not a medical device.</b> Every output is a prompt to review the EEG, "
               "not a diagnosis. Validation figures for each module are listed at the end of this report.</div>")
    s = R["summary"]
    tiles = [("Alarms raised", s["alarms"]), ("Confirmed on review", s["confirmed"]), ("Annotated seizures", s["annotated"]),
             ("Seizures detected", f"{s['detected']}/{s['annotated']}"), ("False alarms / 24 h", s["fa_per_24h"]),
             ("IEDs per hour (median)", s["ied_per_h"]), ("Predominant IED side", s["ied_side"]), ("Seizure types", s["n_types"]),
             ("Risk next hour (now)", s["risk_now"])]
    out.append("<div class=tiles>" + "".join(f"<div class=tile><div class=v>{_e(v_)}</div><div class=k>{_e(k)}</div></div>"
                                              for k, v_ in tiles) + "</div>")
    out.append("<h2>Timeline</h2><div class=panel>")
    out.append("<div class=legend><span><i class=sw style='background:var(--prob)'></i>seizure probability</span>"
               "<span><i class=sw style='background:var(--thr)'></i>alarm threshold</span><span>&#9679; confirmed alarm</span>"
               "<span>&#9675; alarm not confirmed</span><span><i class=sw style='background:var(--szfill);height:10px'></i>annotated seizure</span>"
               "<span><i class=sw style='background:var(--ied);height:8px'></i>IEDs</span><span><i class=sw style='background:var(--risk)'></i>"
               "risk, next hour</span><span><i class=sw style='background:var(--warn);height:8px'></i>reduced signal quality</span></div>")
    out.append(timeline_svg(R) + "</div>")
    out.append("<p class=note>Risk is a calibrated background likelihood from seizure history and the patient's own rate; in validation "
               f"it improved calibration (Brier skill {v['risk']['brier_skill']:+.2f}) but did not time seizures better than chance "
               f"(sensitivity {v['risk']['sens']:.2f} at 20% time in warning, chance 0.20). Do not use it as a forecast.</p>")

    out.append("<h2>Seizures</h2><div class=cards>")
    for sz in R["seizures"]:
        fp = sz["fingerprint"]
        st = ("<span class='chip ok'>detected</span>" if sz.get("detected") else "<span class='chip bad'>missed</span>") if sz["source"] == "annotated" else "<span class='chip warn'>detected, not annotated</span>"
        sim = "".join(f"<li>{_e(x['when'])}: similarity {x['similarity']:.2f}</li>" for x in sz.get("similar", [])[:2]) or "<li class=muted>first of its kind</li>"
        feats = "".join(f"<li>{_e(f['plain'])}: {f['z']:+.1f} SD</li>" for f in sz.get("features", [])[:4])
        out.append(f"<div class=card><h3>{_e(_clock(sz['t_on']))} &middot; {sz['duration_s']:.0f} s &middot; type {chr(65 + sz['type'])}</h3>{st}"
                   f"<span class='chip mono'>onset {_e(', '.join(fp['onset_channels']))}</span><span class=chip>{_e(fp['lateralization']['side'])}</span>"
                   f"<div style='display:flex;gap:14px;align-items:flex-start;margin-top:8px;flex-wrap:wrap'>"
                   f"<div>{channel_map_svg(fp['onset_map'], 150, 'onset map')}<div class=note>onset map (first 16 s)</div></div>"
                   f"<div><div class=note>channels involved over time</div>{spread_svg(fp['spread_curve'])}"
                   f"<div class=note>{('half the channels involved by %.0f s' % fp['time_to_half_s']) if fp['time_to_half_s'] is not None else 'did not reach half of the channels'}</div>"
                   f"<div class=note style='margin-top:6px'>largest changes</div><ul>{feats}</ul><div class=note>most similar earlier seizures</div><ul>{sim}</ul></div></div>")
        if sz.get("eeg"):
            out.append(f"<details><summary>EEG at onset</summary>{eeg_svg(sz['eeg'])}</details>")
        out.append("</div>")
    out.append("</div>" if R["seizures"] else "<p class=muted>No seizures in this period.</p></div>")

    out.append("<h2>Alarm review</h2><div class=panel><table><tr><th>When</th><th>Duration</th><th>Probability</th><th>Agreement</th>"
               "<th>Review</th><th>Why the monitor alarmed</th></tr>")
    for a in R["alarms"]:
        rv = "<span class='chip ok'>confirmed</span>" if a.get("confirmed") else "<span class='chip bad'>rejected</span>"
        out.append(f"<tr><td>{_e(_clock(a['t_on']))}</td><td>{a['duration_s']:.0f} s</td><td>{a['probability']:.2f}</td>"
                   f"<td>{a['agreement']['members_above_half']}/{a['agreement']['n_members']} ({_e(a['uncertainty'])} uncertainty)</td><td>{rv}</td>"
                   f"<td><ul>{''.join(f'<li>{_e(x)}</li>' for x in a['summary'])}</ul></td></tr>")
    if not R["alarms"]:
        out.append("<tr><td colspan=6 class=muted>No alarms at this false-alarm budget.</td></tr>")
    out.append("</table></div>")
    if R.get("personalization"):
        p = R["personalization"]
        out.append(f"<p class=note>Adaptation: {p['n_confirmed']} confirmed seizures trained the patient head; threshold moved from "
                   f"{p['threshold_start']:.2f} to {p['threshold_end']:.2f} after {p['n_rejected']} rejected alarms. "
                   f"Replayed on this record: {p['detected_after']}/{p['n_after']} later seizures detected with "
                   f"{p['fa_after']:.1f} false alarms per 24 h, against {p['detected_before']}/{p['n_after']} and {p['fa_before']:.1f} without adaptation.</p>")

    i = R["ied"]
    out.append("<h2>Interictal activity</h2><div class=cards><div class=card><h3>Discharges</h3><table>"
               f"<tr><th>Detected IEDs</th><td>{i['n']}</td></tr><tr><th>Per hour, median (range)</th><td>{i['per_h_median']} ({i['per_h_min']}-{i['per_h_max']})</td></tr>"
               f"<tr><th>Generalized / focal</th><td>{i['generalized']} / {i['focal']}</td></tr>"
               + "<tr><th colspan=2 class=muted style='font-weight:400'>focal discharges by region of strongest evidence (descriptive)</th></tr>"
               + "".join(f"<tr><th>{_e(k.replace('_', '-'))}</th><td>{n}</td></tr>" for k, n in i["lobes"].items())
               + f"<tr><th>Side (focal)</th><td>left {i['side']['left']}, right {i['side']['right']}, bilateral {i['side']['bilateral']}</td></tr></table>"
               f"<p class=note>IED detector trained on a different cohort (VEPISET, children, routine EEG); counts on this record are not validated against annotation.</p></div>"
               f"<div class=card><h3>Where the discharges are</h3>{channel_map_svg(i['map'], 190, 'IED map')}<div class=note>mean discharge evidence by derivation</div></div></div>")

    out.append("<h2>How far to trust this</h2><div class=panel><table><tr><th>Module</th><th>Validation (development patients, out of patient)</th></tr>")
    for k, txt in v["rows"]:
        out.append(f"<tr><td>{_e(k)}</td><td>{_e(txt)}</td></tr>")
    out.append("</table></div><p class=note>Generated by AURA Monitor " + _e(R.get("version", "")) + ". Data: " + _e(R["data_note"]) + "</p>")
    out.append("<script type=application/json id=report-data>" + json.dumps({k: R[k] for k in ("patient", "summary")}).replace("</", "<\\/") + "</script>")
    out.append("</main></body></html>")
    return "".join(out)


if __name__ == "__main__":                      # re-render a saved report: python -m bsf.monitor.report <report.json>
    import sys
    from pathlib import Path
    src = Path(sys.argv[1])
    (src.parent / "report.html").write_text(render(json.loads(src.read_text(encoding="utf-8"))), encoding="utf-8")
    print(src.parent / "report.html")
