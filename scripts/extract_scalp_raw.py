#!/usr/bin/env python
"""Extract RAW preictal/interictal clips from the CHB-MIT and Siena scalp EDFs.

The scalp cohorts were previously available only as 18-D per-hour feature vectors, which meant no
raw-waveform model could be evaluated on scalp at all -- the CNN, the braindecode zoo and AURA-Net's
encoder were confined to iEEG. That is a real gap: scalp is the modality most epilepsy centres
actually record, and a claim about architectures that only holds on depth electrodes is a narrow one.

Output matches the iEEG clip cache exactly (``W`` sub-windows, ``y``, ``groups``, plus the analytic
feature arms), so the unified benchmark treats scalp and iEEG identically.

Label convention, stated because it drives everything downstream:
  * PREICTAL  = ``[onset - sph - preictal_min, onset - sph]``. The seizure-prediction horizon ``sph``
    is a real gap, not decoration: a warning arriving one second before onset is clinically useless,
    and omitting it lets a model score on ictal onset leakage.
  * INTERICTAL = at least ``guard_h`` hours from ANY annotated seizure, so post-ictal state does not
    contaminate the negative class.
  * ``groups`` = the seizure a clip belongs to, so grouped CV keeps one seizure's clips in one fold.

-> datasets/scalp_raw_cache/ (via bsf.data.clip_cache, same keys as iEEG)
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

DATA = Path(__file__).resolve().parents[1] / "datasets"
CHB = DATA / "chbmit"
SIENA = DATA / "siena-scalp-eeg"


def chb_seizures(pat_dir: Path) -> dict[str, list[tuple[float, float]]]:
    """Parse a CHB-MIT summary file -> {edf filename: [(onset_s, offset_s), ...]}."""
    summ = next(pat_dir.glob("*summary*.txt"), None)
    if summ is None:
        return {}
    out, cur = {}, None
    for line in summ.read_text(errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("File Name:"):
            cur = line.split(":", 1)[1].strip(); out.setdefault(cur, [])
        elif cur and line.lower().startswith("seizure") and "start time" in line.lower():
            m = re.search(r"(\d+)\s*seconds", line)
            if m:
                out[cur].append([float(m.group(1)), None])
        elif cur and line.lower().startswith("seizure") and "end time" in line.lower():
            m = re.search(r"(\d+)\s*seconds", line)
            if m and out[cur] and out[cur][-1][1] is None:
                out[cur][-1][1] = float(m.group(1))
    return {k: [(a, b if b is not None else a + 60.0) for a, b in v] for k, v in out.items()}


def chb_file_times(pat_dir: Path) -> dict[str, float]:
    """{edf name: absolute start second}, unwrapped so consecutive recordings form one timeline.

    CHB-MIT summaries give only a wall clock ("File Start Time: 11:42:54"), and a patient's recordings
    run consecutively across midnight. Accumulating a day whenever the clock goes backwards turns them
    into a monotonic timeline, which is what the interictal guard needs: chb01_02 ends at 13:42 and
    chb01_03 has a seizure at 14:33, so windows at the end of _02 are 51 minutes pre-ictal. Judged
    per-file they look four hours clear of anything and get labelled interictal.
    """
    summ = next(pat_dir.glob("*summary*.txt"), None)
    if summ is None:
        return {}
    out: dict[str, float] = {}
    cur, day, prev = None, 0.0, None
    for line in summ.read_text(errors="ignore").splitlines():
        s = line.strip()
        if s.startswith("File Name:"):
            cur = s.split(":", 1)[1].strip()
        elif cur and s.lower().startswith("file start time"):
            t = _hms(s)
            if t is None:
                continue
            if prev is not None and t + day < prev:
                day += 86400.0
            out[cur] = t + day
            prev = out[cur]
    return out


def _ictal_windows(raw, fs, spans, dur_s, n_sub, win_s, target_fs, buffer_s,
                   budget_ict, budget_non, seizure_ids=(), t0_abs=0.0, all_spans_abs=()):
    """Ictal vs non-ictal windows for the DETECTION task (T3), from an already-decoded recording.

    Detection is a different question from forecasting and is the task actually deployed in clinics
    today, so it gets its own labels rather than being inferred from the preictal ones. The window is
    ``win_s`` long and steps by half a window ACROSS the annotated span, so a 30 s seizure yields
    several positives instead of one and short seizures are not silently dropped.

    Negatives are separated from ictal spans by ``buffer_s`` -- seconds, not the forecasting arm's
    four HOURS. Reusing the forecasting guard here was a mistake with two costs. It is not the
    question: "is a seizure happening now" is answered no by a post-ictal window, and a detector that
    never sees post-ictal slowing during training is not asked to reject the state it will most often
    be asked to reject in deployment, so the reported number flatters it. And it is not affordable:
    Siena records run 2-3 h, so a four-hour guard leaves no negative window anywhere and dropped 10 of
    its 14 patients outright. The buffer exists only to keep windows off the annotation boundary,
    where the label is genuinely ambiguous.

    ``groups`` is the seizure index for positives and a unique id per negative, so event-level metrics
    (sensitivity per seizure, false alarms per hour) aggregate over seizures rather than windows.

    The budgets are REMAINING counts, passed in by the caller and decremented across recordings. Given
    the per-patient totals instead, every EDF would contribute a fresh quota: chb01 alone produced 917
    windows against a budget of 60+60, which would have made one patient outweigh a dozen others in
    the pooled set and quietly distorted every cross-patient number computed from it.
    """
    from bsf.models.auranet_train import subwindows
    guard = float(buffer_s)
    step = max(1.0, win_s / 2)
    ids = list(seizure_ids) or list(range(len(spans)))
    W, y, g = [], [], []
    for k, (on, off) in enumerate(spans):
        # A seizure shorter than the window yields no start t with t+win_s <= off, so the plain loop
        # drops it silently -- 4 of the 245 annotated seizures are under 8 s. Centre one window on the
        # span instead: it contains the whole event, which is exactly what a deployed detector sees,
        # and losing a seizure entirely is far worse than including a little flanking signal.
        starts = []
        if off - on < win_s:
            c = min(max(0.0, (on + off) / 2 - win_s / 2), max(0.0, dur_s - win_s))
            if c + win_s <= dur_s:
                starts = [c]
        else:
            t = max(0.0, on)
            while t + win_s <= min(off, dur_s):
                starts.append(t)
                t += step
        for t in starts:
            if sum(v == 1 for v in y) >= budget_ict:
                break
            seg = raw[:, int(t * fs):int((t + win_s) * fs)]
            sw = subwindows(seg.T, fs, n=n_sub, win_s=win_s, target_fs=target_fs)
            if sw is not None:
                W.append(sw); y.append(1); g.append(ids[k] if k < len(ids) else k)
    # still evaluated on the PATIENT timeline rather than this file's, so a window near a file
    # boundary is not called non-ictal while a seizure sits just across it in the next recording
    allsp = list(all_spans_abs) or [(t0_abs + a, t0_abs + b) for a, b in spans]

    def far(t_abs):
        return all(abs(t_abs - on) > guard and abs(t_abs - off) > guard for on, off in allsp)

    t, gid = 0.0, 10_000
    while t + win_s <= dur_s and sum(v == 0 for v in y) < budget_non:
        if far(t0_abs + t) and far(t0_abs + t + win_s):
            seg = raw[:, int(t * fs):int((t + win_s) * fs)]
            sw = subwindows(seg.T, fs, n=n_sub, win_s=win_s, target_fs=target_fs)
            if sw is not None:
                W.append(sw); y.append(0); g.append(gid); gid += 1
        t += win_s * 8       # stride out so negatives span the record rather than clustering
    return W, y, g


def _hms(s: str) -> float | None:
    m = re.search(r"(\d{1,2})[.:](\d{2})[.:](\d{2})", s)
    return float(m.group(1)) * 3600 + float(m.group(2)) * 60 + float(m.group(3)) if m else None


def siena_seizures(pat_dir: Path) -> list[dict]:
    """Parse a Siena seizure list -> [{"file", "reg", "on", "off"}, ...], clock times in seconds.

    Siena annotates wall-clock times rather than in-file offsets, and each seizure block NAMES the
    recording it belongs to. Using that name is not optional. PN05's three recordings all span roughly
    06:00-09:00, so matching a seizure to a file by "does this clock time fall inside that file's
    duration" put one seizure into all three -- 7 matches from 3 annotations, which means ictal and
    preictal windows cut out of recordings where nothing happened, and the same samples simultaneously
    serving as interictal negatives elsewhere. Seven of the fourteen patients were affected.

    Two annotation formats exist and only one says "Seizure start". PN01 writes a bare ``Seizure n 1``
    header followed by ``Start time:`` / ``End time:``, and matching only the first form returned zero
    seizures for that patient with no error raised -- the cohort read 45 where Siena documents 47.

    The file name appears BEFORE the seizure blocks in PN01's layout and AFTER the ``Seizure n``
    header in PN05's, so the current file is carried forward and read at the moment a block completes,
    which is correct for both.
    """
    lst = next(pat_dir.glob("Seizures-list*.txt"), None)
    if lst is None:
        return []
    out: list[dict] = []
    cur_file, reg, reg_end, on = None, None, None, None
    for line in lst.read_text(errors="ignore").splitlines():
        low = line.strip().lower()
        if low.startswith("file name"):
            cur_file = line.split(":", 1)[1].strip() if ":" in line else None
        elif low.startswith("registration start"):
            reg = _hms(line)
        elif low.startswith("registration end"):
            reg_end = _hms(line)           # kept: it arbitrates a wrong registration start
        elif low.startswith("seizure start") or low.startswith("start time"):
            on = _hms(line)
        elif low.startswith("seizure end") or low.startswith("end time"):
            off = _hms(line)
            if on is not None and off is not None:
                out.append({"file": cur_file, "reg": reg, "reg_end": reg_end,
                            "on": on, "off": off})
            on = None
    return out


def siena_spans(pat_dir: Path, edf: Path, raw_or_dur) -> list[tuple[float, float]]:
    """Seizure spans for ONE Siena recording, converted from wall clock to file-relative seconds.

    Matching is by the file name the annotation itself declares, never by "does this time fall inside
    that file". The time-range guess put PN05's single 07:55 seizure into all three of its recordings,
    because they all span the same morning hours.

    The time base -- the clock time at sample 0 -- is chosen by the recording's own DURATION rather
    than trusted from either source, because the two sources disagree and each is sometimes wrong.
    PN14-3's annotation declares start 16:17:45 and end 06:57:40, but the file is 11.67 h long, and
    16:17:45 + 11.67 h is 03:57, contradicting its own declared end. The EDF header says 19:17:45,
    and 19:17:45 + 11.67 h is exactly 06:57:40. The annotation's start is a typo, and taking it on
    faith placed that seizure three hours into the wrong part of the record. So both candidates are
    tested against ``start + duration == declared end`` and the consistent one wins.

    Offsets are unwrapped across midnight because Siena records overnight: PN01 runs 19:00 to 08:29,
    so its 07:53 seizure subtracts to -40k seconds and would be silently dropped.
    """
    dur = (raw_or_dur if isinstance(raw_or_dur, (int, float))
           else raw_or_dur.n_times / raw_or_dur.info["sfreq"])
    hdr = None
    if not isinstance(raw_or_dur, (int, float)):
        t0 = raw_or_dur.info.get("meas_date")
        if t0 is not None:
            hdr = t0.hour * 3600 + t0.minute * 60 + t0.second

    def circ(a, b):
        """Absolute difference between two clock times, shortest way round the 24 h dial."""
        d = abs(a - b) % 86400.0
        return min(d, 86400.0 - d)

    out = []
    for s in siena_seizures(pat_dir):
        if s["file"] and Path(s["file"]).name.lower() != edf.name.lower():
            continue
        cands = [c for c in (hdr, s["reg"]) if c is not None]
        if not cands:
            continue
        base = (min(cands, key=lambda b: circ(b + dur, s["reg_end"]))
                if s.get("reg_end") is not None and len(cands) > 1 else cands[0])
        o, f = s["on"] - base, s["off"] - base
        if o < 0:
            o += 86400.0
        if f < o:
            f += 86400.0
        if 0 <= o <= dur:
            out.append((o, min(f, dur)))
    return out


def timeline(cohort: str, pat_dir: Path, edfs: list[Path]) -> dict:
    """One patient's recordings on a single clock, with every seizure placed on it.

    Returns ``{"starts": {edf: abs_start_s}, "spans": [(abs_on, abs_off)...], "index": {(edf, on):
    seizure_id}}``. The interictal guard and the preictal ictal-overlap check are both meaningless
    per-file: recordings run consecutively, so the seizure that disqualifies a window is frequently in
    the neighbouring file. ``index`` gives each seizure one stable id across the whole patient so a
    preictal window straddling two recordings stays a single CV group.

    For Siena the per-file base is taken from the annotation block, which is the same arbitration
    ``siena_spans`` performs; the exact offset matters less here than the ordering, since the guard is
    four hours wide.
    """
    starts: dict[str, float] = {}
    if cohort == "chbmit":
        starts = chb_file_times(pat_dir)
    else:
        for s in siena_seizures(pat_dir):
            if s["file"] and s["reg"] is not None:
                starts.setdefault(Path(s["file"]).name, float(s["reg"]))
        # unwrap: Siena recordings also run overnight, and a later file can start at a smaller clock
        order = sorted(starts, key=lambda k: [e.name for e in edfs].index(k)
                       if k in [e.name for e in edfs] else 1 << 30)
        day, prev = 0.0, None
        for k in order:
            if prev is not None and starts[k] + day < prev:
                day += 86400.0
            starts[k] += day
            prev = starts[k]
    spans, index = [], {}
    for e in edfs:
        t0 = starts.get(e.name)
        if t0 is None:
            continue
        if cohort == "chbmit":
            own = chb_seizures(pat_dir).get(e.name, [])
        else:
            own = [(s["on"] - s["reg"] + (86400.0 if s["on"] < s["reg"] else 0.0),
                    s["off"] - s["reg"] + (86400.0 if s["off"] < s["reg"] else 0.0))
                   for s in siena_seizures(pat_dir)
                   if s["file"] and Path(s["file"]).name == e.name and s["reg"] is not None]
        for a, b in own:
            index[(e.name, round(a, 1))] = len(spans)
            spans.append((t0 + a, t0 + b))
    return {"starts": starts, "spans": spans, "index": index}


def _windows(raw, fs, spans, dur_s, clip_s, n_sub, win_s, target_fs, sph_min, pre_min, guard_h,
             budget_pre, budget_int, t0_abs=0.0, all_spans_abs=(), seizure_ids=()):
    """Slice one recording into preictal / interictal clips under the stated label convention.

    ``spans`` are this file's own (onset, offset) pairs, used to cut preictal clips. ``all_spans_abs``
    are the PATIENT's seizures on a global timeline and ``t0_abs`` is this file's start on it; the
    interictal guard is evaluated there, not within the file. Judged per file, a window at the end of
    one recording can sit forty minutes before a seizure at the start of the next and be labelled
    interictal -- CHB-MIT records consecutive hours, so this is the common case rather than an edge one.

    A preictal window is also rejected if it overlaps ANY seizure or its postictal shadow. Patients
    like chb12 have seizures minutes apart, so the 30-minute preictal window ahead of one seizure can
    contain the previous one; calling that "preictal" trains the model on ictal activity.
    """
    from bsf.models.auranet_train import subwindows
    sph, pre, guard = sph_min * 60.0, pre_min * 60.0, guard_h * 3600.0
    allsp = list(all_spans_abs) or [(t0_abs + a, t0_abs + b) for a, b in spans]
    ids = list(seizure_ids) or list(range(len(spans)))

    def clear_of_ictal(a_abs, b_abs, pad=0.0):
        """True when [a,b] touches no seizure span (optionally padded for postictal state)."""
        return all(b_abs <= on - pad or a_abs >= off + pad for on, off in allsp)

    W, y, g = [], [], []
    for k, (on, _off) in enumerate(spans):
        lo, hi = on - sph - pre, on - sph
        t = lo
        while t + clip_s <= hi and t >= 0 and sum(v == 1 for v in y) < budget_pre:
            if clear_of_ictal(t0_abs + t, t0_abs + t + clip_s, pad=60.0):
                seg = raw[:, int(t * fs):int((t + clip_s) * fs)]
                sw = subwindows(seg.T, fs, n=n_sub, win_s=win_s, target_fs=target_fs)
                if sw is not None:
                    W.append(sw); y.append(1); g.append(ids[k] if k < len(ids) else k)
            t += clip_s

    def far(t_abs):
        return all(abs(t_abs - on) > guard and abs(t_abs - off) > guard for on, off in allsp)

    t, gid = 0.0, 10_000
    while t + clip_s <= dur_s and sum(v == 0 for v in y) < budget_int:
        if far(t0_abs + t) and far(t0_abs + t + clip_s):
            seg = raw[:, int(t * fs):int((t + clip_s) * fs)]
            sw = subwindows(seg.T, fs, n=n_sub, win_s=win_s, target_fs=target_fs)
            if sw is not None:
                W.append(sw); y.append(0); g.append(gid); gid += 1
        t += clip_s * 4          # stride out so negatives span the record rather than clustering
    return W, y, g


def main() -> int:
    import mne

    from bsf.data.clip_cache import CACHE_ROOT, load, save
    from bsf.data.features import rich_features, rich_plus
    from bsf.utils.io import save_json
    from bsf.utils.provenance import new_results_dir
    from bsf.utils.seeds import set_global_seeds

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip-s", type=float, default=60.0)
    ap.add_argument("--n-subwin", type=int, default=6)
    ap.add_argument("--win-s", type=float, default=8.0)
    ap.add_argument("--target-fs", type=float, default=250.0,
                    help="CHB-MIT 256 Hz and Siena 512 Hz are both resampled to 250. Not arbitrary: "
                         "250x8s = 2000 samples is divisible by CBraMod's 200-sample patch, so the "
                         "whole zoo runs on scalp with IDENTICAL input. At 256 Hz (2048) CBraMod "
                         "alone would have to be dropped, and a benchmark that quietly excludes the "
                         "models it cannot fit is not a benchmark.")
    ap.add_argument("--sph-min", type=float, default=5.0, help="seizure-prediction horizon (gap)")
    ap.add_argument("--preictal-min", type=float, default=30.0)
    ap.add_argument("--guard-h", type=float, default=4.0,
                    help="forecasting only: an interictal clip must be this far from any seizure")
    ap.add_argument("--detect-buffer-s", type=float, default=120.0,
                    help="detection only: a non-ictal window need only clear the annotation boundary "
                         "by this much. Not the forecasting guard -- see _ictal_windows.")
    ap.add_argument("--max-per-class", type=int, default=60)
    ap.add_argument("--max-patients", type=int, default=0)
    ap.add_argument("--montage", default="native", choices=["native", "db18"],
                    help="db18: every recording mapped by channel name to the 18-channel bipolar double "
                         "banana (scripts/montage.py), so channel k is the same derivation in every "
                         "patient; cached under its own key. native: file order, truncated (as benchmarked)")
    args = ap.parse_args()
    set_global_seeds(1234)
    out = new_results_dir("scalp_raw_extract", seed=1234, config=vars(args))
    params = dict(n_subwin=args.n_subwin, win_s=args.win_s, target_fs=args.target_fs,
                  max_per_class=args.max_per_class, with_features=True, v=5, cohort="scalp_raw")
    if args.montage != "native":
        params["montage"] = args.montage

    jobs = []
    for p in sorted(CHB.glob("chb*")):
        if p.is_dir():
            jobs.append(("chbmit", p))
    for p in sorted(SIENA.glob("PN*")):
        if p.is_dir():
            jobs.append(("siena", p))
    if args.max_patients:
        jobs = jobs[:args.max_patients]
    print(f"=== raw scalp extraction: {len(jobs)} patients ===", flush=True)

    dparams = {**params, "cohort": "scalp_det", "task": "detection",
               "buffer_s": args.detect_buffer_s, "det_v": 2}

    def _spans(cohort, pdir, edf, r):
        """Annotated (onset, offset) seconds for ONE recording, in file-relative time."""
        return siena_spans(pdir, edf, r) if cohort != "chbmit" else [
            (a, b) for a, b in chb_seizures(pdir).get(edf.name, [])
            if 0 <= a <= r.n_times / r.info["sfreq"]]

    def _pack(W, y, g):
        """Common montage + the analytic arms on the SAME clips, so feature and raw stay comparable."""
        n_ch = min(w.shape[1] for w in W)
        Wa = np.stack([w[:, :n_ch] for w in W]).astype(np.float32)
        Xrich = np.stack([rich_features(Wa[i, 0].T, args.target_fs) for i in range(len(Wa))])
        Xplus = np.stack([rich_plus(Wa[i, 0].T, args.target_fs) for i in range(len(Wa))])
        return {"W": Wa, "y": np.asarray(y, np.int64),
                "groups": np.asarray(g, np.int64)[:len(y)],
                "Xrich": np.nan_to_num(Xrich).astype(np.float32),
                "Xplus": np.nan_to_num(Xplus).astype(np.float32),
                "Xslow": np.nan_to_num(Xrich[:, :2]).astype(np.float32),
                "Xsub": np.nan_to_num(Xplus).astype(np.float32)}, n_ch

    summary = {}
    for cohort, pdir in jobs:
        label = f"{cohort}_{pdir.name}"
        have_f = load("scalp_raw", label, **params) is not None
        have_d = load("scalp_det", label, **dparams) is not None
        if have_f and have_d:
            print(f"  (cache) {label}", flush=True); continue
        try:
            # ONE decode pass feeds BOTH label sets. Forecasting and detection ask different questions
            # of the same recording; decoding the CHB-MIT EDFs twice would cost hours for nothing.
            W, y, g = [], [], []
            Wd, yd, gd = [], [], []
            edfs = sorted(pdir.glob("*.edf"))
            tl = timeline(cohort, pdir, edfs)     # patient-global clock + every seizure on it
            for fi, edf in enumerate(edfs):
                full_f = (sum(v == 1 for v in y) >= args.max_per_class and
                          sum(v == 0 for v in y) >= args.max_per_class) or have_f
                full_d = (sum(v == 1 for v in yd) >= args.max_per_class and
                          sum(v == 0 for v in yd) >= args.max_per_class) or have_d
                if full_f and full_d:
                    break
                r = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")
                fs = r.info["sfreq"]; dur = r.n_times / fs
                spans = _spans(cohort, pdir, edf, r)
                raw = r.get_data()
                if args.montage == "db18":
                    from montage import db18
                    raw = db18(raw, r.ch_names)
                    if raw is None:                  # a derivation is missing in this recording
                        del r
                        continue
                t0_abs = tl["starts"].get(edf.name, 0.0)
                sids = [tl["index"].get((edf.name, round(a, 1)), 10_000 + fi * 100 + k)
                        for k, (a, _b) in enumerate(spans)]
                # group ids are offset by a per-FILE stride large enough that no within-file id can
                # reach the next file's block. The previous form offset by 100*len(W), a running
                # window count, which let one file's interictal ids collide with another's.
                off = 1_000_000 * (fi + 1)
                if not full_f:
                    a, b, c = _windows(raw, fs, spans, dur, args.clip_s, args.n_subwin, args.win_s,
                                       args.target_fs, args.sph_min, args.preictal_min, args.guard_h,
                                       args.max_per_class - sum(v == 1 for v in y),
                                       args.max_per_class - sum(v == 0 for v in y),
                                       t0_abs=t0_abs, all_spans_abs=tl["spans"], seizure_ids=sids)
                    W += a; y += b
                    # a preictal clip keeps its SEIZURE id unshifted, so the same seizure stays one
                    # group even if its preictal window straddles two recordings
                    g += [x if lab == 1 else x + off for x, lab in zip(c, b)]
                if not full_d:
                    a, b, c = _ictal_windows(raw, fs, spans, dur, args.n_subwin, args.win_s,
                                             args.target_fs, args.detect_buffer_s,
                                             args.max_per_class - sum(v == 1 for v in yd),
                                             args.max_per_class - sum(v == 0 for v in yd),
                                             seizure_ids=sids, t0_abs=t0_abs,
                                             all_spans_abs=tl["spans"])
                    Wd += a; yd += b
                    gd += [x if lab == 1 else x + off for x, lab in zip(c, b)]
                del r, raw

            if not have_d:
                yda = np.asarray(yd, np.int64)
                if len(yda) >= 6 and len(np.unique(yda)) == 2 and int(np.bincount(yda).min()) >= 3:
                    rec_d, nch_d = _pack(Wd, yd, gd)
                    save("scalp_det", label, rec_d, **dparams)
                    summary.setdefault(label, {})["detection"] = {
                        "n_windows": int(len(yda)), "n_ictal": int((yda == 1).sum()),
                        "n_channels": int(nch_d),
                        "n_seizures": int(len(set(np.asarray(gd)[:len(yda)][yda == 1])))}
                    print(f"  {label}: detection {len(yda)} windows "
                          f"({int((yda==1).sum())} ictal)", flush=True)
                else:
                    summary.setdefault(label, {})["detection"] = {
                        "skipped": "insufficient ictal windows", "n_ictal": int((yda == 1).sum())}
            if have_f:
                continue

            y = np.asarray(y, np.int64)
            if len(y) < 6 or len(np.unique(y)) < 2 or int(np.bincount(y).min()) < 3:
                print(f"  {label}: too few usable clips (pre={int((y==1).sum())} "
                      f"int={int((y==0).sum())}) -- skipped", flush=True)
                summary.setdefault(label, {})["forecasting"] = {
                    "skipped": "insufficient clips",
                    "n_pre": int((y == 1).sum()), "n_int": int((y == 0).sum())}
                continue
            rec, n_ch = _pack(W, y, g)
            save("scalp_raw", label, rec, **params)
            summary.setdefault(label, {})["forecasting"] = {
                "n_clips": int(len(y)), "n_pre": int((y == 1).sum()), "n_channels": int(n_ch),
                "n_seizures": int(len(set(np.asarray(g)[:len(y)][y == 1])))}
            print(f"  {label}: forecasting {len(y)} clips ({int((y==1).sum())} preictal), "
                  f"{n_ch} ch", flush=True)
        except Exception as e:  # noqa: BLE001
            summary.setdefault(label, {})["error"] = str(e)[:150]
            print(f"  {label}: FAILED {str(e)[:90]}", flush=True)

    n_f = sum(1 for v in summary.values() if "n_clips" in v.get("forecasting", {}))
    n_d = sum(1 for v in summary.values() if "n_windows" in v.get("detection", {}))
    save_json({"params": params, "detection_params": dparams, "per_patient": summary,
               "n_forecasting": n_f, "n_detection": n_d}, out / "results.json")
    print(f"\n  extracted forecasting={n_f} detection={n_d} patients -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
