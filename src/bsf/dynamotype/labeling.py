"""Dynamotype onset/offset labelling from a seizure trace (Saggio et al. 2020 operational criteria).

Decision hierarchy (``docs/method_notes/dynamotype_taxonomy_and_labeling.md``), applied to the onset
(forward) and offset (reverse) of a representative seizure channel:

  ONSET:  DC shift -> SN ; else amplitude grows from ~0 -> SupH ; else ISI decreasing -> SNIC ;
          else -> SN/SubH (grouped, indistinguishable without DC coupling).
  OFFSET: DC shift -> SH ; else amplitude decaying -> SupH ; else ISI increasing -> SH/SNIC
          (separated by log-vs-sqrt scaling) ; else -> FLC.

Spike amplitudes/ISIs and a low-frequency baseline drive the rules. Labelling is *descriptive*
(offline, two-sided filtering allowed); only the causal CSD precursor that conditions on the onset
label (see ``conditioned_csd``) must be forward-only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

from bsf.detect.ews_classic import kendall_tau

ONSET_TYPES = ("SN", "SupH", "SNIC", "SN/SubH")
# Realised offset axis: a clean DC-coupled SH, the grouped AC-only "SH/SNIC" (log-vs-sqrt scaling is
# not separable on the few offset spikes without DC coupling), SupH, and FLC. SNIC is NOT separately
# realised on AC data, so the realised taxonomy is 12 (4 onset x 3 offset) not the full 16.
OFFSET_TYPES = ("SH", "SH/SNIC", "SupH", "FLC")


@dataclass
class DynamotypeLabel:
    onset: str
    offset: str
    dc_shift: bool
    dc_shift_z: float
    onset_amp_ratio: float  # first/steady spike amplitude (small => growth from zero => SupH)
    onset_isi_ktau: float  # <0 => ISI shrinking => SNIC
    offset_amp_ratio: float  # last/steady amplitude (small => decay => SupH offset)
    offset_isi_ktau: float  # >0 => ISI growing => SH/SNIC
    n_spikes: int

    def as_dict(self) -> dict:
        return asdict(self)


def _sos(fs: float, cutoff, btype: str):
    nyq = fs / 2.0
    wn = (cutoff / nyq) if np.isscalar(cutoff) else [c / nyq for c in cutoff]
    return butter(4, wn, btype=btype, output="sos")


def _filt(x: np.ndarray, fs: float, cutoff, btype: str) -> np.ndarray:
    return sosfiltfilt(_sos(fs, cutoff, btype), np.asarray(x, dtype=float))


def select_seizure_channel(data: np.ndarray, onset: int, offset: int) -> int:
    """Pick the channel with the largest ictal/preictal variance ratio (the clearest seizure)."""
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        return 0
    pre = slice(max(0, 2 * onset - offset), onset)
    ict = slice(onset, offset)
    pre_var = data[pre].var(axis=0) + 1e-12
    ict_var = data[ict].var(axis=0)
    return int(np.argmax(ict_var / pre_var))


def detect_spikes(x: np.ndarray, fs: float, hp_hz: float = 5.0, min_isi_s: float = 0.05,
                  k: float = 4.0, baseline: slice | None = None):
    """Detect ictal spikes on the rectified high-passed trace; return (peak_indices, amplitudes)."""
    xf = np.abs(_filt(x, fs, hp_hz, "highpass"))
    ref = xf[baseline] if baseline is not None else xf
    med = np.median(ref)
    mad = np.median(np.abs(ref - med)) + 1e-12
    thr = med + k * 1.4826 * mad
    peaks, props = find_peaks(xf, height=thr, distance=max(1, int(min_isi_s * fs)))
    return peaks, xf[peaks]


def detect_dc_shift(x: np.ndarray, fs: float, onset: int, win_s: float = 2.0,
                    persist_s: float = 1.0, z_thresh: float = 5.0) -> tuple[bool, float]:
    """Detect a DC baseline step at onset: |Δbaseline| > z_thresh × preictal baseline SD, persisting.

    (Saggio's '>5× background variance' is read as a >5σ baseline deflection; DC requires DC-coupled
    data — on AC-coupled recordings this correctly returns False and SN folds into SN/SubH.)
    """
    base = _filt(x, fs, 1.0, "lowpass")  # slow baseline (<1 Hz)
    w = int(win_s * fs)
    p = int(persist_s * fs)
    pre = base[max(0, onset - w) : onset]
    post = base[onset : onset + p]
    if len(pre) < 2 or len(post) < 2:
        return False, 0.0
    sd = np.std(pre) + 1e-12
    z = abs(np.mean(post) - np.mean(pre)) / sd
    return bool(z > z_thresh), float(z)


def _amp_ratio_and_trend(amps: np.ndarray):
    """(first/steady amplitude ratio, Kendall tau of amplitude vs spike index)."""
    if len(amps) < 4:
        return 1.0, 0.0
    steady = np.median(amps[len(amps) // 2 :])
    ratio = float(amps[0] / (steady + 1e-12))
    return ratio, kendall_tau(amps)


def classify_onset(x: np.ndarray, fs: float, onset: int, win_s: float = 8.0):
    """Classify onset dynamotype; return (label, evidence dict)."""
    pre = slice(max(0, onset - int(win_s * fs)), onset)
    spikes, amps = detect_spikes(x, fs, baseline=pre)
    spikes_in = spikes[(spikes >= onset) & (spikes < onset + int(win_s * fs))]
    amps_in = amps[(spikes >= onset) & (spikes < onset + int(win_s * fs))]
    isis = np.diff(spikes_in) / fs

    dc, z = detect_dc_shift(x, fs, onset)
    amp_ratio, amp_ktau = _amp_ratio_and_trend(amps_in)
    isi_ktau = kendall_tau(isis) if len(isis) >= 4 else 0.0
    ev = {"dc_shift": dc, "dc_shift_z": z, "amp_ratio": amp_ratio, "amp_ktau": amp_ktau,
          "isi_ktau": isi_ktau, "n_spikes": int(len(spikes_in))}

    if dc:
        return "SN", ev
    # SupH: amplitude grows from ~0 (strong positive trend); guard against SNIC (shrinking ISI).
    if len(amps_in) >= 5 and amp_ktau > 0.35 and amp_ratio < 0.6 and isi_ktau > -0.2:
        return "SupH", ev
    # SNIC: frequency rises => ISI shrinks.
    if len(isis) >= 5 and isi_ktau < -0.2:
        return "SNIC", ev
    return "SN/SubH", ev


def classify_offset(x: np.ndarray, fs: float, offset: int, win_s: float = 8.0):
    """Classify offset dynamotype; return (label, evidence dict). Offset is inherently retrospective."""
    lo = max(0, offset - int(win_s * fs))
    pre = slice(offset, min(len(x), offset + int(win_s * fs)))
    spikes, amps = detect_spikes(x, fs, baseline=pre)
    sel = (spikes >= lo) & (spikes < offset)
    spikes_in, amps_in = spikes[sel], amps[sel]
    isis = np.diff(spikes_in) / fs

    dc, z = detect_dc_shift(x, fs, offset)  # reuse: step at offset
    amp_ratio = float(amps_in[-1] / (np.median(amps_in[: len(amps_in) // 2]) + 1e-12)) if len(amps_in) >= 4 else 1.0
    amp_ktau = kendall_tau(amps_in) if len(amps_in) >= 5 else 0.0  # <0 => decaying
    isi_ktau = kendall_tau(isis) if len(isis) >= 4 else 0.0
    ev = {"dc_shift": dc, "dc_shift_z": z, "amp_ratio": amp_ratio, "amp_ktau": amp_ktau,
          "isi_ktau": isi_ktau, "n_spikes": int(len(spikes_in))}

    if dc:
        return "SH", ev
    # SupH offset: amplitude decays to ~0 (strong negative trend), ISI not growing.
    if len(amps_in) >= 5 and amp_ktau < -0.35 and isi_ktau < 0.2:
        return "SupH", ev
    # SH/SNIC offset: ISI grows ("slowing"). SH vs SNIC (log vs sqrt scaling) is not separable on the
    # few spikes available without DC coupling, so they are reported GROUPED.
    if len(isis) >= 5 and isi_ktau > 0.2:
        return "SH/SNIC", ev
    return "FLC", ev


def label_seizure(data: np.ndarray, fs: float, onset: int, offset: int) -> DynamotypeLabel:
    """Full dynamotype label for a (single- or multi-channel) seizure given onset/offset samples."""
    data = np.asarray(data, dtype=float)
    x = data if data.ndim == 1 else data[:, select_seizure_channel(data, onset, offset)]
    on_label, on_ev = classify_onset(x, fs, onset)
    off_label, off_ev = classify_offset(x, fs, offset)
    return DynamotypeLabel(
        onset=on_label, offset=off_label,
        dc_shift=on_ev["dc_shift"], dc_shift_z=on_ev["dc_shift_z"],
        onset_amp_ratio=on_ev["amp_ratio"], onset_isi_ktau=on_ev["isi_ktau"],
        offset_amp_ratio=off_ev["amp_ratio"], offset_isi_ktau=off_ev["isi_ktau"],
        n_spikes=on_ev["n_spikes"],
    )
