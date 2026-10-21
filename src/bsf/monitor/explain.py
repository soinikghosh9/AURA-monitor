"""Evidence cards: why the monitor raised an alarm, in terms a clinician can check on the EEG."""
from __future__ import annotations

import numpy as np

from bsf.data.features import FEATURE_NAMES_PLUS
from bsf.monitor import ied as I
from bsf.monitor import signals as S

PLAIN = {
    "delta_pow_mean": "slow-wave (1-4 Hz) power", "theta_pow_mean": "theta (4-8 Hz) power",
    "alpha_pow_mean": "alpha (8-13 Hz) power", "beta_pow_mean": "beta (13-30 Hz) power",
    "gamma_pow_mean": "gamma (30-70 Hz) power", "hfo_pow_mean": "high-frequency (>70 Hz) power",
    "delta_pow_spread": "spatial unevenness of slow-wave power", "theta_pow_spread": "spatial unevenness of theta power",
    "alpha_pow_spread": "spatial unevenness of alpha power", "beta_pow_spread": "spatial unevenness of beta power",
    "gamma_pow_spread": "spatial unevenness of gamma power", "hfo_pow_spread": "spatial unevenness of high-frequency power",
    "spectral_edge": "spectral edge frequency", "line_length": "signal complexity (line length)",
    "synchrony": "inter-channel synchrony", "spike_rate": "sharp-transient (spike) rate",
    "kurtosis": "peakiness of the signal", "theta_gamma_pac": "theta-gamma coupling"}


def top_features(Fz: np.ndarray, k: int = 5) -> list[dict]:
    """Fz (n_windows, 18) refreshed z of the alarm's windows -> the k largest median departures."""
    med = np.median(np.asarray(Fz, float), 0)
    idx = np.argsort(np.abs(med))[::-1][:k]
    return [{"feature": FEATURE_NAMES_PLUS[i], "plain": PLAIN.get(FEATURE_NAMES_PLUS[i], FEATURE_NAMES_PLUS[i]),
             "z": round(float(med[i]), 2), "direction": "up" if med[i] > 0 else "down"} for i in idx]


def alarm_card(t_on: float, t_off: float, prob: float, members: dict[str, float], Fz_alarm: np.ndarray,
               Zch_alarm: np.ndarray | None, quality: float, flagged: list[str], ied_recent: float,
               ied_typical: float, similar: list[dict], threshold: float) -> dict:
    """Assemble one card. ``members``: each ensemble member's mean probability over the alarm."""
    card = {"t_on": t_on, "t_off": t_off, "duration_s": round(t_off - t_on, 1), "probability": round(prob, 3),
            "threshold": round(threshold, 3), "features": top_features(Fz_alarm)}
    vals = np.array(list(members.values()), float)
    card["agreement"] = {"members_above_half": int((vals >= 0.5).sum()), "n_members": int(len(vals)),
                         "spread": round(float(vals.std()), 3),
                         "members": {k: round(float(v), 3) for k, v in members.items()}}
    card["uncertainty"] = ("low" if card["agreement"]["members_above_half"] >= 0.75 * len(vals) else
                           "high" if card["agreement"]["members_above_half"] < 0.5 * len(vals) else "moderate")
    if Zch_alarm is not None:
        from bsf.monitor.patterns import evidence
        cmap = evidence(Zch_alarm).mean(0)
        card["channel_map"] = cmap.round(2).tolist()
        card["channels"] = [S.DB18_NAMES[i] for i in np.argsort(cmap)[::-1][:3]]
        card["side"] = I.lateralize(cmap)
        card["regions"] = I.region_scores(cmap)
    card["signal_quality"] = {"score": round(quality, 2), "flagged_channels": flagged,
                              "caution": quality < 0.8}
    card["ied_context"] = {"last_30_min": round(ied_recent, 1), "typical_30_min": round(ied_typical, 1),
                           "elevated": ied_recent > 2 * max(ied_typical, 1.0)}
    card["similar_seizures"] = similar[:3]
    reasons = [f"{f['plain']} {'increased' if f['direction'] == 'up' else 'decreased'} ({f['z']:+.1f} SD from this patient's last 30 min)"
               for f in card["features"][:3]]
    if "channels" in card:
        reasons.append(f"strongest change at {', '.join(card['channels'])} ({card['side']['side']})")
    if similar:
        reasons.append(f"resembles seizure at {similar[0]['when']} (similarity {similar[0]['similarity']:.2f})")
    if card["signal_quality"]["caution"]:
        reasons.append("caution: signal quality reduced; check for artefact")
    card["summary"] = reasons
    return card
