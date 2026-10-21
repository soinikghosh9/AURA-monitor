"""A full clinical metric suite, so a model can be understood rather than merely ranked.

AUC alone has repeatedly misled this project. It said 0.67 on continuous forecasting while the same
model caught 8 % of pre-ictal windows at a 10 % warning budget -- *below* the 10 % a coin flip
achieves -- and cleared the analytic chance model on zero of three animals. A single threshold-free
ranking number cannot express that, so this module reports the quantities a clinician, a regulator and
a statistician would each ask for, grouped by the question they answer.

  discrimination  can it rank?          AUC, AUPRC, partial AUC, sensitivity at fixed specificity
  calibration     are the numbers true? Brier, skill score, ECE/MCE, Cox slope+intercept, reliability
  decision        is it useful?         net benefit / decision curve, Youden point, MCC, kappa, F1
  clinical        would it deploy?      sensitivity at a warning budget, false alarms per day/hour,
                                        detection latency, time in warning, event-level rates
  uncertainty     how sure are we?      bootstrap CIs on any of the above, resampled at the right unit

Two conventions are load-bearing and deliberate:
  * **AUPRC, not just AUC.** At a 0.3 % base rate the ROC is dominated by true negatives and looks
    flat-out optimistic; precision-recall is the honest picture of a rare-event detector.
  * **Events, not windows.** One seizure produces many correlated positive windows. Any rate that
    treats them as independent overstates significance by orders of magnitude, so event-level
    functions collapse contiguous positives first.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- discrimination


def auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(int)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def auprc(y, p) -> float:
    """Average precision. The honest discrimination metric when positives are rare."""
    from sklearn.metrics import average_precision_score
    y = np.asarray(y).astype(int)
    return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def auprc_skill(y, p) -> float:
    """AUPRC rescaled against the base rate, so cohorts with different prevalence are comparable.

    Raw AUPRC is bounded below by prevalence, so a 0.3 %-prevalence record and a balanced clip cohort
    cannot be compared directly. This maps chance to 0 and perfect to 1.
    """
    y = np.asarray(y).astype(int)
    base = float(y.mean())
    if base in (0.0, 1.0):
        return float("nan")
    return float((auprc(y, p) - base) / (1.0 - base))


def partial_auc(y, p, max_fpr: float = 0.1) -> float:
    """AUC restricted to the low-false-positive region, standardised to [0.5, 1].

    Clinically only the left of the ROC matters: nobody deploys an alarm operating at 50 % FPR.
    """
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p, max_fpr=max_fpr))


def sensitivity_at_specificity(y, p, spec: float = 0.95) -> float:
    from sklearn.metrics import roc_curve
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, p)
    ok = fpr <= (1.0 - spec)
    return float(tpr[ok].max()) if ok.any() else 0.0


# --------------------------------------------------------------------------- calibration


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def brier_skill(y, p, clim: float | None = None) -> float:
    """Brier skill against the climatological forecaster -- the honest reference for rare events."""
    y = np.asarray(y, float)
    clim = float(y.mean()) if clim is None else clim
    den = np.mean((clim - y) ** 2)
    return float(1.0 - brier(y, p) / den) if den > 0 else float("nan")


def calibration_error(y, p, n_bins: int = 10, norm: str = "l1") -> float:
    """Expected (l1) or maximum (max) calibration error over equal-width probability bins."""
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), 0, 1)
    edges = np.linspace(0, 1, n_bins + 1)
    errs, wts = [], []
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= 1.0)
        if m.sum() == 0:
            continue
        errs.append(abs(p[m].mean() - y[m].mean())); wts.append(m.sum())
    if not errs:
        return float("nan")
    errs, wts = np.array(errs), np.array(wts, float)
    return float(errs.max() if norm == "max" else (errs * wts).sum() / wts.sum())


def calibration_slope_intercept(y, p) -> tuple[float, float]:
    """Cox calibration: regress the outcome on the predicted logit.

    Slope 1 and intercept 0 is perfect. Slope < 1 means over-confident (predictions too extreme),
    which is the failure mode of an uncalibrated deep model; intercept captures systematic over- or
    under-forecasting of the base rate. This separates the two, which a single ECE cannot.
    """
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(y).astype(int); p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    lg = np.log(p / (1 - p)).reshape(-1, 1)
    m = LogisticRegression(max_iter=1000).fit(lg, y)
    return float(m.coef_[0][0]), float(m.intercept_[0])


def reliability_curve(y, p, n_bins: int = 10) -> list[tuple[float, float, int]]:
    """(mean predicted, observed frequency, n) per bin -- the data behind a reliability diagram."""
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), 0, 1)
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= 1.0)
        if m.sum():
            out.append((round(float(p[m].mean()), 4), round(float(y[m].mean()), 4), int(m.sum())))
    return out


def conformal_coverage(y, p, cal_frac: float = 0.3, alpha: float = 0.1, seed: int = 0) -> dict:
    """Split-conformal risk control: does a claimed (1-alpha) guarantee actually hold out of sample?

    Distribution-free and finite-sample, which is what a regulator wants from a risk statement -- and
    unlike a reliability curve it makes a claim that can simply be false, so it can be checked.
    """
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    ncal = max(10, int(cal_frac * len(y)))
    cal, test = idx[:ncal], idx[ncal:]
    if len(test) < 10 or y[cal].sum() == 0:
        return {"target": 1 - alpha, "empirical": None, "n_test": int(len(test))}
    # nonconformity: 1 - p on positives; threshold at the (1-alpha) quantile of calibration positives
    scores = 1.0 - p[cal][y[cal] == 1]
    q = float(np.quantile(scores, min(1.0, (1 - alpha) * (1 + 1 / max(1, len(scores))))))
    covered = (1.0 - p[test][y[test] == 1]) <= q
    return {"target": round(1 - alpha, 3),
            "empirical": round(float(covered.mean()), 3) if covered.size else None,
            "threshold": round(1 - q, 4), "n_test_pos": int(covered.size)}


# --------------------------------------------------------------------------- decision value


def net_benefit(y, p, threshold: float) -> float:
    """Decision-curve net benefit at a threshold probability.

    Net benefit puts true and false positives on one scale using the threshold as the implied
    exchange rate: acting at threshold t means one missed seizure is worth t/(1-t) false alarms. It
    answers "is using this model better than treating everyone or no one", which no discrimination
    metric does, and it is the standard tool for clinical prediction models.
    """
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    n = len(y)
    if n == 0 or not 0 < threshold < 1:
        return float("nan")
    tp = int(((p >= threshold) & (y == 1)).sum())
    fp = int(((p >= threshold) & (y == 0)).sum())
    return float(tp / n - (fp / n) * (threshold / (1 - threshold)))


def decision_curve(y, p, thresholds=(0.05, 0.1, 0.2, 0.3, 0.5)) -> dict:
    """Net benefit of the model, of always-act, and of never-act (which is 0) at each threshold."""
    y = np.asarray(y).astype(int)
    base = float(y.mean())
    return {str(t): {"model": round(net_benefit(y, p, t), 5),
                     "treat_all": round(base - (1 - base) * (t / (1 - t)), 5),
                     "treat_none": 0.0} for t in thresholds}


def operating_point(y, p, rule: str = "youden") -> dict:
    """Threshold and the full confusion-derived panel at it."""
    from sklearn.metrics import roc_curve
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return {}
    fpr, tpr, thr = roc_curve(y, p)
    i = int(np.argmax(tpr - fpr)) if rule == "youden" else int(np.argmax(tpr))
    t = float(thr[i])
    pred = (np.asarray(p) >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1); npv = tn / max(tn + fn, 1)
    mcc_den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return {"threshold": round(t, 4), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sensitivity": round(sens, 4), "specificity": round(spec, 4),
            "ppv": round(ppv, 4), "npv": round(npv, 4),
            "balanced_acc": round(0.5 * (sens + spec), 4),
            "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 4),
            "mcc": round(float((tp * tn - fp * fn) / mcc_den) if mcc_den > 0 else 0.0, 4),
            "youden_j": round(sens + spec - 1, 4)}


# --------------------------------------------------------------------------- clinical / temporal


def _events(y, max_gap: int = 1) -> list[np.ndarray]:
    idx = np.flatnonzero(np.asarray(y).astype(bool))
    if idx.size == 0:
        return []
    return np.split(idx, np.where(np.diff(idx) > max_gap + 1)[0] + 1)


def event_metrics(y, p, threshold: float, hours_per_window: float = 1.0) -> dict:
    """Event-level sensitivity, false-alarm rate and detection latency.

    The clinical unit is the seizure, not the window: a model that fires on 40 of one seizure's 50
    windows has detected ONE event, and counting 40 inflates both sensitivity and significance.
    """
    y = np.asarray(y).astype(int); warn = np.asarray(p) >= threshold
    evs = _events(y)
    caught = [e for e in evs if warn[e].any()]
    # false alarms: warning runs that overlap no event
    fa_runs = [r for r in _events(warn.astype(int)) if not y[r].any()]
    total_h = len(y) * hours_per_window
    lat = [int(np.argmax(warn[e])) * hours_per_window for e in caught if warn[e].any()]
    return {"n_events": len(evs), "n_events_detected": len(caught),
            "event_sensitivity": round(len(caught) / max(len(evs), 1), 4),
            "false_alarms": len(fa_runs),
            "false_alarms_per_day": round(len(fa_runs) / max(total_h / 24.0, 1e-9), 4),
            "false_alarms_per_hour": round(len(fa_runs) / max(total_h, 1e-9), 5),
            "time_in_warning": round(float(warn.mean()), 4),
            "median_detection_latency_h": round(float(np.median(lat)), 4) if lat else None}


def sensitivity_at_budget(y, p, budget: float = 0.10) -> dict:
    """Event sensitivity when the alarm is allowed to be on for `budget` of the time.

    The analytic chance model (Winterhalder/Snyder) gives S_nc = time-in-warning, so `budget` is also
    the score to beat: 8 % sensitivity at a 10 % budget is worse than random warning.
    """
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    thr = float(np.quantile(p, 1.0 - budget))
    em = event_metrics(y, p, thr)
    return {"budget": budget, "threshold": round(thr, 5),
            "event_sensitivity": em["event_sensitivity"],
            "chance_sensitivity": budget,
            "beats_chance": bool(em["event_sensitivity"] > budget),
            "false_alarms_per_day": em["false_alarms_per_day"]}


# --------------------------------------------------------------------------- uncertainty


def bootstrap_ci(fn, y, p, groups=None, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Bootstrap CI for any metric, resampling GROUPS when given.

    The resampling unit must be the unit of generalisation -- subjects for a cross-patient claim,
    seizures for an event-rate claim. Resampling windows when the claim is about subjects is the
    single most common way CIs in this literature end up far too narrow.
    """
    y = np.asarray(y); p = np.asarray(p)
    rng = np.random.default_rng(seed)
    vals = []
    if groups is None:
        n = len(y)
        for _ in range(n_boot):
            i = rng.integers(0, n, n)
            v = fn(y[i], p[i])
            if v == v:
                vals.append(v)
    else:
        groups = np.asarray(groups); uq = np.unique(groups)
        for _ in range(n_boot):
            g = rng.choice(uq, len(uq), replace=True)
            i = np.concatenate([np.where(groups == q)[0] for q in g])
            v = fn(y[i], p[i])
            if v == v:
                vals.append(v)
    if not vals:
        return {"point": None, "lo": None, "hi": None}
    return {"point": round(float(fn(y, p)), 4),
            "lo": round(float(np.percentile(vals, 100 * alpha / 2)), 4),
            "hi": round(float(np.percentile(vals, 100 * (1 - alpha / 2))), 4)}


# --------------------------------------------------------------------------- the full panel


def full_panel(y, p, groups=None, hours_per_window: float = 1.0, budgets=(0.05, 0.10, 0.25),
               with_ci: bool = True, n_boot: int = 1000) -> dict:
    """Every metric above, grouped by the question it answers. One call per model per task."""
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    slope, intercept = calibration_slope_intercept(y, p)
    out = {
        "n": int(len(y)), "n_pos": int(y.sum()), "prevalence": round(float(y.mean()), 5),
        "discrimination": {
            "auc": round(auc(y, p), 4), "auprc": round(auprc(y, p), 4),
            "auprc_skill": round(auprc_skill(y, p), 4),
            "partial_auc_fpr10": round(partial_auc(y, p, 0.1), 4),
            "sens_at_spec95": round(sensitivity_at_specificity(y, p, 0.95), 4),
            "sens_at_spec99": round(sensitivity_at_specificity(y, p, 0.99), 4),
        },
        "calibration": {
            "brier": round(brier(y, p), 5),
            "brier_skill": round(brier_skill(y, p), 4),
            "ece": round(calibration_error(y, p), 4),
            "mce": round(calibration_error(y, p, norm="max"), 4),
            "cox_slope": round(slope, 4), "cox_intercept": round(intercept, 4),
            "reliability": reliability_curve(y, p),
            "conformal": conformal_coverage(y, p),
        },
        "decision": {"operating_point": operating_point(y, p), "net_benefit": decision_curve(y, p)},
        "clinical": {f"budget_{int(b*100)}pct": sensitivity_at_budget(y, p, b) for b in budgets},
    }
    op = out["decision"]["operating_point"]
    if op:
        out["clinical"]["at_youden"] = event_metrics(y, p, op["threshold"], hours_per_window)
    if with_ci:
        out["ci"] = {"auc": bootstrap_ci(auc, y, p, groups, n_boot),
                     "auprc": bootstrap_ci(auprc, y, p, groups, n_boot)}
    return out
