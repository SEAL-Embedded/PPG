"""
Physiological-plausibility validity gate (neutral signal acceptance).

This is the Tier-1 acceptance check for the PPG-vs-ECG agreement study. It
decides whether a stretch of beats is *physiologically possible*, NOT whether
its morphology is clean. That distinction is the whole point: skewness (SSQI)
and zero-crossing rate (ZSQI) are OUTCOME variables in this study (they are how
we rank body sites / skin tones for signal quality), so they must never be used
as an inclusion filter — doing so would select on the dependent variable and
make the site/skin-tone comparison circular. The checks here condition only on
beat timing, so they are safe to use as a gate.

The three rules are the physiological-plausibility half of the Orphanidou 2015
signal-quality index (we deliberately DROP its template-correlation term, which
is a morphology/quality measure and would reintroduce the bias above):

    1. Heart rate within [40, 180] bpm.
    2. No gap between successive beats > 3 s (i.e. no more than ~one missed
       beat in a window).
    3. Ratio of the largest to smallest beat-to-beat interval in a window
       < 2.2.

A window is "valid" only if all three pass. Validity is assessed on
non-overlapping windows so a recording's valid fraction has a clean
interpretation (no double-counting) for the eventual N -> M flow accounting.

This is intentionally a WEAK gate: it removes only the truly unanalyzable
stretches (no detectable beats, impossible rates, huge dropouts) while keeping
noisy-but-analyzable signal — which is correct here, because that noisy signal
carries the quality variation the study exists to measure.

Reference
---------
Orphanidou, C., Bonnici, T., Charlton, P., Clifton, D., Vallance, D., &
Tarassenko, L. (2015). Signal-quality indices for the electrocardiogram and
photoplethysmogram: derivation and applications to wireless monitoring. IEEE
Journal of Biomedical and Health Informatics, 19(3), 832-838.
doi:10.1109/JBHI.2014.2338351

VALIDATION PENDING: the specific numeric thresholds below (40-180 bpm, 3 s gap,
2.2 ratio, 10 s window) are taken from Orphanidou 2015 but are flagged for an
independent literature confirmation pass before being cited in the manuscript.
Change them in one place (the constants here) once confirmed.
"""

import numpy as np


# ── Literature thresholds (Orphanidou 2015) — VALIDATION PENDING ──────────────
HR_MIN_BPM = 40.0          # window mean HR must be >= this
HR_MAX_BPM = 180.0         # window mean HR must be <= this
MAX_GAP_S = 3.0            # no inter-beat interval may exceed this
MAX_MIN_RATIO = 2.2        # max(interval) / min(interval) in a window must be < this
WINDOW_S = 10.0            # non-overlapping assessment window length

# Implementation guard (NOT a literature quality threshold): a window needs at
# least this many intervals before the ratio / HR rules are meaningful. A
# window with fewer detected beats is treated as invalid ("too_few_beats")
# because the detector found essentially no analyzable cardiac activity there.
MIN_INTERVALS_PER_WINDOW = 4


def _assess_one_window(intervals_ms):
    """Apply the three physiological rules to one window's intervals (ms).

    Returns ``(valid, reasons, metrics)`` where ``reasons`` lists the failed
    rule keys and ``metrics`` carries the computed values for reporting.
    """
    w = np.asarray(intervals_ms, dtype=float)
    metrics = {
        "n_intervals": int(len(w)),
        "hr_bpm": float("nan"),
        "max_min_ratio": float("nan"),
        "max_gap_ms": float("nan"),
    }
    reasons = []

    if len(w) < MIN_INTERVALS_PER_WINDOW:
        reasons.append("too_few_beats")
        return False, reasons, metrics

    mean_interval = float(np.mean(w))
    hr = 60000.0 / mean_interval if mean_interval > 0 else float("nan")
    ratio = float(np.max(w) / np.min(w)) if np.min(w) > 0 else float("inf")
    max_gap = float(np.max(w))

    metrics["hr_bpm"] = hr
    metrics["max_min_ratio"] = ratio
    metrics["max_gap_ms"] = max_gap

    if not (HR_MIN_BPM <= hr <= HR_MAX_BPM):
        reasons.append("hr_out_of_range")
    if not (ratio < MAX_MIN_RATIO):
        reasons.append("interval_ratio_too_high")
    if not (max_gap <= MAX_GAP_S * 1000.0):
        reasons.append("gap_too_long")

    return (len(reasons) == 0), reasons, metrics


def assess_windows(intervals_ms, times_s, window_s=WINDOW_S):
    """Slice an interval series into non-overlapping windows and judge each.

    Parameters
    ----------
    intervals_ms : np.ndarray
        Beat-to-beat intervals in ms (RR for ECG, PPI for PPG).
    times_s : np.ndarray
        Timestamp of each interval in seconds (same length as
        ``intervals_ms``); the timestamp of the *later* peak of each pair, as
        returned by ``sqi.ccc.peaks_to_intervals``.
    window_s : float
        Non-overlapping window length in seconds.

    Returns
    -------
    list of dict
        One entry per window: ``{start_s, end_s, valid, reasons, n_intervals,
        hr_bpm, max_min_ratio, max_gap_ms}``. Empty list when there are no
        intervals.
    """
    intervals = np.asarray(intervals_ms, dtype=float)
    times = np.asarray(times_s, dtype=float)
    if len(intervals) != len(times):
        raise ValueError(
            f"intervals_ms ({len(intervals)}) and times_s ({len(times)}) "
            "must have the same length"
        )
    if len(intervals) == 0:
        return []

    t0 = float(np.min(times))
    t1 = float(np.max(times))
    span = t1 - t0
    # Recording shorter than one window: assess the whole span as a single
    # (partial) window rather than returning nothing.
    n_windows = max(1, int(np.ceil(span / window_s))) if span > 0 else 1

    windows = []
    for k in range(n_windows):
        lo = t0 + k * window_s
        hi = lo + window_s
        # Last window is closed on the right so the final interval is included.
        if k == n_windows - 1:
            mask = (times >= lo) & (times <= hi)
        else:
            mask = (times >= lo) & (times < hi)
        valid, reasons, metrics = _assess_one_window(intervals[mask])
        windows.append({
            "start_s": float(lo),
            "end_s": float(hi),
            "valid": bool(valid),
            "reasons": reasons,
            **metrics,
        })
    return windows


def physiological_validity(intervals_ms, times_s, window_s=WINDOW_S):
    """Summarise the physiological validity of a beat series.

    Wraps :func:`assess_windows` into a recording-level summary. Does NOT yet
    enforce a recording-acceptance threshold (e.g. ">= 80% valid windows") —
    that decision belongs to the pipeline driver and is added in a later step;
    here we only report the fraction so the threshold can be applied (and
    swept) downstream.

    Returns
    -------
    dict
        ``{n_windows, n_valid_windows, frac_valid, reason_counts, windows}``.
        ``frac_valid`` is NaN when there are no windows. ``reason_counts``
        tallies each failure reason across windows for the exclusion ledger.
    """
    windows = assess_windows(intervals_ms, times_s, window_s=window_s)
    n = len(windows)
    n_valid = sum(1 for w in windows if w["valid"])

    reason_counts = {}
    for w in windows:
        for r in w["reasons"]:
            reason_counts[r] = reason_counts.get(r, 0) + 1

    return {
        "n_windows": n,
        "n_valid_windows": n_valid,
        "frac_valid": (n_valid / n) if n else float("nan"),
        "reason_counts": reason_counts,
        "windows": windows,
    }
