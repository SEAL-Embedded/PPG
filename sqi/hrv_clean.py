"""
NN-interval cleaning for HRV computation.

Two-pass filter applied between peak detection and HRV feature
computation to ensure SDNN, RMSSD, LF/HF, SampEn are computed on
artefact-free intervals (NN = "Normal-to-Normal", per HRV convention).

Pass 1 — physiological range gate
    Drop intervals outside [300, 2000] ms (200 BPM .. 30 BPM).

Pass 2 — Karlsson 1987 local-median rule
    Drop intervals that differ by more than ``karlsson_pct`` from a
    local rolling median computed over ``window`` neighbours. Default
    ``karlsson_pct=0.20`` (20% rule).

Reference
---------
Karlsson, M., et al. (1987). Detection of ectopic heartbeats in
long-term electrocardiograms. Computers and Biomedical Research,
20(4), 333-340.
"""

import numpy as np


DEFAULT_RANGE_MS = (300.0, 2000.0)
DEFAULT_KARLSSON_PCT = 0.20
DEFAULT_WINDOW = 5


def clean_intervals(intervals_ms, times_s, range_ms=DEFAULT_RANGE_MS,
                    karlsson_pct=DEFAULT_KARLSSON_PCT,
                    window=DEFAULT_WINDOW):
    """Apply two-pass cleaning to an interval series.

    Parameters
    ----------
    intervals_ms : np.ndarray
        Interval lengths in ms.
    times_s : np.ndarray
        Corresponding timestamps in seconds (same shape as
        ``intervals_ms``).
    range_ms : (low, high)
        Acceptance range for the physiological range gate.
    karlsson_pct : float in (0, 1)
        Fractional tolerance vs local median.
    window : int
        Number of neighbours for the rolling median (centered window
        of ``±window // 2`` on each side).

    Returns
    -------
    cleaned_intervals : np.ndarray
        Intervals that survived both passes.
    cleaned_times : np.ndarray
        Aligned timestamps.
    mask : np.ndarray of bool
        ``len(intervals_ms)``-long boolean mask, True where the
        interval was kept.
    """
    intervals = np.asarray(intervals_ms, dtype=float)
    times = np.asarray(times_s, dtype=float)

    # Pass 1: physiological range gate
    low, high = range_ms
    in_range = (intervals >= low) & (intervals <= high)

    mask = in_range
    return intervals[mask].copy(), times[mask].copy(), mask
