"""
Per-session analysis driver for the SEAL PPG webapp.

For one ``session_<ts>/`` folder we run:

    SSQI    skewness of the raw PPG    (sqi.SSQI_algorithm.Ssqi)
    ZSQI    windowed zero-crossing     (sqi.zcr_sqi.windowed_zcr)
    KSQI    kurtosis of the raw PPG    (sqi.KSQI_algorithm.Ksqi)
    CCC     RR (ECG) vs PPI (PPG)      (sqi.ccc.*)
            -> Lin's CCC, Pearson r, Bland-Altman, RMSE, MAE

The peak detection and interval-matching reuse exactly the functions in
sqi/ccc.py (detect_r_peaks, detect_ppg_peaks, peaks_to_intervals,
match_intervals, compute_ccc). We do not re-implement any of them.

We also expose a signal-loading helper that downsamples the raw CSVs
into JSON-able arrays for the front-end Plotly plots — this is the
``vis.py`` / ``fullvis.py`` equivalent the user asked for, rendered in
the browser instead of a matplotlib popup.
"""

import glob
import os
import re

# pingouin builds throwaway matplotlib figures during
# analysis. This runs on uvicorn worker threads, so the default interactive
# Tk backend creates Tk objects off the main thread — their garbage-collected
# __del__ then raises "main thread is not in main loop" and the process-killing
# "Tcl_AsyncDelete: async handler deleted by the wrong thread". Pin the headless
# Agg backend before any import can pull in pyplot. Must precede the
# pingouin / sqi imports below.
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import butter, detrend, filtfilt, find_peaks, welch

from sqi.ccc import (
    bandpass,
    ccc_label,
    compute_ccc,
    detect_r_peaks,
    lowpass,
    match_intervals,
    peaks_to_intervals,
)
from sqi.KSQI_algorithm import Ksqi
from sqi.SSQI_algorithm import Ssqi
from sqi.zcr_sqi import windowed_zcr
from sqi.hrv_clean import clean_intervals

from . import sessions

# Pingouin powers ICC3 on matched RR/PPI pairs — same library sqi/ICC.py uses
# for the cross-subject form. Import lazily-guarded so a missing install
# degrades to "icc": None rather than crashing the analyze endpoint.
try:
    import pingouin as _pg
except Exception:        # pragma: no cover — package optional
    _pg = None

# Frequency-domain HRV (LF / HF / LF-HF ratio) runs on scipy directly — see
# _freq_domain_metrics. This used to call pyhrv.frequency_domain.welch_psd
# behind a try/except, but pyhrv pulls in `spectrum` (needs a C/Fortran
# toolchain), is unmaintained, and predates numpy 2, so it is not installed
# here — and requirements.txt lists it as optional/legacy on purpose. The
# guard meant every LF/HF cell silently came back NaN and the LF/HF batch
# table rendered empty. scipy (already a hard dependency) does the same
# Welch PSD, so the metric now always computes.


CHANNEL_RE = re.compile(r".*ppg_data_ch(\d+)\.csv$")


# ── Loaders (schema matches PPG_ECG_Full_Unpacking.py output) ────────────────

def _unwrap_ticks_us(ts_us):
    """Unwrap a MicroPython ticks_us series that may have wrapped at 2^30.

    The Pi Pico's ``ticks_us()`` uses a 30-bit counter that wraps every
    2^30 microseconds (~17.89 minutes). Any recording longer than that
    will contain a backward jump in the CSV's first column that poisons
    every diff-based downstream metric (``infer_fs``,
    ``peaks_to_intervals``, etc.). This helper detects backward jumps
    larger than 2^29 (half the wrap period — anything smaller is just
    out-of-order jitter, not a wrap) and adds the wrap period to all
    subsequent samples. Returns a monotone float64 array so a later
    ``/ 1000.0`` to milliseconds keeps full precision.
    """
    arr = np.asarray(ts_us, dtype=np.float64)
    if len(arr) < 2:
        return arr
    WRAP = float(1 << 30)
    THRESH = float(1 << 29)
    diffs = np.diff(arr)
    # 1 where a wrap occurred, 0 elsewhere
    wraps = (diffs < -THRESH).astype(np.float64)
    offset = np.concatenate([[0.0], np.cumsum(wraps) * WRAP])
    return arr + offset


def load_ppg(path):
    """``ppg_data_ch{N}.csv`` — col0=ts_us, col1=sample (headerless).

    Tolerates a partially-flushed last line so the live-view poll can
    read the file while the receiver is still writing to it.
    """
    df = pd.read_csv(path, header=None, names=["ts_us", "sample"],
                     on_bad_lines="skip")
    ts_us = pd.to_numeric(df["ts_us"], errors="coerce").to_numpy(dtype=float)
    sig = pd.to_numeric(df["sample"], errors="coerce").to_numpy(dtype=float)
    valid = ~(np.isnan(ts_us) | np.isnan(sig))
    ts_us_unwrapped = _unwrap_ticks_us(ts_us[valid])
    return ts_us_unwrapped / 1000.0, sig[valid]   # return ms, sample


def load_ecg(path):
    """``ecg_data.csv`` — col0=ts_us, col1=sample, col2=leads_off."""
    df = pd.read_csv(path, header=None, names=["ts_us", "sample", "leads_off"],
                     on_bad_lines="skip")
    ts_us = pd.to_numeric(df["ts_us"], errors="coerce").to_numpy(dtype=float)
    sig = pd.to_numeric(df["sample"], errors="coerce").to_numpy(dtype=float)
    leads_off = pd.to_numeric(df["leads_off"], errors="coerce").to_numpy(dtype=float)
    valid = ~(np.isnan(ts_us) | np.isnan(sig))
    ts_us_unwrapped = _unwrap_ticks_us(ts_us[valid])
    return ts_us_unwrapped / 1000.0, sig[valid], leads_off[valid].astype(int)


def ppg_bandpass(sig, fs, lowcut=0.5, highcut=8.0, order=2):
    """Cardiac bandpass overlay for the dashboard PPG view.

    Delegates to ``sqi.ccc.ppg_bandpass`` so the display overlay and the
    PPG peak detector consume the same filtered waveform — peaks the user
    sees on the dashboard are exactly the peaks the detector ran on.

    The defaults match the canonical filter (0.5-8 Hz, 2nd-order
    zero-phase Butterworth); the kwargs are forwarded so any external
    caller passing them keeps the same parameter semantics.
    """
    from sqi.ccc import ppg_bandpass as _ccc_ppg_bandpass
    return _ccc_ppg_bandpass(sig, fs, low=lowcut, high=highcut, order=order)


def infer_fs(ts_ms):
    """Instantaneous cadence: 1 / median(inter-sample interval).

    This is the *typical* spacing between consecutive good samples, so it
    is robust to a handful of oversized gaps. It is the right rate for the
    signal-processing grid (``_resample_uniform``) and for Butterworth
    filter design, which want the local sampling period rather than the
    long-run average. It is NOT the true throughput — see ``effective_fs``.
    """
    if len(ts_ms) < 2:
        return float("nan")
    dt = float(np.median(np.diff(ts_ms)))
    return 1000.0 / dt if dt > 0 else float("nan")


def effective_fs(ts_ms):
    """Actual average sampling rate: samples over the wall-clock span they
    cover, i.e. (N - 1) / (t_last - t_first). This is what the dashboard
    reports.

    Unlike ``infer_fs`` (median inter-sample rate), this counts every gap,
    stall, and dropped burst against the rate, so a channel that briefly
    hits 400 Hz but is starved for ~40% of the recording reports its true
    ~185 Hz throughput rather than the 375 Hz the median would show.
    """
    if len(ts_ms) < 2:
        return float("nan")
    span_ms = float(ts_ms[-1] - ts_ms[0])
    return (len(ts_ms) - 1) * 1000.0 / span_ms if span_ms > 0 else float("nan")


# ── Hybrid PPG conditioning + peak detection ─────────────────────────────────
#
# This mirrors the signal conditioning in signal_visualization/ppgvis.py
# (drop non-monotonic timestamps -> cubic resample onto a uniform grid ->
# rolling-median outlier removal -> 0.5-8 Hz bandpass before peak
# detection) but keeps two things from the original detector that are
# better on this rig's raw-ADC data: a *scale-relative* prominence
# (ppgvis's absolute prominence=1.1 is ~0 on counts-in-the-thousands), and
# interval timing off the *recorded* timestamps rather than the resample
# grid. SSQI/KSQI are deliberately NOT computed on the bandpassed signal —
# the band-pass distorts the pulse shape, and with it both the skew and the
# tail weight the two moments measure.

def _fix_timestamp_spikes(ts_ms, sig, jump_factor=50.0):
    """Repair samples whose timestamp jumps implausibly far (in either
    direction) from the recording's own sample spacing, e.g. a firmware
    tick-counter glitch/32-bit rollover, by resetting that one step to the
    median inter-sample interval instead of trusting the raw delta.

    Left uncorrected, a single such jump inflates any duration computed as
    ts[-1]-ts[0] (see analyze_channel's mean_hr_bpm) and, downstream, makes
    _drop_non_monotonic discard every real sample that follows once the
    corrupted value becomes the new running max. ``jump_factor`` is kept
    large so ordinary back-dated bursts (handled separately below) are left
    alone — only jumps far outside anything plausible get reset.
    """
    if len(ts_ms) < 3:
        return ts_ms, sig
    deltas = np.diff(ts_ms).astype(float)
    median_dt = np.median(deltas)
    if median_dt <= 0:
        return ts_ms, sig
    bad = np.abs(deltas - median_dt) > jump_factor * median_dt
    if not bad.any():
        return ts_ms, sig
    deltas[bad] = median_dt
    fixed = np.empty(len(ts_ms), dtype=float)
    fixed[0] = ts_ms[0]
    fixed[1:] = ts_ms[0] + np.cumsum(deltas)
    return fixed, sig


def _drop_non_monotonic(ts_ms, sig):
    """Drop samples whose timestamp doesn't strictly exceed the running max.

    The drain-all firmware back-dates burst samples, so a channel's
    timestamps can step backward mid-stream; left in, interp1d rejects the
    duplicate/decreasing x and peaks_to_intervals can emit a negative PPI.
    """
    if len(ts_ms) < 2:
        return ts_ms, sig
    running_max = np.maximum.accumulate(ts_ms)
    keep = np.empty(len(ts_ms), dtype=bool)
    keep[0] = True
    keep[1:] = ts_ms[1:] > running_max[:-1]
    return ts_ms[keep], sig[keep]


def _resample_uniform(ts_ms, sig, fs):
    """Cubic-resample (ts_ms, sig) onto a uniform grid at ``fs`` Hz.

    Returns ``(grid_ts_ms, grid_sig)``. Falls back to the input unchanged
    when fs is unusable or there are too few points for a cubic spline —
    so a degenerate channel still flows through the rest of the pipeline.
    """
    if not np.isfinite(fs) or fs <= 0 or len(ts_ms) < 4:
        return ts_ms, sig
    t0, t1 = float(ts_ms[0]), float(ts_ms[-1])
    n = int(np.ceil((t1 - t0) / 1000.0 * fs))
    if n < 2:
        return ts_ms, sig
    grid = np.linspace(t0, t1, n)
    interp = interp1d(ts_ms, sig, kind="cubic", fill_value="extrapolate")
    return grid, interp(grid)


def _remove_outliers(sig, window=15, n_sigma=4.25):
    """ppgvis's outlier scrub: flag samples that sit > n_sigma from a
    centered rolling median, replace them by linear interpolation across
    the good neighbours. Operates on the (uniform) sample index, so it's
    meant to run after _resample_uniform."""
    if len(sig) < 3:
        return sig
    s = pd.Series(sig)
    med = s.rolling(window=window, center=True, min_periods=1).median()
    gap = (med - s).abs()
    thr = n_sigma * float(np.std(gap.to_numpy()))
    if not np.isfinite(thr) or thr <= 0:
        return sig
    outliers = (gap > thr).to_numpy()
    if not outliers.any() or outliers.all():
        return sig
    out = np.asarray(sig, dtype=float).copy()
    idx = np.arange(len(out))
    out[outliers] = np.interp(idx[outliers], idx[~outliers], out[~outliers])
    return out


def _filter_doublets(peaks, sig, fs, frac=0.4):
    """Drop doublet detections, keeping the taller peak of each too-close pair.

    Ported from signal_visualization/ppgvis.py: take the median inter-peak
    interval, then walk left to right keeping a peak only when it sits more
    than ``frac * median`` from the last kept peak. When two peaks are too
    close to be separate beats, keep whichever has the larger amplitude in
    ``sig`` (the true systolic peak) and drop the other — so a spurious low
    peak doesn't survive at the expense of the real one, regardless of which
    came first. ``peaks`` index into ``sig``; intervals use the uniform grid
    spacing 1/fs."""
    if len(peaks) < 3:
        return peaks
    peak_times = peaks / fs
    median_rr = float(np.median(np.diff(peak_times)))
    if not np.isfinite(median_rr) or median_rr <= 0:
        return peaks
    kept = [0]   # positions into ``peaks`` of the peaks kept so far
    for i in range(1, len(peaks)):
        rr = peak_times[i] - peak_times[kept[-1]]
        if rr > frac * median_rr:
            kept.append(i)
        elif sig[peaks[i]] > sig[peaks[kept[-1]]]:
            kept[-1] = i
    return peaks[np.asarray(kept, dtype=int)]


def detect_ppg_peaks_bp(sig, fs):
    """0.5-8 Hz bandpass (ppg_bandpass) then find_peaks with a scale-relative
    prominence (1.1·std of the bandpassed signal) and width 0.13, and a
    0.5·fs minimum spacing (~120 bpm cap), followed by the ppgvis median-RR
    doublet filter. Returns indices into ``sig``."""
    bp = ppg_bandpass(sig, fs)
    if bp is None:
        return np.array([], dtype=int)
    prominence = 1.1 * float(np.std(bp))
    if not np.isfinite(prominence) or prominence <= 0:
        return np.array([], dtype=int)
    distance = max(1, int(0.5 * fs))
    peaks, _ = find_peaks(bp, distance=distance, prominence=prominence, width=0.13)
    peaks = _filter_doublets(peaks, bp, fs)
    return peaks


def _moving_average(x, w):
    """Centered moving average of window length ``w`` samples, same length as
    ``x`` (zero-padded at the edges via ``np.convolve(mode="same")``)."""
    if w <= 1:
        return np.asarray(x, dtype=float)
    kernel = np.ones(int(w), dtype=float) / float(w)
    return np.convolve(np.asarray(x, dtype=float), kernel, mode="same")


def detect_ppg_peaks_terma(sig, fs, w1_ms=111.0, w2_ms=667.0, beta=0.02):
    """Elgendi 2013 systolic-peak detector (two event-related moving averages
    + offset threshold).

    Reference: Elgendi M, et al. "Systolic Peak Detection in Acceleration
    Photoplethysmograms Measured from Emergency Responders in Tropical
    Conditions." PLoS ONE 2013;8(10):e76585.

    Pipeline:
      1. 0.5-8 Hz bandpass (ppg_bandpass) — shared with detect_ppg_peaks_bp.
      2. Clip negatives to zero, then square -> emphasises systolic upslopes,
         suppresses diastolic/noise.
      3. MA_peak: moving average over ``w1_ms`` (~systolic-peak width).
         MA_beat: moving average over ``w2_ms`` (~one heartbeat).
      4. Threshold THR1 = MA_beat + beta*mean(squared); "blocks of interest"
         are the runs where MA_peak > THR1.
      5. Reject blocks narrower than ``w1`` (noise); within each surviving
         block, the index of the max *bandpassed* sample is the systolic peak.
      6. Small-RR doublet filter (_filter_doublets, frac=0.6): drop peaks
         closer than 0.6*median to the last kept peak, keeping the taller.

    Unlike the find_peaks detector this needs no prominence/distance tuning —
    the threshold adapts to the local signal level. Returns indices into
    ``sig``."""
    bp = ppg_bandpass(sig, fs)
    if bp is None:
        return np.array([], dtype=int)

    # Step 2: clip + square.
    clipped = np.clip(bp, 0.0, None)
    squared = clipped * clipped

    w1 = max(1, int(round(w1_ms / 1000.0 * fs)))
    w2 = max(1, int(round(w2_ms / 1000.0 * fs)))
    if len(squared) <= w2:
        return np.array([], dtype=int)

    # Step 3 + 4: two moving averages and the adaptive threshold.
    ma_peak = _moving_average(squared, w1)
    ma_beat = _moving_average(squared, w2)
    alpha = beta * float(np.mean(squared))
    blocks = ma_peak > (ma_beat + alpha)

    # Step 5: keep blocks at least w1 wide; take the bandpassed max in each.
    peaks = []
    n = len(blocks)
    i = 0
    while i < n:
        if not blocks[i]:
            i += 1
            continue
        j = i
        while j < n and blocks[j]:
            j += 1
        if (j - i) >= w1:
            peaks.append(i + int(np.argmax(bp[i:j])))
        i = j
    peaks = np.asarray(peaks, dtype=int)
    # Same small-RR doublet filter as the ppgvis pipeline: drop peaks whose
    # interval from the last kept peak is below 0.6*median, keeping the taller
    # of any too-close pair (amplitude on the bandpassed signal).
    return _filter_doublets(peaks, bp, fs, frac=0.6)


# Active PPG peak detector. "terma" = Elgendi 2013 adaptive two-moving-average
# detector (detect_ppg_peaks_terma); set to "prominence" to revert to the
# scale-relative find_peaks detector (detect_ppg_peaks_bp).
PPG_PEAK_DETECTOR = "terma"


def _detect_ppg_peaks(sig, fs):
    """Dispatch to the active PPG peak detector (see ``PPG_PEAK_DETECTOR``)."""
    if PPG_PEAK_DETECTOR == "terma":
        return detect_ppg_peaks_terma(sig, fs)
    return detect_ppg_peaks_bp(sig, fs)


# ── Windowed HR agreement (within a single session) ─────────────────────────

def _windowed_hr_agreement(ecg_ts_ms, ecg_peaks, ppg_ts_ms, ppg_peaks,
                            window_s=30.0, step_s=5.0, min_peaks_per_window=3):
    """Pair ECG and PPG mean HR over sliding windows within one session and
    run CCC / ICC / Bland-Altman on the paired window vectors.

    Window HR is count-based (``60·n_peaks/window_s``) — the same formula
    the per-session mean HR uses, so a single-window equivalent collapses
    to the original number. Windows that don't carry ≥``min_peaks_per_window``
    peaks on *both* sides are dropped so a stretch of missed detections
    can't inflate the LOA.

    Returns ``None`` if the recording is shorter than one window. The
    result dict has the same key names ``_hr_agreement_per_channel`` uses
    so the frontend can render it with the same column set."""
    if len(ecg_ts_ms) < 2 or len(ppg_ts_ms) < 2:
        return None

    ecg_peak_ms = ecg_ts_ms[ecg_peaks] if len(ecg_peaks) else np.array([], dtype=float)
    ppg_peak_ms = ppg_ts_ms[ppg_peaks] if len(ppg_peaks) else np.array([], dtype=float)

    t_start = max(float(ecg_ts_ms[0]), float(ppg_ts_ms[0]))
    t_end   = min(float(ecg_ts_ms[-1]), float(ppg_ts_ms[-1]))
    if (t_end - t_start) < window_s * 1000.0:
        return None

    win_ms  = window_s * 1000.0
    step_ms = step_s * 1000.0

    e_hrs, p_hrs = [], []
    t = t_start
    while t + win_ms <= t_end:
        t_hi = t + win_ms
        n_e = int(np.sum((ecg_peak_ms >= t) & (ecg_peak_ms < t_hi)))
        n_p = int(np.sum((ppg_peak_ms >= t) & (ppg_peak_ms < t_hi)))
        if n_e >= min_peaks_per_window and n_p >= min_peaks_per_window:
            e_hrs.append(60.0 * n_e / window_s)
            p_hrs.append(60.0 * n_p / window_s)
        t += step_ms

    n = len(e_hrs)
    out = {
        "window_s": float(window_s),
        "step_s":   float(step_s),
        "min_peaks_per_window": int(min_peaks_per_window),
        "n_windows":     n,
        "mean_hr_ecg_bpm": float("nan"),
        "mean_hr_ppg_bpm": float("nan"),
        "ccc":            float("nan"),
        "pearson_r":      float("nan"),
        "bias_bpm":       float("nan"),
        "loa_lower_bpm":  float("nan"),
        "loa_upper_bpm":  float("nan"),
        "rmse_bpm":       float("nan"),
        "mae_bpm":        float("nan"),
        "icc":            float("nan"),
        "icc_ci_low":     float("nan"),
        "icc_ci_high":    float("nan"),
    }
    if n < 2:
        return out

    e = np.asarray(e_hrs, dtype=float)
    p = np.asarray(p_hrs, dtype=float)
    out["mean_hr_ecg_bpm"] = float(np.mean(e))
    out["mean_hr_ppg_bpm"] = float(np.mean(p))
    try:
        st = compute_ccc(p, e)
        out.update({
            "ccc":           float(st["ccc"]),
            "pearson_r":     float(st["pearson_r"]),
            "bias_bpm":      float(st["bias"]),
            "loa_lower_bpm": float(st["loa_lower"]),
            "loa_upper_bpm": float(st["loa_upper"]),
            "rmse_bpm":      float(st["rmse"]),
            "mae_bpm":       float(st["mae"]),
        })
    except (ValueError, ZeroDivisionError):
        pass
    icc = _safe_icc(e, p)
    if icc:
        out.update({
            "icc":         float(icc["icc"]),
            "icc_ci_low":  float(icc["ci_low"]),
            "icc_ci_high": float(icc["ci_high"]),
        })
    return out


def _windowed_sdnn_agreement(ecg_ts_ms, ecg_peaks, ppg_ts_ms, ppg_peaks,
                              window_s=60.0, step_s=10.0,
                              min_intervals_per_window=10):
    """Pair ECG and PPG SDNN over sliding windows within one session and
    run CCC / ICC / Bland-Altman on the paired vectors.

    SDNN is a second-moment statistic so it needs more samples per window
    to be stable than HR does: 60 s / 10 s step with a floor of ten
    intervals on each side is roughly the shortest window that gives a
    usable per-window SDNN at resting HR. An interval is assigned to the
    window its trailing (later) peak falls in.

    Returns ``None`` if the recording is shorter than one window.
    Result-dict keys mirror ``_windowed_hr_agreement`` (s/ms units swapped
    in the obvious places) so the frontend can render with one helper."""
    if len(ecg_peaks) < 2 or len(ppg_peaks) < 2:
        return None

    # Build (interval_ms, trailing-peak time_ms) for both sides.
    ecg_int_ms = np.diff(ecg_ts_ms[ecg_peaks])
    ecg_end_ms = ecg_ts_ms[ecg_peaks][1:]
    ppg_int_ms = np.diff(ppg_ts_ms[ppg_peaks])
    ppg_end_ms = ppg_ts_ms[ppg_peaks][1:]

    t_start = max(float(ecg_ts_ms[0]), float(ppg_ts_ms[0]))
    t_end   = min(float(ecg_ts_ms[-1]), float(ppg_ts_ms[-1]))
    if (t_end - t_start) < window_s * 1000.0:
        return None

    win_ms  = window_s * 1000.0
    step_ms = step_s * 1000.0

    e_sdnns, p_sdnns = [], []
    t = t_start
    while t + win_ms <= t_end:
        t_hi = t + win_ms
        e_mask = (ecg_end_ms >= t) & (ecg_end_ms < t_hi)
        p_mask = (ppg_end_ms >= t) & (ppg_end_ms < t_hi)
        n_e = int(e_mask.sum())
        n_p = int(p_mask.sum())
        if n_e >= min_intervals_per_window and n_p >= min_intervals_per_window:
            e_sdnns.append(float(np.std(ecg_int_ms[e_mask], ddof=1)))
            p_sdnns.append(float(np.std(ppg_int_ms[p_mask], ddof=1)))
        t += step_ms

    n = len(e_sdnns)
    out = {
        "window_s": float(window_s),
        "step_s":   float(step_s),
        "min_intervals_per_window": int(min_intervals_per_window),
        "n_windows":       n,
        "mean_sdnn_ecg_ms": float("nan"),
        "mean_sdnn_ppg_ms": float("nan"),
        "ccc":             float("nan"),
        "pearson_r":       float("nan"),
        "bias_ms":         float("nan"),
        "loa_lower_ms":    float("nan"),
        "loa_upper_ms":    float("nan"),
        "rmse_ms":         float("nan"),
        "mae_ms":          float("nan"),
        "icc":             float("nan"),
        "icc_ci_low":      float("nan"),
        "icc_ci_high":     float("nan"),
    }
    if n < 2:
        return out

    e = np.asarray(e_sdnns, dtype=float)
    p = np.asarray(p_sdnns, dtype=float)
    out["mean_sdnn_ecg_ms"] = float(np.mean(e))
    out["mean_sdnn_ppg_ms"] = float(np.mean(p))
    try:
        st = compute_ccc(p, e)
        out.update({
            "ccc":          float(st["ccc"]),
            "pearson_r":    float(st["pearson_r"]),
            "bias_ms":      float(st["bias"]),
            "loa_lower_ms": float(st["loa_lower"]),
            "loa_upper_ms": float(st["loa_upper"]),
            "rmse_ms":      float(st["rmse"]),
            "mae_ms":       float(st["mae"]),
        })
    except (ValueError, ZeroDivisionError):
        pass
    icc = _safe_icc(e, p)
    if icc:
        out.update({
            "icc":         float(icc["icc"]),
            "icc_ci_low":  float(icc["ci_low"]),
            "icc_ci_high": float(icc["ci_high"]),
        })
    return out


# Task Force (1996) frequency bands, in Hz — the same edges pyhrv's
# welch_psd integrates over by default. VLF starts at 0.003 Hz rather than 0
# so the DC/trend bin doesn't land in the band.
_HRV_BANDS = (("vlf", 0.003, 0.04), ("lf", 0.04, 0.15), ("hf", 0.15, 0.40))

# The NN series carries one value per beat, i.e. it is sampled unevenly at
# ~1 Hz. 4 Hz is the standard HRV resampling rate (Task Force 1996) and is
# what pyhrv used: comfortably above twice the 0.4 Hz top of the HF band.
_HRV_RESAMPLE_HZ = 4.0

# numpy 2 renamed trapz -> trapezoid; trapz still exists but deprecation-warns.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _freq_domain_metrics(nn_ms, min_beats=50):
    """Frequency-domain HRV by Welch PSD. Returns absolute VLF / LF / HF
    power (ms²) and the LF/HF ratio.

    Method (unchanged from the pyhrv.welch_psd call this replaces, which
    documented exactly these steps): place each NN interval at its cumulative
    beat time, cubic-spline resample onto a uniform 4 Hz grid, remove the
    linear trend, run ``scipy.signal.welch`` with a Hamming window, then
    integrate the PSD over the Task Force 1996 bands (VLF 0.003-0.04,
    LF 0.04-0.15, HF 0.15-0.40 Hz). Integrating a density in ms²/Hz over Hz
    gives ms², so NN intervals must be in **milliseconds** — passing seconds
    (as old/PPGanalysis.py does, almost certainly a bug) gives band powers
    1e-6× too small.

    Returns a NaN-filled dict when the series is shorter than ``min_beats``
    or too short to resample. Any unexpected failure also returns the NaN
    dict rather than raising — one bad channel must not poison a session."""
    out = {
        "vlf_power_ms2": float("nan"),
        "lf_power_ms2":  float("nan"),
        "hf_power_ms2":  float("nan"),
        "lf_hf_ratio":   float("nan"),
    }
    nn = np.asarray(nn_ms, dtype=float)
    nn = nn[np.isfinite(nn)]
    if len(nn) < min_beats:
        return out
    try:
        # Beat times: the k-th NN interval ends at the cumulative sum of the
        # intervals before it. Rebased to 0 so the grid starts at the origin.
        t_s = np.cumsum(nn) / 1000.0
        t_s -= t_s[0]
        if t_s[-1] <= 0:
            return out

        grid = np.arange(0.0, t_s[-1], 1.0 / _HRV_RESAMPLE_HZ)
        # Welch needs enough samples for at least one full segment; a
        # recording this short has no meaningful LF content anyway.
        if len(grid) < 16:
            return out
        x = interp1d(t_s, nn, kind="cubic")(grid)
        # Linear detrend: an HRV series drifts with respiration/posture, and
        # that trend would otherwise dump power into VLF and leak upward.
        x = detrend(x, type="linear")

        # nperseg=300 (75 s at 4 Hz) matches the pyhrv default this replaces
        # and resolves the 0.04 Hz LF edge; clamped so short series still
        # produce a single segment. nfft is padded for a smooth band edge.
        nperseg = int(min(len(x), 300))
        f, pxx = welch(x, fs=_HRV_RESAMPLE_HZ, window="hamming",
                       nperseg=nperseg, nfft=max(nperseg, 4096),
                       scaling="density")

        powers = {}
        for name, lo, hi in _HRV_BANDS:
            mask = (f >= lo) & (f < hi)
            powers[name] = float(_trapz(pxx[mask], f[mask])) if mask.any() else float("nan")

        out["vlf_power_ms2"] = powers["vlf"]
        out["lf_power_ms2"]  = powers["lf"]
        out["hf_power_ms2"]  = powers["hf"]
        if np.isfinite(powers["lf"]) and np.isfinite(powers["hf"]) and powers["hf"] > 0:
            out["lf_hf_ratio"] = powers["lf"] / powers["hf"]
    except Exception:
        pass
    return out


# ── Signal loader for the Plotly views ───────────────────────────────────────

def _downsample(x, y, max_points):
    """Peak-preserving (min/max) decimation.

    Plain stride subsampling aliases sharp features: an ECG QRS is a
    ~10 ms spike, so ``x[::22]`` on a 323 Hz trace samples R-peaks on
    their edges or skips them, and the webapp ECG no longer matches
    fullvis.py (which plots every sample). Splitting into buckets and
    keeping each bucket's min and max sample, in time order, retains
    R-peak height and position at any point budget. Smooth PPG is
    unaffected. x stays monotonic so the front-end interpY still works.
    """
    n = len(x)
    if max_points <= 0 or n <= max_points:
        return x, y
    n_buckets = max(1, max_points // 2)
    edges = np.linspace(0, n, n_buckets + 1, dtype=int)
    idx = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        seg = y[lo:hi]
        i_min = lo + int(np.argmin(seg))
        i_max = lo + int(np.argmax(seg))
        if i_min == i_max:
            idx.append(i_min)
        elif i_min < i_max:
            idx.append(i_min)
            idx.append(i_max)
        else:
            idx.append(i_max)
            idx.append(i_min)
    sel = np.asarray(idx, dtype=int)
    return x[sel], y[sel]


def _leads_off_spans(ts_s, leads_off):
    """Collapse the per-sample leads_off flag into (start_s, end_s) spans."""
    spans = []
    start = None
    for t, lo in zip(ts_s, leads_off):
        if lo and start is None:
            start = float(t)
        elif not lo and start is not None:
            spans.append([start, float(t)])
            start = None
    if start is not None and len(ts_s):
        spans.append([start, float(ts_s[-1])])
    return spans


def _channel_paths(session_dir):
    paths = []
    for p in sorted(glob.glob(os.path.join(session_dir, "ppg_data_ch*.csv"))):
        m = CHANNEL_RE.match(p.replace("\\", "/"))
        if m:
            paths.append((int(m.group(1)), p))
    return paths


def load_session_signals(name, max_points=5000, tail_seconds=None,
                          start_s=None, end_s=None):
    """Return ECG + each PPG channel, downsampled to ``max_points``.

    This is the data the frontend needs to reproduce ``fullvis.py``'s
    combined ECG+PPG quick-look plot (and ``vis.py``'s PPG-only one) as
    interactive Plotly traces. All channels share a common t0 — the
    earliest timestamp across ECG and PPG, in seconds since acquisition
    start, so R-peaks and pulses line up visually.

    When ``tail_seconds`` is set we trim every channel to the last
    ``tail_seconds`` of recorded data. That's how the live-view poll
    keeps payload size and Plotly render cost flat regardless of how
    long the recording has been running.

    ``start_s`` / ``end_s`` apply the user's pre-processing crop
    (seconds since the session origin t0). The crop happens before the
    bandpass, so the displayed bandpass matches the cropped window.

    Each PPG channel is returned as the *interpolated* signal — conditioned
    exactly like ``analyze_channel`` (drop non-monotonic timestamps -> cubic
    resample to a uniform grid -> rolling-median outlier scrub) — so the peak
    markers, which analyze_channel times in grid seconds, sit on the same
    trace they were detected on rather than on the raw samples.
    """
    sdir = sessions.session_path(name)
    if not os.path.isdir(sdir):
        return {"error": f"session not found: {name}"}

    ecg_payload = None
    t0_ms = None

    ecg_path = os.path.join(sdir, "ecg_data.csv")
    if os.path.isfile(ecg_path):
        ts_ms, sig, leads_off = load_ecg(ecg_path)
        if len(ts_ms):
            t0_ms = float(ts_ms[0])
            ts_ms, sig, leads_off = _tail(ts_ms, sig, leads_off, tail_seconds)
            ts_ms, (sig, leads_off) = _crop_window(
                ts_ms, [sig, leads_off], t0_ms, start_s, end_s)
            ts_s = (ts_ms - t0_ms) / 1000.0
            xs, ys = _downsample(ts_s, sig, max_points)
            # _leads_off_spans runs on the full pre-downsample ts_s,
            # but the front-end draws those spans on the downsampled
            # xs trace. The min/max bucket decimator can drop the
            # first/last sample, leaving xs[0] > ts_s[0] or
            # xs[-1] < ts_s[-1]. Clamp every span to [xs[0], xs[-1]]
            # so the red-shaded regions stay inside the visible trace,
            # and drop spans that collapse to zero or negative width
            # after clamping.
            spans = _leads_off_spans(ts_s, leads_off)
            if len(xs):
                x_lo, x_hi = float(xs[0]), float(xs[-1])
                spans = [
                    [max(s, x_lo), min(e, x_hi)]
                    for s, e in spans
                    if max(s, x_lo) <= min(e, x_hi)
                ]
            ecg_payload = {
                "name": "ecg",
                "time_s": xs.tolist(),
                "signal": ys.tolist(),
                "leads_off_spans": spans,
                "fs_hz": effective_fs(ts_ms),
                "n_samples": int(len(sig)),
            }

    channels = []
    for ch, p in _channel_paths(sdir):
        ts_ms, sig = load_ppg(p)
        if not len(ts_ms):
            continue
        if t0_ms is None:
            t0_ms = float(ts_ms[0])
        ts_ms, sig, _ = _tail(ts_ms, sig, None, tail_seconds)
        ts_ms, (sig,) = _crop_window(ts_ms, [sig], t0_ms, start_s, end_s)
        if not len(ts_ms):
            continue
        n_recorded = int(len(sig))
        # Condition exactly like analyze_channel so the displayed trace IS the
        # signal the peaks were detected on: reset implausible timestamp
        # spikes -> drop backward-stepping timestamps -> cubic resample onto
        # a uniform grid -> rolling-median outlier scrub. The peak markers
        # (timed in grid seconds by analyze_channel) then sit on this
        # interpolated trace instead of having their Y pulled from the raw
        # samples.
        ts_ms, sig = _fix_timestamp_spikes(ts_ms, sig)
        ts_ms, sig = _drop_non_monotonic(ts_ms, sig)
        fs = infer_fs(ts_ms)          # grid + filter cadence
        disp_fs = effective_fs(ts_ms)  # actual throughput shown in the UI
        grid_ts_ms, grid_sig = _resample_uniform(ts_ms, sig, fs)
        clean_sig = _remove_outliers(grid_sig)
        ts_s = (grid_ts_ms - t0_ms) / 1000.0
        xs, ys = _downsample(ts_s, clean_sig, max_points)

        bp = ppg_bandpass(clean_sig, fs)
        if bp is not None:
            xb, yb = _downsample(ts_s, bp, max_points)
            bp_x, bp_y = xb.tolist(), yb.tolist()
        else:
            bp_x, bp_y = None, None

        channels.append({
            "channel": ch,
            "name": f"ch{ch}",
            "time_s": xs.tolist(),
            "signal": ys.tolist(),
            "time_bp_s": bp_x,
            "signal_bp": bp_y,
            "fs_hz": disp_fs,
            "n_samples": n_recorded,
        })

    return {"ecg": ecg_payload, "channels": channels}


def ecg_detail(name, start_s=None, end_s=None):
    """Full-resolution ECG for the detailed-view popup (ECGvis.py-style).

    Returns *every* sample (no decimation) plus the R-peaks detect_r_peaks
    finds, on the same crop window the dashboard uses. ECG-only, so even at
    full resolution the payload is a single ~100-130k-point trace rather
    than the six the main signals endpoint ships.
    """
    sdir = sessions.session_path(name)
    ecg_path = os.path.join(sdir, "ecg_data.csv")
    if not os.path.isfile(ecg_path):
        return {"error": "session has no ecg_data.csv"}

    ts_ms, sig, leads_off = load_ecg(ecg_path)
    if not len(ts_ms):
        return {"error": "ecg_data.csv is empty"}

    t0_ms = float(ts_ms[0])
    ts_ms, (sig, leads_off) = _crop_window(ts_ms, [sig, leads_off], t0_ms, start_s, end_s)
    if not len(ts_ms):
        return {"error": "crop window selected no ECG samples"}

    ts_s = (ts_ms - t0_ms) / 1000.0
    fs = infer_fs(ts_ms)
    try:
        peaks = detect_r_peaks(sig, fs) if not np.isnan(fs) else np.array([], dtype=int)
    except Exception:
        peaks = np.array([], dtype=int)

    dur = (float(ts_ms[-1]) - float(ts_ms[0])) / 1000.0 if len(ts_ms) > 1 else 0.0
    mean_hr = (60.0 * len(peaks) / dur) if (dur > 0 and len(peaks)) else float("nan")

    return {
        "name": name,
        "time_s": ts_s.tolist(),
        "signal": sig.tolist(),
        "fs_hz": effective_fs(ts_ms),
        "n_samples": int(len(sig)),
        "duration_s": dur,
        "n_peaks": int(len(peaks)),
        "mean_hr_bpm": mean_hr,
        "peak_times_s": ts_s[peaks].tolist() if len(peaks) else [],
        "peak_values": sig[peaks].tolist() if len(peaks) else [],
        "leads_off_spans": _leads_off_spans(ts_s, leads_off),
    }


def _crop_window(ts_ms, arrays, t0_ms, start_s, end_s):
    """Keep only samples whose time (seconds since the session origin
    ``t0_ms``) falls in [start_s, end_s]. This is the pre-processing
    crop: it runs on the raw signal before the bandpass and before any
    peak detection, so the bandpass and every SQI/CCC metric are
    computed on exactly the selected window. ``arrays`` are sliced with
    the same mask; None entries pass through."""
    if start_s is None and end_s is None:
        return ts_ms, arrays
    rel_s = (ts_ms - t0_ms) / 1000.0
    mask = np.ones(len(ts_ms), dtype=bool)
    if start_s is not None:
        mask &= rel_s >= float(start_s)
    if end_s is not None:
        mask &= rel_s <= float(end_s)
    return ts_ms[mask], [a[mask] if a is not None else None for a in arrays]


def _tail(ts_ms, sig, leads_off, tail_seconds):
    """Trim arrays to the last ``tail_seconds`` of data, by timestamp."""
    if not tail_seconds or len(ts_ms) == 0:
        return ts_ms, sig, leads_off
    cutoff_ms = float(ts_ms[-1]) - float(tail_seconds) * 1000.0
    mask = ts_ms >= cutoff_ms
    return (ts_ms[mask], sig[mask],
            leads_off[mask] if leads_off is not None else None)


# ── SQI + agreement (per PPG channel against ECG) ────────────────────────────

def _safe_ssqi(sig):
    if len(sig) < 2:
        return float("nan")
    std = float(np.std(sig, ddof=0))
    if std == 0.0:
        return float("nan")
    return float(Ssqi(sig))


def _safe_ksqi(sig):
    """Pearson kurtosis of the same trace SSQI sees (same NaN guards)."""
    if len(sig) < 2:
        return float("nan")
    std = float(np.std(sig, ddof=0))
    if std == 0.0:
        return float("nan")
    return float(Ksqi(sig))


def _safe_icc(matched_rr_ms, matched_ppi_ms):
    """ICC(A,1) on matched RR (ECG) vs PPI (PPG) intervals.

    Builds the long-format frame ``sqi/ICC.py`` builds — one row per
    (beat, rater), with ``targets=beat``, ``raters="ECG"|"PPG"``,
    ``ratings=interval_ms`` — then picks the ``ICC(A,1)`` row (two-way
    mixed, single rater, **absolute agreement**). Absolute-agreement is
    the right form here because a PPG vs ECG comparison cares about
    systematic offset (PTT bias) as well as proportional similarity —
    the same reason ``compute_ccc`` includes the bias term. Pingouin's
    table also exposes ``ICC(C,1)`` (consistency, not penalising
    offset) if a future caller wants that form instead.

    Returns ``{"icc": value, "ci_low": ..., "ci_high": ..., "n": ...}``
    when there are >= 4 matched beats (pingouin needs more than two
    targets), otherwise ``None``. Any pingouin failure is swallowed so
    a single weird channel doesn't poison the whole session's analysis."""
    if _pg is None:
        return None
    n = int(min(len(matched_rr_ms), len(matched_ppi_ms)))
    if n < 4:
        return None
    try:
        beats = np.arange(n)
        long_df = pd.DataFrame({
            "beat":   np.concatenate([beats, beats]),
            "rater":  np.array(["ECG"] * n + ["PPG"] * n),
            "value":  np.concatenate([np.asarray(matched_rr_ms, dtype=float),
                                       np.asarray(matched_ppi_ms, dtype=float)]),
        })
        tbl = _pg.intraclass_corr(data=long_df, targets="beat",
                                   raters="rater", ratings="value",
                                   nan_policy="omit")
        row = tbl[tbl["Type"] == "ICC(A,1)"]
        if row.empty:
            return None
        icc_val = float(row["ICC"].values[0])
        ci = row["CI95"].values[0] if "CI95" in row.columns else None
        ci_low  = float(ci[0]) if ci is not None and len(ci) >= 2 and np.isfinite(ci[0]) else float("nan")
        ci_high = float(ci[1]) if ci is not None and len(ci) >= 2 and np.isfinite(ci[1]) else float("nan")
        return {"icc": icc_val, "ci_low": ci_low, "ci_high": ci_high, "n": n}
    except Exception:
        return None


def analyze_channel(ppg_ts_ms, ppg_sig, ecg_ts_ms, ecg_sig, t0_ms=0.0,
                    window_sec=5.0, step_sec=1.0):
    """Compute SQI + ECG agreement for one PPG channel.

    Reuses the peak detectors and the interval matcher from sqi/ccc.py
    so the numbers stay consistent between the webapp and a manual
    `python sqi/ccc.py` run on the same CSV pair.

    ``t0_ms`` is the session-wide time origin (the ECG's first
    timestamp). Peak times come back in seconds-since-t0 so the front
    end can overlay markers on the downsampled Plotly signal traces
    without doing its own unit conversion.
    """
    # Hybrid conditioning: reset implausible timestamp spikes, drop
    # backward-stepping timestamps, resample onto a uniform grid at the
    # measured fs (so the Butterworth filters run on evenly-sampled data),
    # then scrub motion spikes. clean_sig is the resampled, outlier-removed,
    # *pre-bandpass* trace — SSQI/KSQI/ZSQI run on it; the bandpass happens
    # inside the peak detector only.
    ppg_ts_ms, ppg_sig = _fix_timestamp_spikes(ppg_ts_ms, ppg_sig)
    ppg_ts_ms, ppg_sig = _drop_non_monotonic(ppg_ts_ms, ppg_sig)
    ppg_fs = infer_fs(ppg_ts_ms)
    ecg_fs = infer_fs(ecg_ts_ms)

    grid_ts_ms, grid_sig = _resample_uniform(ppg_ts_ms, ppg_sig, ppg_fs)
    clean_sig = _remove_outliers(grid_sig)

    result = {
        "ppg_fs_hz": effective_fs(ppg_ts_ms),
        "ecg_fs_hz": effective_fs(ecg_ts_ms),
        "n_ppg_samples": int(len(ppg_sig)),
        "n_ecg_samples": int(len(ecg_sig)),
        "ssqi": _safe_ssqi(clean_sig),
        "zsqi_mean": float("nan"),
        "zsqi_std": float("nan"),
        "zsqi_max": float("nan"),
        "ksqi": _safe_ksqi(clean_sig),
        "n_rr_intervals": 0,
        "n_ppi_intervals": 0,
        "n_matched_beats": 0,
        "mean_hr_bpm": float("nan"),
        "sdnn_ms": float("nan"),
        "vlf_power_ms2": float("nan"),
        "lf_power_ms2":  float("nan"),
        "hf_power_ms2":  float("nan"),
        "lf_hf_ratio":   float("nan"),
        "ppg_peak_times_s": [],
        "stats": None,
        "error": None,
    }

    # ZSQI — windowed zero-crossing rate of the mean-subtracted signal.
    try:
        if not np.isnan(ppg_fs) and len(clean_sig) > int(window_sec * ppg_fs):
            _, zcrs = windowed_zcr(clean_sig, ppg_fs, window_sec, step_sec)
            if len(zcrs):
                result["zsqi_mean"] = float(np.nanmean(zcrs))
                result["zsqi_std"] = float(np.nanstd(zcrs))
                result["zsqi_max"] = float(np.nanmax(zcrs))
    except Exception as e:
        result["error"] = f"zsqi failed: {e}"

    # Peak detection + agreement.
    try:
        ecg_peaks = detect_r_peaks(ecg_sig, ecg_fs) if not np.isnan(ecg_fs) else np.array([], dtype=int)
        # Detect on the bandpassed uniform grid. Peak times come from the
        # grid's own (evenly spaced, real-time) axis, keeping the sub-sample
        # timing the cubic resample recovered — no snap back to the nearest
        # recorded sample, which would re-quantize PPI to the raw spacing.
        ppg_peaks = _detect_ppg_peaks(clean_sig, ppg_fs) if not np.isnan(ppg_fs) else np.array([], dtype=int)

        rr_ms, rr_times = peaks_to_intervals(ecg_peaks, ecg_ts_ms) if len(ecg_peaks) else (np.array([]), np.array([]))
        ppi_ms, ppi_times = peaks_to_intervals(ppg_peaks, grid_ts_ms) if len(ppg_peaks) else (np.array([]), np.array([]))

        matched_rr, matched_ppi = match_intervals(rr_ms, rr_times, ppi_ms, ppi_times)

        result["n_rr_intervals"] = int(len(rr_ms))
        result["n_ppi_intervals"] = int(len(ppi_ms))
        result["n_matched_beats"] = int(len(matched_rr))

        # Per-channel mean HR (bpm), same count-based formula the ECG side
        # uses in analyze_session — so the two are directly comparable in
        # the agreement aggregator.
        ppg_dur_s = ((float(ppg_ts_ms[-1]) - float(ppg_ts_ms[0])) / 1000.0
                     if len(ppg_ts_ms) > 1 else 0.0)
        if ppg_dur_s > 0 and len(ppg_peaks):
            result["mean_hr_bpm"] = float(60.0 * len(ppg_peaks) / ppg_dur_s)

        # Manuscript §2.5: ectopic + outlier rejection on the PP series
        # before HRV metric extraction. [300, 2000] ms physiological range
        # gate + Karlsson 1987 ±20% local-median rule (sqi/hrv_clean.py).
        # The raw ``ppi_ms`` is preserved for the per-RR CCC pipeline so
        # the matched-beat agreement still reflects detector behaviour.
        if len(ppi_ms):
            ppi_clean_ms, _, _ = clean_intervals(ppi_ms, ppi_times)
        else:
            ppi_clean_ms = ppi_ms

        # Per-channel SDNN (ms) — sample std of the cleaned NN series; same
        # definition the ECG side uses, so the cross-session SDNN-agreement
        # aggregator can pair them directly.
        if len(ppi_clean_ms) >= 2:
            result["sdnn_ms"] = float(np.std(ppi_clean_ms, ddof=1))

        # Per-channel frequency-domain HRV (Welch PSD) on the cleaned
        # NN series. Matches manuscript §2.6 (Lomb-Scargle / Welch on NN,
        # not RR).
        ppg_fd = _freq_domain_metrics(ppi_clean_ms)
        result["vlf_power_ms2"] = ppg_fd["vlf_power_ms2"]
        result["lf_power_ms2"]  = ppg_fd["lf_power_ms2"]
        result["hf_power_ms2"]  = ppg_fd["hf_power_ms2"]
        result["lf_hf_ratio"]   = ppg_fd["lf_hf_ratio"]

        if len(ppg_peaks):
            result["ppg_peak_times_s"] = (
                (grid_ts_ms[ppg_peaks] - t0_ms) / 1000.0
            ).tolist()

        if len(matched_rr) >= 2:
            stats = compute_ccc(matched_ppi, matched_rr)
            icc_result = _safe_icc(matched_rr, matched_ppi)
            result["stats"] = {
                "ccc": float(stats["ccc"]),
                "ccc_label": ccc_label(stats["ccc"]),
                "pearson_r": float(stats["pearson_r"]),
                "mean_ppg_ppi_ms": float(stats["mean_ppg"]),
                "mean_ecg_rr_ms": float(stats["mean_ecg"]),
                "bias_ms": float(stats["bias"]),
                "loa_upper_ms": float(stats["loa_upper"]),
                "loa_lower_ms": float(stats["loa_lower"]),
                "rmse_ms": float(stats["rmse"]),
                "mae_ms": float(stats["mae"]),
                "icc": icc_result["icc"] if icc_result else float("nan"),
                "icc_ci_low": icc_result["ci_low"] if icc_result else float("nan"),
                "icc_ci_high": icc_result["ci_high"] if icc_result else float("nan"),
                "matched_rr_ms": matched_rr.tolist(),
                "matched_ppi_ms": matched_ppi.tolist(),
            }
    except Exception as e:
        prev = result["error"]
        result["error"] = f"agreement failed: {e}" if not prev else f"{prev}; agreement failed: {e}"

    return result


def analyze_session(name, start_s=None, end_s=None):
    """Walk a session_<ts>/, run analyze_channel on every PPG channel.

    ``start_s`` / ``end_s`` crop every signal to that window (seconds
    since the session origin = ECG's first timestamp) before any peak
    detection, so SSQI, ZSQI, R-peaks, RR/PPI and CCC are all computed
    on the selected window only.

    The ECG R-peak detection happens once at the session level (it's
    the same R-peak series no matter which PPG channel we're agreeing
    against) and the peak times come back in the top-level ``ecg``
    block — that lets the front-end overlay them on the ECG hero plot
    without re-deriving them per row.
    """
    sdir = sessions.session_path(name)
    if not os.path.isdir(sdir):
        return {"error": f"session not found: {name}", "results": []}

    ecg_path = os.path.join(sdir, "ecg_data.csv")
    if not os.path.isfile(ecg_path):
        return {"error": "session has no ecg_data.csv", "results": []}

    ecg_ts_ms, ecg_sig, leads_off = load_ecg(ecg_path)
    # t0 = session origin (first ECG sample), set before cropping so the
    # window is interpreted in the same absolute seconds the dashboard
    # plots use.
    t0_ms = float(ecg_ts_ms[0]) if len(ecg_ts_ms) else 0.0
    ecg_ts_ms, (ecg_sig, leads_off) = _crop_window(
        ecg_ts_ms, [ecg_sig, leads_off], t0_ms, start_s, end_s)
    ecg_fs = infer_fs(ecg_ts_ms)

    try:
        ecg_peaks = detect_r_peaks(ecg_sig, ecg_fs) if not np.isnan(ecg_fs) else np.array([], dtype=int)
    except Exception:
        ecg_peaks = np.array([], dtype=int)

    # Span of the (possibly cropped) ECG itself, not time-since-origin —
    # otherwise a window starting at 10 s inflates duration (and deflates
    # mean HR) by the uncounted lead-in.
    duration_s = ((float(ecg_ts_ms[-1]) - float(ecg_ts_ms[0])) / 1000.0) if len(ecg_ts_ms) > 1 else 0.0
    mean_hr = (60.0 * len(ecg_peaks) / duration_s) if (duration_s > 0 and len(ecg_peaks)) else float("nan")

    # ECG SDNN (ms) — same definition as Task Force 1996; sample std (ddof=1)
    # of the NN intervals over the cropped window. Used as the reference
    # value for the SDNN-agreement aggregator.
    if len(ecg_peaks) >= 2:
        ecg_rr_ms = np.diff(ecg_ts_ms[ecg_peaks])
        ecg_rr_t_s = ecg_ts_ms[ecg_peaks][1:] / 1000.0
        # Manuscript §2.5: ectopic + outlier rejection on RR before metric
        # extraction. Range gate + Karlsson 1987 ±20% rule (sqi/hrv_clean.py).
        ecg_nn_ms, _, _ = clean_intervals(ecg_rr_ms, ecg_rr_t_s)
    else:
        ecg_nn_ms = np.array([], dtype=float)
    ecg_sdnn_ms = float(np.std(ecg_nn_ms, ddof=1)) if len(ecg_nn_ms) >= 2 else float("nan")

    # ECG frequency-domain HRV via Welch PSD (matches old/PPGanalysis.py).
    ecg_fd = _freq_domain_metrics(ecg_nn_ms)

    ecg_info = {
        "fs_hz": effective_fs(ecg_ts_ms),
        "n_samples": int(len(ecg_sig)),
        "duration_s": duration_s,
        "n_peaks": int(len(ecg_peaks)),
        "mean_hr_bpm": mean_hr,
        "sdnn_ms": ecg_sdnn_ms,
        "vlf_power_ms2": ecg_fd["vlf_power_ms2"],
        "lf_power_ms2":  ecg_fd["lf_power_ms2"],
        "hf_power_ms2":  ecg_fd["hf_power_ms2"],
        "lf_hf_ratio":   ecg_fd["lf_hf_ratio"],
        "leads_off_samples": int((leads_off == 1).sum()) if len(leads_off) else 0,
        "peak_times_s": (
            ((ecg_ts_ms[ecg_peaks] - t0_ms) / 1000.0).tolist() if len(ecg_peaks) else []
        ),
    }

    participant = sessions.load_participant_metadata(name) or {}
    site_map = participant.get("channel_sites", {}) or {}

    results = []
    for ch, p in _channel_paths(sdir):
        ppg_ts_ms, ppg_sig = load_ppg(p)
        ppg_ts_ms, (ppg_sig,) = _crop_window(
            ppg_ts_ms, [ppg_sig], t0_ms, start_s, end_s)
        r = analyze_channel(ppg_ts_ms, ppg_sig, ecg_ts_ms, ecg_sig, t0_ms=t0_ms)
        r["channel"] = ch
        r["site"] = site_map.get(str(ch), "")
        r["interpretation"] = interpret_channel(r)
        results.append(r)

    return {
        "participant": participant,
        "ecg": ecg_info,
        "results": results,
        "interpretation": interpret_session(ecg_info, results),
    }


# ── Batch analysis across every session in MDPIdata ──────────────────────────

def _mean_std(values):
    """``{mean, std, n}`` over finite numeric values; both NaN if n==0."""
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(arr)),
        "std":  float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "n":    int(arr.size),
    }


# Body sites in mux-lane order (ch0 → ch4: finger, earlobe, shoulder,
# forehead, wrist) rather than alphabetical. Alphabetical put earlobe first,
# which read as arbitrary and did not line up with the per-channel tables
# above it — those are ordered by channel, so both now agree finger-first.
# A site outside the default map (a hand-edited label) sorts alphabetically
# after the known ones instead of being dropped.
_SITE_ORDER = list(dict.fromkeys(
    sessions.DEFAULT_CHANNEL_SITES[k]
    for k in sorted(sessions.DEFAULT_CHANNEL_SITES, key=int)
))


def _site_sort_key(site):
    try:
        return (0, _SITE_ORDER.index(site), "")
    except ValueError:
        return (1, 0, site)


def _aggregate_per_site(per_session):
    """Collapse per-session × per-channel rows into one row per body site.

    Groups every channel in every session by its labelled site (using the
    fixed mux-lane → site fallback from sessions._with_default_sites for
    sessions without an explicit map), then summarises SSQI, ZSQI, KSQI, CCC,
    Pearson, bias, LOA span, RMSE, MAE, and ICC3 as mean ± std across
    the group. This is the site-level table the manuscript's results
    section is built around — currently unstratified by Fitzpatrick
    because none of the MDPI sessions carry FST metadata; see the
    `fst_unavailable` flag on the response."""
    by_site = {}
    for s in per_session:
        for row in s.get("results", []):
            site = row.get("site") or "unassigned"
            by_site.setdefault(site, []).append(row)

    out = []
    for site, rows in sorted(by_site.items(), key=lambda kv: _site_sort_key(kv[0])):
        ssqis    = [r.get("ssqi") for r in rows]
        ksqis    = [r.get("ksqi") for r in rows]
        zsqi_mu  = [r.get("zsqi_mean") for r in rows]
        zsqi_sd  = [r.get("zsqi_std") for r in rows]
        cccs     = [r["stats"]["ccc"] for r in rows if r.get("stats")]
        iccs     = [r["stats"]["icc"] for r in rows if r.get("stats")]
        pearsons = [r["stats"]["pearson_r"] for r in rows if r.get("stats")]
        biases   = [r["stats"]["bias_ms"] for r in rows if r.get("stats")]
        rmses    = [r["stats"]["rmse_ms"] for r in rows if r.get("stats")]
        maes     = [r["stats"]["mae_ms"] for r in rows if r.get("stats")]
        loa_span = [r["stats"]["loa_upper_ms"] - r["stats"]["loa_lower_ms"]
                    for r in rows if r.get("stats")]
        beats    = [r.get("n_matched_beats", 0) for r in rows]

        out.append({
            "site": site,
            "n_channels": len(rows),
            "n_sessions": len({r.get("_session_name") for r in rows}),
            "ssqi":       _mean_std(ssqis),
            "zsqi_mean":  _mean_std(zsqi_mu),
            "zsqi_std":   _mean_std(zsqi_sd),
            "ksqi":       _mean_std(ksqis),
            "ccc":        _mean_std(cccs),
            "icc":        _mean_std(iccs),
            "pearson_r":  _mean_std(pearsons),
            "bias_ms":    _mean_std(biases),
            "loa_span_ms": _mean_std(loa_span),
            "rmse_ms":    _mean_std(rmses),
            "mae_ms":     _mean_std(maes),
            "matched_beats_total": int(sum(beats)),
        })
    return out


def _lfhf_agreement_per_channel(per_session):
    """Per-session LF/HF ratio (Welch PSD) paired with ECG LF/HF and
    aggregated across sessions per channel. Same shape as ``_hr_agreement_
    per_channel``; unitless ratio (no _ms / _bpm suffix on bias/LOA)."""
    by_ch = {}
    for s in per_session:
        ecg = s.get("ecg") or {}
        e_val = ecg.get("lf_hf_ratio")
        if e_val is None or not np.isfinite(e_val):
            continue
        for row in s.get("results") or []:
            ch = row.get("channel")
            p_val = row.get("lf_hf_ratio")
            if ch is None or p_val is None or not np.isfinite(p_val):
                continue
            slot = by_ch.setdefault(ch, {"ecg": [], "ppg": [], "site": row.get("site") or "unassigned"})
            slot["ecg"].append(float(e_val))
            slot["ppg"].append(float(p_val))

    rows = []
    for ch in sorted(by_ch.keys()):
        e = np.asarray(by_ch[ch]["ecg"], dtype=float)
        p = np.asarray(by_ch[ch]["ppg"], dtype=float)
        n = len(e)
        row = {
            "channel": ch,
            "site": by_ch[ch]["site"],
            "n_sessions": n,
            "mean_lfhf_ecg": float(np.mean(e)) if n else float("nan"),
            "mean_lfhf_ppg": float(np.mean(p)) if n else float("nan"),
            "ccc": float("nan"), "pearson_r": float("nan"),
            "bias": float("nan"),
            "loa_lower": float("nan"), "loa_upper": float("nan"),
            "rmse": float("nan"), "mae": float("nan"),
            "icc": float("nan"), "icc_ci_low": float("nan"), "icc_ci_high": float("nan"),
        }
        if n >= 2:
            try:
                st = compute_ccc(p, e)
                row.update({
                    "ccc":       float(st["ccc"]),
                    "pearson_r": float(st["pearson_r"]),
                    "bias":      float(st["bias"]),
                    "loa_lower": float(st["loa_lower"]),
                    "loa_upper": float(st["loa_upper"]),
                    "rmse":      float(st["rmse"]),
                    "mae":       float(st["mae"]),
                })
            except (ValueError, ZeroDivisionError):
                pass
            icc = _safe_icc(e, p)
            if icc:
                row.update({
                    "icc":         float(icc["icc"]),
                    "icc_ci_low":  float(icc["ci_low"]),
                    "icc_ci_high": float(icc["ci_high"]),
                })
        rows.append(row)
    return rows


def _sdnn_agreement_per_channel(per_session):
    """Per-session SDNN (ms) of every PPG channel paired with that session's
    ECG SDNN, aggregated across sessions per channel index. Mirrors
    ``_hr_agreement_per_channel`` exactly — same shape, ms units."""
    by_ch = {}
    for s in per_session:
        ecg = s.get("ecg") or {}
        sdnn_e = ecg.get("sdnn_ms")
        if sdnn_e is None or not np.isfinite(sdnn_e):
            continue
        for row in s.get("results") or []:
            ch = row.get("channel")
            sdnn_p = row.get("sdnn_ms")
            if ch is None or sdnn_p is None or not np.isfinite(sdnn_p):
                continue
            slot = by_ch.setdefault(ch, {"ecg": [], "ppg": [], "site": row.get("site") or "unassigned"})
            slot["ecg"].append(float(sdnn_e))
            slot["ppg"].append(float(sdnn_p))

    rows = []
    for ch in sorted(by_ch.keys()):
        e = np.asarray(by_ch[ch]["ecg"], dtype=float)
        p = np.asarray(by_ch[ch]["ppg"], dtype=float)
        n = len(e)
        row = {
            "channel": ch,
            "site": by_ch[ch]["site"],
            "n_sessions": n,
            "mean_sdnn_ecg_ms": float(np.mean(e)) if n else float("nan"),
            "mean_sdnn_ppg_ms": float(np.mean(p)) if n else float("nan"),
            "ccc": float("nan"), "pearson_r": float("nan"),
            "bias_ms": float("nan"),
            "loa_lower_ms": float("nan"), "loa_upper_ms": float("nan"),
            "rmse_ms": float("nan"), "mae_ms": float("nan"),
            "icc": float("nan"), "icc_ci_low": float("nan"), "icc_ci_high": float("nan"),
        }
        if n >= 2:
            try:
                st = compute_ccc(p, e)
                row.update({
                    "ccc":          float(st["ccc"]),
                    "pearson_r":    float(st["pearson_r"]),
                    "bias_ms":      float(st["bias"]),
                    "loa_lower_ms": float(st["loa_lower"]),
                    "loa_upper_ms": float(st["loa_upper"]),
                    "rmse_ms":      float(st["rmse"]),
                    "mae_ms":       float(st["mae"]),
                })
            except (ValueError, ZeroDivisionError):
                pass
            icc = _safe_icc(e, p)
            if icc:
                row.update({
                    "icc":         float(icc["icc"]),
                    "icc_ci_low":  float(icc["ci_low"]),
                    "icc_ci_high": float(icc["ci_high"]),
                })
        rows.append(row)
    return rows


def _hr_agreement_per_channel(per_session):
    """Per-session mean HR (bpm) of every PPG channel paired with that
    session's ECG mean HR, aggregated across sessions per channel index.

    Returns one row per channel with CCC, ICC (CI), Pearson r, Bland-Altman
    bias + LOA, RMSE/MAE on the (HR_ecg_i, HR_ppg_i) vectors. Same agreement
    pipeline the interval CCC uses; the only difference is the unit — bpm
    instead of ms — and that each session contributes one paired point,
    not one per beat. With small N (few sessions) the bias/LOA carry more
    weight than CCC/ICC, which need more sessions to stabilise."""
    by_ch = {}
    for s in per_session:
        ecg = s.get("ecg") or {}
        hr_e = ecg.get("mean_hr_bpm")
        if hr_e is None or not np.isfinite(hr_e):
            continue
        for row in s.get("results") or []:
            ch = row.get("channel")
            hr_p = row.get("mean_hr_bpm")
            if ch is None or hr_p is None or not np.isfinite(hr_p):
                continue
            slot = by_ch.setdefault(ch, {"ecg": [], "ppg": [], "site": row.get("site") or "unassigned"})
            slot["ecg"].append(float(hr_e))
            slot["ppg"].append(float(hr_p))

    rows = []
    for ch in sorted(by_ch.keys()):
        e = np.asarray(by_ch[ch]["ecg"], dtype=float)
        p = np.asarray(by_ch[ch]["ppg"], dtype=float)
        n = len(e)
        row = {
            "channel": ch,
            "site": by_ch[ch]["site"],
            "n_sessions": n,
            "mean_hr_ecg_bpm": float(np.mean(e)) if n else float("nan"),
            "mean_hr_ppg_bpm": float(np.mean(p)) if n else float("nan"),
            "ccc": float("nan"), "pearson_r": float("nan"),
            "bias_bpm": float("nan"),
            "loa_lower_bpm": float("nan"), "loa_upper_bpm": float("nan"),
            "rmse_bpm": float("nan"), "mae_bpm": float("nan"),
            "icc": float("nan"), "icc_ci_low": float("nan"), "icc_ci_high": float("nan"),
        }
        if n >= 2:
            try:
                st = compute_ccc(p, e)
                row.update({
                    "ccc":           float(st["ccc"]),
                    "pearson_r":     float(st["pearson_r"]),
                    "bias_bpm":      float(st["bias"]),
                    "loa_lower_bpm": float(st["loa_lower"]),
                    "loa_upper_bpm": float(st["loa_upper"]),
                    "rmse_bpm":      float(st["rmse"]),
                    "mae_bpm":       float(st["mae"]),
                })
            except (ValueError, ZeroDivisionError):
                pass
            icc = _safe_icc(e, p)
            if icc:
                row.update({
                    "icc":         float(icc["icc"]),
                    "icc_ci_low":  float(icc["ci_low"]),
                    "icc_ci_high": float(icc["ci_high"]),
                })
        rows.append(row)
    return rows


# Fitzpatrick skin-tone bands used to stratify the batch. Light = I-II,
# medium = III-IV, dark = V-VI (2 grades each, unlike sleepiness.py's
# I-III / IV-VI split — this is the manuscript's three-band scheme).
_SKIN_GROUPS = [
    ("light",  "I-II",   (1, 2)),
    ("medium", "III-IV", (3, 4)),
    ("dark",   "V-VI",   (5, 6)),
]


def _skin_group_of(fst):
    """Map a Fitzpatrick grade (1-6) to a skin-tone band name, or None if
    ungraded / out of range."""
    if fst is None:
        return None
    try:
        f = int(fst)
    except (TypeError, ValueError):
        return None
    for name, _, (lo, hi) in _SKIN_GROUPS:
        if lo <= f <= hi:
            return name
    return None


def _stratify_by_skin(per_session):
    """Re-run the four batch aggregations (per-site, HR, SDNN, LF/HF) within
    each Fitzpatrick skin-tone band. Sessions with no FST grade are dropped
    from every stratum. Always returns all three bands (empty tables when a
    band has no graded sessions) so the frontend renders a stable layout."""
    buckets = {name: [] for name, _, _ in _SKIN_GROUPS}
    for s in per_session:
        group = _skin_group_of((s.get("participant") or {}).get("fitzpatrick"))
        if group is not None:
            buckets[group].append(s)

    out = []
    for name, fst_range, _ in _SKIN_GROUPS:
        subset = buckets[name]
        out.append({
            "group":            name,
            "fst_range":        fst_range,
            "n_sessions":       len(subset),
            "per_site":         _aggregate_per_site(subset),
            "hr_per_channel":   _hr_agreement_per_channel(subset),
            "sdnn_per_channel": _sdnn_agreement_per_channel(subset),
            "lfhf_per_channel": _lfhf_agreement_per_channel(subset),
        })
    return out


def analyze_all_sessions(start_s=None, end_s=None, use_saved_windows=False):
    """Run analyze_session on every session_*/ folder under MDPIdata/.

    Per-session payloads keep the same shape the single-session endpoint
    returns (so the frontend can reuse the same row renderer), with one
    extra ``session_name`` field tagged on each result row so the
    per-site aggregator can count distinct sessions per site without
    re-fetching the session list.

    When ``use_saved_windows`` is True, each session is cropped to its own
    saved "best window" (sessions.get_session_window) instead of the
    global ``start_s``/``end_s`` — a session with no saved window falls
    back to its full length. This lets one batch compare session X over
    30-330 s against session Y over 100-400 s. The window actually applied
    to each session is recorded on its payload as ``crop_window``.

    ``fst_unavailable`` flags whether *any* session carried a
    Fitzpatrick grade — when False, an FST × site cross-tab would be
    empty, which is why this endpoint does not produce one. Sessions
    that error out (no ECG, parse failure) are captured in
    ``failed_sessions`` rather than aborting the whole batch."""
    summaries = sessions.list_sessions()

    per_session = []
    failed = []
    any_fst = False
    for s in summaries:
        if use_saved_windows:
            w = sessions.get_session_window(s["name"])
            s_start, s_end = w.get("start_s"), w.get("end_s")
        else:
            s_start, s_end = start_s, end_s
        try:
            r = analyze_session(s["name"], start_s=s_start, end_s=s_end)
        except Exception as e:
            failed.append({"name": s["name"], "error": str(e)})
            continue
        if r.get("error"):
            failed.append({"name": s["name"], "error": r["error"]})
            continue
        # Tag every row with its session — the aggregator needs this and
        # the frontend uses it to label the per-session table.
        for row in r.get("results", []):
            row["_session_name"] = s["name"]
        r["session_name"] = s["name"]
        r["started_at"] = s.get("started_at")
        # Record the window actually applied so the per-session table can
        # show which span each row was scored over.
        r["crop_window"] = {"start_s": s_start, "end_s": s_end}
        fst = (r.get("participant") or {}).get("fitzpatrick")
        if fst:
            any_fst = True
        per_session.append(r)

    per_site = _aggregate_per_site(per_session)
    hr_per_channel = _hr_agreement_per_channel(per_session)
    sdnn_per_channel = _sdnn_agreement_per_channel(per_session)
    lfhf_per_channel = _lfhf_agreement_per_channel(per_session)
    return {
        "n_sessions_total":     len(summaries),
        "n_sessions_analyzed":  len(per_session),
        "failed_sessions":      failed,
        "sessions":             per_session,
        "per_site":             per_site,
        "hr_per_channel":       hr_per_channel,
        "sdnn_per_channel":     sdnn_per_channel,
        "lfhf_per_channel":     lfhf_per_channel,
        "stratified_by_skin":   _stratify_by_skin(per_session),
        "fst_unavailable":      not any_fst,
        "use_saved_windows":    use_saved_windows,
        "crop_window": ({"per_session": True} if use_saved_windows
                        else {"start_s": start_s, "end_s": end_s}),
        "interpretation":       interpret_batch(per_session, per_site, failed, not any_fst),
    }


# ── Text interpretation of metrics ───────────────────────────────────────────

# Lin's CCC bins (Lin 1989) — same thresholds ``sqi.ccc.ccc_label`` uses.
# Cicchetti 1994 ICC bins are very close: <0.40 poor, 0.40-0.59 fair,
# 0.60-0.74 good, >=0.75 excellent. We re-use the CCC bins for ICC display
# because they line up with Lin's published cut-offs and the dashboard
# already colour-codes CCC the same way; this keeps the colour and the
# verbal verdict consistent. Both metrics estimate the same quantity for
# paired interval data and almost always agree to 3 decimal places.

def _grade_ccc_text(v):
    if v is None or not np.isfinite(v):
        return ("undefined", "no CCC available — too few matched beats")
    if v > 0.99: return ("almost perfect", "almost perfect agreement (Lin 1989)")
    if v > 0.95: return ("substantial",    "substantial agreement (Lin 1989)")
    if v > 0.90: return ("moderate",       "moderate agreement (Lin 1989)")
    if v > 0.50: return ("poor",           "poor agreement (Lin 1989: <0.90)")
    return            ("very poor",        "very poor agreement — interval timing does not track ECG")


def _grade_ssqi_text(v):
    """SSQI = skewness of the raw PPG. A clean, well-perfused PPG has a
    sharp systolic upstroke followed by a slower diastolic decline →
    *positive* skew (typically 0.5-2.0, Krishnan et al. 2010). Near-zero
    or negative skew means the signal is either too noisy to show the
    shape, contains motion artifact, or the optical lead is inverted."""
    if v is None or not np.isfinite(v):
        return ("undefined", "SSQI undefined (signal too short or constant)")
    if v > 1.5:   return ("very good", f"SSQI {v:+.2f} — strong positive skew, classic well-shaped PPG (Krishnan 2010 says SSQI≥1 is clean)")
    if v > 0.5:   return ("good",      f"SSQI {v:+.2f} — positive skew, pulse shape is recognisable")
    if v > -0.5:  return ("borderline",f"SSQI {v:+.2f} — near-zero skew, waveform shape is weak or noisy")
    return                ("bad",      f"SSQI {v:+.2f} — negative skew, signal likely inverted, saturated, or dominated by noise")


def _grade_ksqi_text(v):
    """KSQI = Pearson (non-excess) kurtosis of the raw PPG — the fourth
    standardised moment, so Gaussian noise scores 3.0.

    Unlike SSQI this index is *two-sided*: a clean pulsatile PPG sits near 2
    (Elgendi 2016 measured 2.06 ± 0.16 on adjudicator-rated "excellent" 60 s
    finger PPG; a pure sinusoid is 1.5), and both directions away from that
    band mean something different. Drifting up toward 3 means the pulse no
    longer dominates the amplitude distribution — the trace is noise-shaped.
    Well above 3 means heavy tails from impulsive motion / contact spikes.
    Below ~1.5 means a sub-sinusoidal, bimodal distribution: clipping, ADC
    saturation, or a squared-off waveform.

    Bands are anchored to Elgendi 2016 Table 2 (excellent-class mean ± 2 SD
    ≈ [1.74, 2.38]) plus the two analytic reference points (sinusoid 1.5,
    Gaussian 3.0). Note Elgendi ranked KSQI *last* of eight PPG SQIs for
    class discrimination, so it is reported, never used to gate."""
    if v is None or not np.isfinite(v):
        return ("undefined", "KSQI undefined (signal too short or constant)")
    if v < 1.2:
        return ("bad",       f"KSQI {v:.2f} — far below the sinusoid floor (1.5), amplitude distribution is bimodal: clipping or ADC saturation")
    if v < 1.5:
        return ("borderline",f"KSQI {v:.2f} — below the sinusoid floor, waveform is squared-off or partially clipped")
    if v <= 2.5:
        return ("very good", f"KSQI {v:.2f} — in the clean-PPG band around 2 (Elgendi 2016: 2.06±0.16 for excellent finger PPG)")
    if v <= 3.0:
        return ("good",      f"KSQI {v:.2f} — slightly peaked but still pulse-dominated")
    if v <= 5.0:
        return ("borderline",f"KSQI {v:.2f} — at/above the Gaussian value (3.0), the pulse no longer dominates the amplitude distribution")
    return                  ("bad",       f"KSQI {v:.2f} — heavy-tailed, dominated by impulsive motion / contact-loss spikes")


def _grade_zsqi_text(mu, sd):
    """ZSQI = windowed zero-crossing rate of the mean-subtracted signal.
    A clean cardiac-band PPG crosses zero roughly 2× per beat ⇒ ZSQI
    around 0.02-0.05 at typical resting HR. Larger values mean
    high-frequency noise is dominating; high variance across windows
    means contact is intermittent (jostling, motion)."""
    if mu is None or not np.isfinite(mu):
        return ("undefined", "ZSQI undefined")
    sd_part = f" (σ {sd:.3f})" if sd is not None and np.isfinite(sd) else ""
    if mu < 0.06 and (sd is None or not np.isfinite(sd) or sd < 0.02):
        return ("very good", f"ZSQI {mu:.3f}{sd_part} — low, stable zero-crossing rate, sensor contact looks consistent")
    if mu < 0.10:
        return ("good",      f"ZSQI {mu:.3f}{sd_part} — within the typical clean-PPG range")
    if mu < 0.20:
        return ("borderline",f"ZSQI {mu:.3f}{sd_part} — elevated, signal is noisy or contact is loose")
    return                  ("bad",       f"ZSQI {mu:.3f}{sd_part} — very high, dominated by noise / motion artifact")


def _grade_bias_text(bias_ms):
    """PPG peaks lag ECG R-peaks by the pulse transit time (PTT), so a
    small *positive* PPI-RR bias is expected (~50-300 ms is normal for
    forehead/finger). Huge biases mean the PPG peak detector locked onto
    the wrong feature (respiration baseline, harmonic of HR, …) and the
    matched intervals are nonsense."""
    if bias_ms is None or not np.isfinite(bias_ms):
        return ("undefined", "bias undefined")
    ab = abs(bias_ms)
    if ab < 20:    return ("very small", f"bias {bias_ms:+.1f} ms — tracks ECG with no meaningful offset")
    if ab < 100:   return ("small",      f"bias {bias_ms:+.1f} ms — within typical PTT range")
    if ab < 500:   return ("moderate",   f"bias {bias_ms:+.1f} ms — bigger than PTT alone; check for missed/double beats")
    return                ("huge",       f"bias {bias_ms:+.1f} ms — orders of magnitude larger than PTT, peak detector likely matched the wrong feature")


def interpret_channel(row):
    """Produce a structured plain-English interpretation of one PPG channel.

    Returns a dict the frontend can render directly:
        {
          "verdict":  short headline e.g. "Usable agreement, finger",
          "grade":    {good|ok|warn|bad}    — used for colour-coding,
          "lines":    [str, ...]            — one sentence per metric in order,
          "advice":   short next-step suggestion or empty,
        }
    """
    site  = row.get("site") or "unassigned"
    ch    = row.get("channel")
    matched = int(row.get("n_matched_beats") or 0)
    s     = row.get("stats") or {}
    ccc   = s.get("ccc")
    icc   = s.get("icc")
    bias  = s.get("bias_ms")
    loa_lo = s.get("loa_lower_ms")
    loa_hi = s.get("loa_upper_ms")
    rmse  = s.get("rmse_ms")
    n_ppi  = int(row.get("n_ppi_intervals") or 0)
    n_rr   = int(row.get("n_rr_intervals") or 0)

    lines = []
    ccc_word, ccc_text = _grade_ccc_text(ccc)
    ssqi_word, ssqi_text = _grade_ssqi_text(row.get("ssqi"))
    # KSQI is reported but deliberately not folded into the verdict roll-up
    # below: Elgendi 2016 ranked it last of eight PPG SQIs for discriminating
    # rated quality, so it informs the reader without moving the grade.
    _, ksqi_text = _grade_ksqi_text(row.get("ksqi"))
    zsqi_word, zsqi_text = _grade_zsqi_text(row.get("zsqi_mean"), row.get("zsqi_std"))

    lines.append(ssqi_text + ".")
    lines.append(zsqi_text + ".")
    lines.append(ksqi_text + ".")

    if matched == 0:
        lines.append(
            f"No matched beats: PPG detector found {n_ppi} PPI intervals, "
            f"ECG found {n_rr} RR intervals, but none fell within the matching window."
        )
        advice = ("Check the PPG card for this channel — sensor probably did not maintain contact, "
                  "or the detection threshold rejected the systolic peaks. Try the bandpass overlay.")
        return {"verdict": f"ch{ch} ({site}) — no usable beats", "grade": "bad",
                "lines": lines, "advice": advice}

    if matched < 30:
        lines.append(f"Only {matched} matched beats — agreement statistics are noisy.")

    # compute_ccc needs >=2 pairs; below that, stats is None even though
    # there were matched beats. Skip the CCC/ICC and bias commentary in
    # that case rather than format-stringing through None.
    if ccc is not None:
        icc_disp = f"{icc:.3f}" if icc is not None and np.isfinite(icc) else "—"
        lines.append(f"CCC {ccc:.3f}, ICC {icc_disp} — {ccc_text}.")

    bias_word, bias_text = _grade_bias_text(bias)
    if loa_lo is not None and loa_hi is not None and np.isfinite(loa_lo) and np.isfinite(loa_hi):
        loa_span = loa_hi - loa_lo
        lines.append(f"Bland-Altman {bias_text}; ±LOA span {loa_span:.0f} ms"
                     + (f", RMSE {rmse:.1f} ms." if rmse is not None and np.isfinite(rmse) else "."))
    elif bias is not None and np.isfinite(bias):
        lines.append(bias_text.capitalize() + ".")

    # Roll the metric grades into one channel-level verdict.
    if ccc is not None and ccc > 0.95 and ssqi_word in ("very good", "good"):
        grade, verdict = "good",  f"ch{ch} ({site}) — substantial agreement, ECG-grade signal"
        advice = ""
    elif ccc is not None and ccc > 0.90:
        grade, verdict = "ok",    f"ch{ch} ({site}) — moderate agreement, usable with caveats"
        advice = "Bias and LOA are inside clinically reported ranges; usable for HR/HRV summaries."
    elif ccc is not None and ccc > 0.50:
        grade, verdict = "warn",  f"ch{ch} ({site}) — poor agreement, inspect before using"
        advice = "Look at the PPG trace for ectopic beats, missed peaks, or motion bursts."
    else:
        grade, verdict = "bad",   f"ch{ch} ({site}) — no agreement"
        advice = ("Peak detector probably matched respiration or a harmonic, not systolic peaks. "
                  "Re-examine sensor placement at the " + str(site) + " site.")

    if matched < 30 and grade in ("good", "ok"):
        grade = "warn"
        advice = (advice + " Small matched-beat count — re-run on a longer window before quoting numbers.").strip()

    return {"verdict": verdict, "grade": grade, "lines": lines, "advice": advice}


def interpret_session(ecg_info, results):
    """Session-level summary text built from the per-channel interpretations.

    Returns a dict:
        {
          "headline": one-line overview,
          "ecg_text": narrative about the ECG reference,
          "channel_summaries": [{channel, site, verdict, grade, lines, advice}],
          "best_channel": channel index or None,
          "worst_channel": channel index or None,
          "notes": [str, ...]   — overall caveats / next steps,
        }
    """
    ecg_text = (f"ECG reference: {ecg_info.get('n_peaks', 0)} R-peaks, "
                f"mean HR {ecg_info.get('mean_hr_bpm', float('nan')):.0f} bpm "
                f"over {ecg_info.get('duration_s', 0):.0f} s "
                f"@ {ecg_info.get('fs_hz', float('nan')):.0f} Hz")
    leads_off = ecg_info.get("leads_off_samples") or 0
    if leads_off:
        ecg_text += f"; {leads_off} samples flagged leads-off (red-shaded on the ECG trace)"
    ecg_text += "."

    chan_summaries = []
    best, worst = None, None
    best_ccc, worst_ccc = -2.0, 2.0
    n_usable = 0
    for row in results:
        ch = row.get("channel")
        interp = row.get("interpretation") or interpret_channel(row)
        chan_summaries.append({
            "channel": ch,
            "site":    row.get("site") or "unassigned",
            **interp,
        })
        s = row.get("stats") or {}
        ccc = s.get("ccc")
        if ccc is not None and np.isfinite(ccc):
            if ccc > 0.90:
                n_usable += 1
            if ccc > best_ccc:
                best_ccc, best = ccc, ch
            if ccc < worst_ccc:
                worst_ccc, worst = ccc, ch

    if n_usable == len(results) and results:
        headline = f"All {len(results)} PPG channels reached moderate-or-better agreement with ECG."
    elif n_usable:
        headline = f"{n_usable} of {len(results)} PPG channels reached moderate-or-better agreement; the rest need inspection."
    else:
        headline = "No PPG channel reached moderate (CCC>0.90) agreement with ECG in this window."

    notes = []
    if best is not None and best_ccc > 0:
        notes.append(f"Best channel: ch{best} (CCC {best_ccc:.3f}).")
    if worst is not None and worst != best:
        notes.append(f"Weakest channel: ch{worst} (CCC {worst_ccc:.3f}).")
    if any((r.get("n_matched_beats") or 0) == 0 for r in results):
        notes.append("Channels with zero matched beats are highlighted — investigate sensor contact or peak-detection threshold.")
    if ecg_info.get("fs_hz") and ecg_info["fs_hz"] < 250:
        notes.append(f"ECG sampled at only {ecg_info['fs_hz']:.0f} Hz — R-peak timing is granular to ~{1000/ecg_info['fs_hz']:.1f} ms; expect a small RR-vs-PPI floor of noise.")

    return {
        "headline": headline,
        "ecg_text": ecg_text,
        "channel_summaries": chan_summaries,
        "best_channel":  best,
        "worst_channel": worst,
        "notes": notes,
    }


def _site_summary(site_row):
    """Plain-English single-site verdict for the per-site aggregate table."""
    site = site_row.get("site") or "unassigned"
    n    = site_row.get("n_channels") or 0
    matched_total = site_row.get("matched_beats_total") or 0
    ccc_mu = (site_row.get("ccc") or {}).get("mean")
    ssqi_mu = (site_row.get("ssqi") or {}).get("mean")
    ksqi_mu = (site_row.get("ksqi") or {}).get("mean")
    bias_mu = (site_row.get("bias_ms") or {}).get("mean")

    parts = [f"{site} (n={n} channels, Σ matched {matched_total} beats)"]
    if ccc_mu is None or not np.isfinite(ccc_mu):
        parts.append("no CCC available")
        grade = "bad"
    elif ccc_mu > 0.95:
        parts.append(f"mean CCC {ccc_mu:.3f} — substantial agreement on average")
        grade = "good"
    elif ccc_mu > 0.90:
        parts.append(f"mean CCC {ccc_mu:.3f} — moderate agreement")
        grade = "ok"
    elif ccc_mu > 0.50:
        parts.append(f"mean CCC {ccc_mu:.3f} — poor agreement, results vary by session")
        grade = "warn"
    else:
        parts.append(f"mean CCC {ccc_mu:.3f} — failing agreement at this site")
        grade = "bad"

    if ssqi_mu is not None and np.isfinite(ssqi_mu):
        if ssqi_mu > 0.5:
            parts.append(f"SSQI {ssqi_mu:+.2f} (good waveform shape)")
        elif ssqi_mu > -0.5:
            parts.append(f"SSQI {ssqi_mu:+.2f} (borderline shape)")
        else:
            parts.append(f"SSQI {ssqi_mu:+.2f} (poor/inverted shape)")
    if ksqi_mu is not None and np.isfinite(ksqi_mu):
        if 1.5 <= ksqi_mu <= 3.0:
            parts.append(f"KSQI {ksqi_mu:.2f} (pulse-dominated amplitude distribution)")
        elif ksqi_mu > 3.0:
            parts.append(f"KSQI {ksqi_mu:.2f} (heavy-tailed — impulsive artifact)")
        else:
            parts.append(f"KSQI {ksqi_mu:.2f} (sub-sinusoidal — clipping/saturation)")
    if bias_mu is not None and np.isfinite(bias_mu) and abs(bias_mu) > 500:
        parts.append(f"large mean bias {bias_mu:+.0f} ms — peak detector likely off for several channels")

    return {"site": site, "grade": grade, "text": "; ".join(parts) + "."}


def interpret_batch(per_session, per_site, failed, fst_unavailable):
    """Top-level batch interpretation for the Analyze-all view.

    Combines per-site verdicts with cohort-level commentary about which
    site performed best, which performed worst, and what's missing
    before the paper's stratified analysis can be produced."""
    n_sess = len(per_session)
    if not per_site:
        return {
            "headline": "Batch analysed but no per-site aggregation was possible.",
            "site_summaries": [],
            "notes": ["No channels had assigned sites — every channel-site mapping was empty."],
        }

    site_texts = [_site_summary(s) for s in per_site]
    best_site = max(per_site, key=lambda s: (s.get("ccc") or {}).get("mean") or -2)
    worst_site = min(per_site, key=lambda s: (s.get("ccc") or {}).get("mean") or 2)

    best_mu  = (best_site.get("ccc") or {}).get("mean")
    worst_mu = (worst_site.get("ccc") or {}).get("mean")

    headline = (
        f"Across {n_sess} session{'s' if n_sess != 1 else ''} and "
        f"{len(per_site)} body site{'s' if len(per_site) != 1 else ''}: "
    )
    if best_mu is not None and np.isfinite(best_mu):
        if best_mu > 0.90:
            headline += f"best site is {best_site['site']} (mean CCC {best_mu:.3f}, moderate-or-better)."
        elif best_mu > 0.50:
            headline += f"best site is {best_site['site']} (mean CCC {best_mu:.3f}, only poor agreement)."
        else:
            headline += "no site reached moderate agreement on average — re-examine the pipeline."
    else:
        headline += "no site produced a usable mean CCC."

    notes = []
    if best_site and worst_site and best_site["site"] != worst_site["site"]:
        notes.append(
            f"Performance gap: {best_site['site']} ({best_mu:.3f}) → "
            f"{worst_site['site']} ({worst_mu:.3f}). Expect a similar ranking in the paper's site-level table."
        )
    if failed:
        names = ", ".join(f["name"] for f in failed[:3])
        more = "" if len(failed) <= 3 else f", +{len(failed)-3} more"
        notes.append(f"Failed sessions: {len(failed)} ({names}{more}). They're skipped, not aggregated.")
    if fst_unavailable:
        notes.append(
            "Fitzpatrick stratification unavailable — no session in this batch carries an FST grade in participant.json. "
            "Save metadata on each session (left sidebar) to unlock the FST I-III vs IV-VI subgroup tables the manuscript expects."
        )
    if any(((s.get("ccc") or {}).get("mean") or -2) > 0.90 for s in per_site):
        notes.append("Sites with mean CCC > 0.90 are colour-coded green; these are the candidate good-quality sites for the results section.")

    return {
        "headline": headline,
        "site_summaries": site_texts,
        "notes": notes,
    }
