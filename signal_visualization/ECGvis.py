"""
ECGvis.py — standalone ECG R-peak view that mirrors the webapp analysis.

This loads an ``ecg_data.csv`` exactly the way ``webapp/analysis.load_ecg``
does (headerless: col0 = timestamp µs, col1 = sample, col2 = leads_off),
infers fs from the median sample interval, and runs the *same* R-peak
detector the dashboard uses — ``sqi.ccc.detect_r_peaks`` is imported, not
reimplemented, so the peaks plotted here are identical to the ones the
per-session analysis and the dashboard's ECG detail view produce.

An optional crop window (start_s / end_s, seconds since the first sample)
matches ``analysis._crop_window`` so you can reproduce a windowed run too.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from scipy.signal import find_peaks, butter, filtfilt


# Make the repo-root packages importable regardless of the working directory,
# then reuse the real detector so this view can never drift from the analysis.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Filtering ────────────────────────────────────────────────────────────────

def bandpass(signal, fs, low=0.5, high=40.0, order=2):
    """Butterworth bandpass to remove baseline wander and high-freq noise."""
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)


def lowpass(signal, fs, cutoff=8.0, order=2):
    """Butterworth lowpass for PPG (removes motion noise above 8 Hz)."""
    nyq = fs / 2.0
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, signal)

def detect_r_peaks(ecg, fs):
    """
    Detect R-peaks in a bandpass-filtered ECG signal.

    Strategy:
      - Bandpass filter (0.5-40 Hz) to remove baseline wander
      - Auto-flip if the lead is inverted (|min| > max after centering)
      - find_peaks with minimum distance = 0.4s (caps at ~150 BPM)
        and height threshold derived from the 90th-percentile amplitude

    Parameters
    ----------
    ecg : np.ndarray  -- raw ECG signal
    fs  : float       -- sampling frequency (Hz)

    Returns
    -------
    peaks : np.ndarray  -- sample indices of R-peaks
    """
    #filtered = bandpass(ecg, fs, low=0.5, high=40.0)

    filtered = ecg

    # Auto-detect polarity: if the negative excursion dominates, the lead is inverted.
    centered = filtered - np.mean(filtered)
    if np.abs(centered.min()) > centered.max():
        filtered = -filtered
    min_distance  = int(0.25 * fs)         # minimum 300 ms between beats (~200 BPM max)
    print(np.percentile(filtered, 90))
    height_thresh = np.percentile(filtered, 90) * 1.5   # 50% of 90th percentile
    peaks, _ = find_peaks(filtered, distance=min_distance, height=height_thresh)
    return peaks



def load_ecg(csv_path):
    """Identical schema/handling to webapp/analysis.load_ecg.

    Returns (ts_ms, signal, leads_off) with rows dropped where ts or sample
    failed to parse.
    """
    df = pd.read_csv(csv_path, header=None, names=["ts_us", "sample", "leads_off"],
                     on_bad_lines="skip")
    ts_us = pd.to_numeric(df["ts_us"], errors="coerce").to_numpy(dtype=float)
    sig = pd.to_numeric(df["sample"], errors="coerce").to_numpy(dtype=float)
    leads_off = pd.to_numeric(df["leads_off"], errors="coerce").to_numpy(dtype=float)
    valid = ~(np.isnan(ts_us) | np.isnan(sig))
    return ts_us[valid] / 1000.0, sig[valid], leads_off[valid].astype(int)


def infer_fs(ts_ms):
    """Median-interval sample rate, same as webapp/analysis.infer_fs."""
    if len(ts_ms) < 2:
        return float("nan")
    dt = float(np.median(np.diff(ts_ms)))
    return 1000.0 / dt if dt > 0 else float("nan")


def visualize_ecg(csv_path, start_s=None, end_s=None, title="ECG — R-peak detection"):
    """Plot the ECG with the R-peaks detect_r_peaks finds.

    Mirrors the analysis: same loader, same fs inference, same detector, and
    the same crop-window semantics (seconds since the first sample).
    """
    ts_ms, sig, leads_off = load_ecg(csv_path)
    if len(ts_ms) < 2:
        raise SystemExit(f"[ERROR] {csv_path}: fewer than 2 valid rows after parsing.")

    t0 = ts_ms[0]
    t_s_full = (ts_ms - t0) / 1000.0

    # Crop window in seconds since the first sample (analysis._crop_window).
    mask = np.ones(len(t_s_full), dtype=bool)
    if start_s is not None:
        mask &= t_s_full >= float(start_s)
    if end_s is not None:
        mask &= t_s_full <= float(end_s)
    if not mask.any():
        raise SystemExit("[ERROR] crop window selected no samples.")

    ts_ms = ts_ms[mask]
    sig = sig[mask]
    leads_off = leads_off[mask]
    t_s = (ts_ms - t0) / 1000.0

    fs = infer_fs(ts_ms)
    peaks = detect_r_peaks(sig, fs) if np.isfinite(fs) else np.array([], dtype=int)

    dur = (t_s[-1] - t_s[0]) if len(t_s) > 1 else 0.0
    hr = 60.0 * len(peaks) / dur if (dur > 0 and len(peaks)) else float("nan")

    plt.figure(figsize=(14, 5))
    plt.plot(t_s, sig, lw=0.8, color="steelblue", label="ECG")
    if len(peaks):
        plt.plot(t_s[peaks], sig[peaks], "o", ms=5, color="red",
                 zorder=5, label=f"R-peaks ({len(peaks)})")

    # Shade leads-off spans, same idea as the dashboard's red rectangles.
    in_span, span_start = False, None
    for i, lo in enumerate(leads_off):
        if lo and not in_span:
            in_span, span_start = True, t_s[i]
        elif not lo and in_span:
            plt.axvspan(span_start, t_s[i], color="red", alpha=0.12)
            in_span = False
    if in_span:
        plt.axvspan(span_start, t_s[-1], color="red", alpha=0.12)

    hr_txt = f"{hr:.0f} bpm" if np.isfinite(hr) else "—"
    plt.title(f"{title}   |   fs {fs:.0f} Hz · {len(peaks)} R-peaks · HR {hr_txt}",
              fontsize=13, fontweight="bold")
    plt.xlabel("Time (s)")
    plt.ylabel("ECG (ADC)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_file = os.path.join(repo_root, "MDPIdata", "session_20260521_162352", "ecg_data.csv")
    # Full recording. To reproduce a windowed run, pass start_s / end_s, e.g.:
    #   visualize_ecg(csv_file, start_s=20, end_s=320)
    visualize_ecg(csv_file, title="ecg", start_s=20, end_s=620)
