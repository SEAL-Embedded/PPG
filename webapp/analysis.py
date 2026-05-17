"""
Per-session analysis driver for the SEAL PPG webapp.

For one ``session_<ts>/`` folder we run:

    SSQI    skewness of the raw PPG    (sqi.SSQI_algorithm.Ssqi)
    ZSQI    windowed zero-crossing     (sqi.zcr_sqi.windowed_zcr)
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

import numpy as np
import pandas as pd

from sqi.ccc import (
    bandpass,
    ccc_label,
    compute_ccc,
    detect_ppg_peaks,
    detect_r_peaks,
    lowpass,
    match_intervals,
    peaks_to_intervals,
)
from sqi.SSQI_algorithm import Ssqi
from sqi.zcr_sqi import windowed_zcr

from . import sessions


CHANNEL_RE = re.compile(r".*ppg_data_ch(\d+)\.csv$")


# ── Loaders (schema matches PPG_ECG_Full_Unpacking.py output) ────────────────

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
    return ts_us[valid] / 1000.0, sig[valid]   # return ms, sample


def load_ecg(path):
    """``ecg_data.csv`` — col0=ts_us, col1=sample, col2=leads_off."""
    df = pd.read_csv(path, header=None, names=["ts_us", "sample", "leads_off"],
                     on_bad_lines="skip")
    ts_us = pd.to_numeric(df["ts_us"], errors="coerce").to_numpy(dtype=float)
    sig = pd.to_numeric(df["sample"], errors="coerce").to_numpy(dtype=float)
    leads_off = pd.to_numeric(df["leads_off"], errors="coerce").to_numpy(dtype=float)
    valid = ~(np.isnan(ts_us) | np.isnan(sig))
    return ts_us[valid] / 1000.0, sig[valid], leads_off[valid].astype(int)


def infer_fs(ts_ms):
    if len(ts_ms) < 2:
        return float("nan")
    dt = float(np.median(np.diff(ts_ms)))
    return 1000.0 / dt if dt > 0 else float("nan")


# ── Signal loader for the Plotly views ───────────────────────────────────────

def _downsample(x, y, max_points):
    if max_points <= 0 or len(x) <= max_points:
        return x, y
    stride = max(1, len(x) // max_points)
    return x[::stride], y[::stride]


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


def load_session_signals(name, max_points=5000, tail_seconds=None):
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
            ts_s = (ts_ms - t0_ms) / 1000.0
            xs, ys = _downsample(ts_s, sig, max_points)
            ecg_payload = {
                "name": "ecg",
                "time_s": xs.tolist(),
                "signal": ys.tolist(),
                "leads_off_spans": _leads_off_spans(ts_s, leads_off),
                "fs_hz": infer_fs(ts_ms),
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
        ts_s = (ts_ms - t0_ms) / 1000.0
        xs, ys = _downsample(ts_s, sig, max_points)
        channels.append({
            "channel": ch,
            "name": f"ch{ch}",
            "time_s": xs.tolist(),
            "signal": ys.tolist(),
            "fs_hz": infer_fs(ts_ms),
            "n_samples": int(len(sig)),
        })

    return {"ecg": ecg_payload, "channels": channels}


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
    ppg_fs = infer_fs(ppg_ts_ms)
    ecg_fs = infer_fs(ecg_ts_ms)

    result = {
        "ppg_fs_hz": ppg_fs,
        "ecg_fs_hz": ecg_fs,
        "n_ppg_samples": int(len(ppg_sig)),
        "n_ecg_samples": int(len(ecg_sig)),
        "ssqi": _safe_ssqi(ppg_sig),
        "zsqi_mean": float("nan"),
        "zsqi_std": float("nan"),
        "zsqi_max": float("nan"),
        "n_rr_intervals": 0,
        "n_ppi_intervals": 0,
        "n_matched_beats": 0,
        "ppg_peak_times_s": [],
        "stats": None,
        "error": None,
    }

    # ZSQI — windowed zero-crossing rate of the mean-subtracted signal.
    try:
        if not np.isnan(ppg_fs) and len(ppg_sig) > int(window_sec * ppg_fs):
            _, zcrs = windowed_zcr(ppg_sig, ppg_fs, window_sec, step_sec)
            if len(zcrs):
                result["zsqi_mean"] = float(np.nanmean(zcrs))
                result["zsqi_std"] = float(np.nanstd(zcrs))
                result["zsqi_max"] = float(np.nanmax(zcrs))
    except Exception as e:
        result["error"] = f"zsqi failed: {e}"

    # Peak detection + agreement.
    try:
        ecg_peaks = detect_r_peaks(ecg_sig, ecg_fs) if not np.isnan(ecg_fs) else np.array([], dtype=int)
        ppg_peaks = detect_ppg_peaks(ppg_sig, ppg_fs) if not np.isnan(ppg_fs) else np.array([], dtype=int)

        rr_ms, rr_times = peaks_to_intervals(ecg_peaks, ecg_ts_ms) if len(ecg_peaks) else (np.array([]), np.array([]))
        ppi_ms, ppi_times = peaks_to_intervals(ppg_peaks, ppg_ts_ms) if len(ppg_peaks) else (np.array([]), np.array([]))

        matched_rr, matched_ppi = match_intervals(rr_ms, rr_times, ppi_ms, ppi_times)

        result["n_rr_intervals"] = int(len(rr_ms))
        result["n_ppi_intervals"] = int(len(ppi_ms))
        result["n_matched_beats"] = int(len(matched_rr))

        if len(ppg_peaks):
            result["ppg_peak_times_s"] = (
                (ppg_ts_ms[ppg_peaks] - t0_ms) / 1000.0
            ).tolist()

        if len(matched_rr) >= 2:
            stats = compute_ccc(matched_ppi, matched_rr)
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
                "matched_rr_ms": matched_rr.tolist(),
                "matched_ppi_ms": matched_ppi.tolist(),
            }
    except Exception as e:
        prev = result["error"]
        result["error"] = f"agreement failed: {e}" if not prev else f"{prev}; agreement failed: {e}"

    return result


def analyze_session(name):
    """Walk a session_<ts>/, run analyze_channel on every PPG channel.

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
    t0_ms = float(ecg_ts_ms[0]) if len(ecg_ts_ms) else 0.0
    ecg_fs = infer_fs(ecg_ts_ms)

    try:
        ecg_peaks = detect_r_peaks(ecg_sig, ecg_fs) if not np.isnan(ecg_fs) else np.array([], dtype=int)
    except Exception:
        ecg_peaks = np.array([], dtype=int)

    duration_s = ((float(ecg_ts_ms[-1]) - t0_ms) / 1000.0) if len(ecg_ts_ms) > 1 else 0.0
    mean_hr = (60.0 * len(ecg_peaks) / duration_s) if (duration_s > 0 and len(ecg_peaks)) else float("nan")

    ecg_info = {
        "fs_hz": ecg_fs,
        "n_samples": int(len(ecg_sig)),
        "duration_s": duration_s,
        "n_peaks": int(len(ecg_peaks)),
        "mean_hr_bpm": mean_hr,
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
        r = analyze_channel(ppg_ts_ms, ppg_sig, ecg_ts_ms, ecg_sig, t0_ms=t0_ms)
        r["channel"] = ch
        r["site"] = site_map.get(str(ch), "")
        results.append(r)

    return {"participant": participant, "ecg": ecg_info, "results": results}
