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
from scipy.signal import butter, filtfilt

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

# Pingouin powers ICC3 on matched RR/PPI pairs — same library sqi/ICC.py uses
# for the cross-subject form. Import lazily-guarded so a missing install
# degrades to "icc": None rather than crashing the analyze endpoint.
try:
    import pingouin as _pg
except Exception:        # pragma: no cover — package optional
    _pg = None


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


def ppg_bandpass(sig, fs, lowcut=0.6, highcut=3.3, order=2):
    """The cardiac bandpass ppgvis.py applies to PPG: zero-phase
    Butterworth, 0.6-3.3 Hz, order 2 (signal_visualization/ppgvis.py
    bandpass()). Filters the full-resolution signal so decimation
    afterwards preserves the filtered waveform.

    Returns None when the channel's fs/length can't support the filter
    (NaN fs, cutoffs not below Nyquist, or too few samples for filtfilt)
    so the front end can grey out that channel's checkbox.
    """
    if not np.isfinite(fs) or fs <= 0:
        return None
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    if not (0.0 < low < high < 1.0):
        return None
    b, a = butter(order, [low, high], btype="band")
    # filtfilt's default padlen is 3*max(len(a),len(b)); guard short signals.
    if len(sig) <= 3 * max(len(a), len(b)):
        return None
    return filtfilt(b, a, sig)


def infer_fs(ts_ms):
    if len(ts_ms) < 2:
        return float("nan")
    dt = float(np.median(np.diff(ts_ms)))
    return 1000.0 / dt if dt > 0 else float("nan")


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
        ts_ms, (sig,) = _crop_window(ts_ms, [sig], t0_ms, start_s, end_s)
        if not len(ts_ms):
            continue
        ts_s = (ts_ms - t0_ms) / 1000.0
        fs = infer_fs(ts_ms)
        xs, ys = _downsample(ts_s, sig, max_points)

        bp = ppg_bandpass(sig, fs)
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
            "fs_hz": fs,
            "n_samples": int(len(sig)),
        })

    return {"ecg": ecg_payload, "channels": channels}


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


def _aggregate_per_site(per_session):
    """Collapse per-session × per-channel rows into one row per body site.

    Groups every channel in every session by its labelled site (using the
    fixed mux-lane → site fallback from sessions._with_default_sites for
    sessions without an explicit map), then summarises SSQI, ZSQI, CCC,
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
    for site, rows in sorted(by_site.items()):
        ssqis    = [r.get("ssqi") for r in rows]
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


def analyze_all_sessions(start_s=None, end_s=None):
    """Run analyze_session on every session_*/ folder under MDPIdata/.

    Per-session payloads keep the same shape the single-session endpoint
    returns (so the frontend can reuse the same row renderer), with one
    extra ``session_name`` field tagged on each result row so the
    per-site aggregator can count distinct sessions per site without
    re-fetching the session list.

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
        try:
            r = analyze_session(s["name"], start_s=start_s, end_s=end_s)
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
        fst = (r.get("participant") or {}).get("fitzpatrick")
        if fst:
            any_fst = True
        per_session.append(r)

    per_site = _aggregate_per_site(per_session)
    return {
        "n_sessions_total":     len(summaries),
        "n_sessions_analyzed":  len(per_session),
        "failed_sessions":      failed,
        "sessions":             per_session,
        "per_site":             per_site,
        "fst_unavailable":      not any_fst,
        "crop_window": {"start_s": start_s, "end_s": end_s},
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
    zsqi_word, zsqi_text = _grade_zsqi_text(row.get("zsqi_mean"), row.get("zsqi_std"))

    lines.append(ssqi_text + ".")
    lines.append(zsqi_text + ".")

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
