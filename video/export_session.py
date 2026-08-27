"""Export session_20260715_164725 to a compact JSON for the Remotion video.

Signals are resampled onto uniform grids so the renderer can index by time
(idx = t * rate) instead of searching timestamps. PPG is bandpassed with the
same canonical 0.6-3.3 Hz filter the peak detector runs on, so what the video
shows is what the detector saw.
"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from scipy.signal import detrend, welch

from sqi.hrv_clean import clean_intervals
from webapp import analysis, sessions

NAME = "session_20260715_164725"
CLIP = 1.5
ECG_RATE = 150.0   # Hz on the render grid — enough to keep the QRS sharp
PPG_RATE = 50.0    # Hz — pulse waves are smooth, 50 is plenty

# Rolling HRV. SDNN needs less history than a frequency split does: the LF band
# starts at 0.04 Hz (25 s per cycle), so a 120 s window buys ~5 LF cycles, which
# is the shortest defensible window. Both are short-term estimates, not the
# Task Force 5-minute standard — the on-screen labels say so.
HRV_STEP_S = 2.0
SDNN_WINDOW_S = 60.0
LFHF_WINDOW_S = 120.0
LF_BAND = (0.04, 0.15)
HF_BAND = (0.15, 0.40)
RESAMPLE_HZ = 4.0

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public", "session.json")


def _nn_series(peaks_s):
    """Cleaned NN intervals (ms) and their timestamps (s) for a beat train."""
    p = np.asarray(peaks_s, dtype=float)
    if len(p) < 3:
        return np.array([]), np.array([])
    ibi_ms = np.diff(p) * 1000.0
    t_s = p[1:]
    nn, tt, _ = clean_intervals(ibi_ms, t_s)
    return nn, tt


def _lf_hf(nn_ms, t_s):
    """LF/HF from the NN tachogram: interpolate to 4 Hz, detrend, Welch PSD."""
    if len(nn_ms) < 20:
        return None
    span = t_s[-1] - t_s[0]
    if span < 30.0:
        return None
    grid = np.arange(t_s[0], t_s[-1], 1.0 / RESAMPLE_HZ)
    if len(grid) < 32:
        return None
    tach = np.interp(grid, t_s, nn_ms)
    tach = detrend(tach, type="linear")
    nperseg = int(min(len(tach), 256))
    f, pxx = welch(tach, fs=RESAMPLE_HZ, nperseg=nperseg)
    lf = float(np.trapz(pxx[(f >= LF_BAND[0]) & (f < LF_BAND[1])],
                        f[(f >= LF_BAND[0]) & (f < LF_BAND[1])]))
    hf = float(np.trapz(pxx[(f >= HF_BAND[0]) & (f < HF_BAND[1])],
                        f[(f >= HF_BAND[0]) & (f < HF_BAND[1])]))
    if hf <= 0 or lf <= 0:
        return None
    return lf / hf


HR_LOOKBACK_S = 20.0


def rolling_hrv(peaks_s, grid_t):
    """Trailing-window HR (bpm), SDNN (ms) and LF/HF for each point on grid_t.

    Precomputed here rather than in the renderer: doing it per frame meant
    ~1,250 interval scans every frame, which OOM-crashed the render browser.
    """
    nn, tt = _nn_series(peaks_s)
    p = np.asarray(peaks_s, dtype=float)

    # HR uses the raw beat train (not the NN-cleaned one) so it matches the
    # per-row BPM readout the renderer computes from the same peaks.
    hr_out = []
    for t in grid_t:
        win = p[(p > t - HR_LOOKBACK_S) & (p <= t)]
        if len(win) < 4:
            hr_out.append(None)
            continue
        med = float(np.median(np.diff(win)))
        hr_out.append(round(60.0 / med, 2) if med > 0 else None)

    sdnn_out, lfhf_out = [], []
    for t in grid_t:
        if len(nn) == 0:
            sdnn_out.append(None)
            lfhf_out.append(None)
            continue
        m = (tt > t - SDNN_WINDOW_S) & (tt <= t)
        seg = nn[m]
        sdnn_out.append(round(float(np.std(seg, ddof=1)), 2) if len(seg) >= 10 else None)

        m2 = (tt > t - LFHF_WINDOW_S) & (tt <= t)
        ratio = _lf_hf(nn[m2], tt[m2]) if np.count_nonzero(m2) >= 20 else None
        lfhf_out.append(round(ratio, 3) if ratio is not None else None)
    return hr_out, sdnn_out, lfhf_out

sdir = sessions.session_path(NAME)
r = analysis.analyze_session(NAME)
meta = r.get("participant") or {}
ecg_info = r["ecg"]

# ── ECG ──────────────────────────────────────────────────────────────────
ecg_ts_ms, ecg_sig, leads_off = analysis.load_ecg(os.path.join(sdir, "ecg_data.csv"))
t0 = float(ecg_ts_ms[0])
ecg_t = (ecg_ts_ms - t0) / 1000.0
duration = float(ecg_t[-1])

grid_e = np.arange(0.0, duration, 1.0 / ECG_RATE)
ecg_r = np.interp(grid_e, ecg_t, ecg_sig)

# Robust scale to ~[-1,1]; ECG baseline wander removed with a slow moving median
# Scale off the 99.8th percentile of |deviation| so the R-peak tips land near
# 1.15 and survive the clip intact — a 99.5th-percentile span flattens them.
base = float(np.percentile(ecg_r, 50))
scale = max(float(np.percentile(np.abs(ecg_r - base), 99.8)), 1e-9)
ecg_n = np.clip((ecg_r - base) / scale * 1.15, -CLIP, CLIP)

lo_spans = analysis._leads_off_spans(ecg_t, leads_off)

out = {
    "session": NAME,
    # Participant name is masked here, at the source, so the name never
    # reaches the JSON that ships with the video bundle.
    "participant_id": "*" * len(str(meta.get("participant_id") or "")) or "—",
    "fitzpatrick": meta.get("fitzpatrick"),
    "duration_s": round(duration, 3),
    "ecg": {
        "rate": ECG_RATE,
        "fs_native": round(float(ecg_info["fs_hz"]), 1),
        "n_native": int(ecg_info["n_samples"]),
        "mean_hr_bpm": round(float(ecg_info["mean_hr_bpm"]), 1),
        "values": [round(float(v), 3) for v in ecg_n],
        "peaks_s": [round(float(t), 3) for t in ecg_info["peak_times_s"]],
        "leads_off": [[round(float(a), 2), round(float(b), 2)] for a, b in lo_spans],
    },
    "channels": [],
}

# ── PPG channels ─────────────────────────────────────────────────────────
for row in r["results"]:
    ch = row["channel"]
    path = os.path.join(sdir, f"ppg_data_ch{ch}.csv")
    ts_ms, sig = analysis.load_ppg(path)
    t = (ts_ms - t0) / 1000.0
    fs = analysis.infer_fs(ts_ms)

    # Same band the detector uses, so the trace and the peak markers agree.
    filt = analysis.ppg_bandpass(sig, fs)

    grid = np.arange(0.0, duration, 1.0 / PPG_RATE)
    vals = np.interp(grid, t, filt, left=np.nan, right=np.nan)
    finite = vals[np.isfinite(vals)]
    # 95th percentile keeps low-amplitude sites (forehead) visibly tall while
    # the clip still contains the occasional motion artefact.
    scale = max(float(np.percentile(np.abs(finite), 95.0)), 1e-9) if len(finite) else 1.0
    vals = np.clip(vals / scale * 1.05, -CLIP, CLIP)
    vals = np.nan_to_num(vals, nan=0.0)

    out["channels"].append({
        "ch": ch,
        "site": row["site"],
        "rate": PPG_RATE,
        "fs_native": round(float(row["ppg_fs_hz"]), 1),
        "n_native": int(row["n_ppg_samples"]),
        "mean_hr_bpm": round(float(row["mean_hr_bpm"]), 1),
        "ssqi": round(float(row["ssqi"]), 3),
        "ccc": round(float((row["stats"] or {}).get("ccc") or float("nan")), 3),
        "values": [round(float(v), 3) for v in vals],
        "peaks_s": [round(float(x), 3) for x in (row.get("ppg_peak_times_s") or [])],
    })

# ── Rolling HRV for every stream ─────────────────────────────────────────
hrv_grid = np.arange(0.0, duration, HRV_STEP_S)
hrv_series = []
for key, peaks in [("ECG", out["ecg"]["peaks_s"])] + [
    (c["site"], c["peaks_s"]) for c in out["channels"]
]:
    hr, sdnn, lfhf = rolling_hrv(peaks, hrv_grid)
    hrv_series.append({"key": key, "hr": hr, "sdnn": sdnn, "lfhf": lfhf})
    ok_s = sum(v is not None for v in sdnn)
    ok_l = sum(v is not None for v in lfhf)
    print(f"  hrv {key:<9} sdnn {ok_s}/{len(sdnn)} pts   lf/hf {ok_l}/{len(lfhf)} pts")

out["hrv"] = {
    "t_s": [round(float(t), 1) for t in hrv_grid],
    "sdnn_window_s": SDNN_WINDOW_S,
    "lfhf_window_s": LFHF_WINDOW_S,
    "series": hrv_series,
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(out, f, separators=(",", ":"))

mb = os.path.getsize(OUT) / 1e6
print(f"wrote {OUT}  ({mb:.2f} MB)")
print(f"duration {duration:.1f}s  ECG {len(ecg_n)} pts @ {ECG_RATE}Hz  {len(out['ecg']['peaks_s'])} R-peaks")
print(f"leads_off spans: {len(lo_spans)}")
for c in out["channels"]:
    print(f"  ch{c['ch']:<2}{c['site']:<10} {len(c['values'])} pts  {len(c['peaks_s'])} peaks  "
          f"HR={c['mean_hr_bpm']}  SSQI={c['ssqi']}  CCC={c['ccc']}")
