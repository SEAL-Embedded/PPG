"""
Cohort Sleepiness Proxy Index (SPI) analysis for the SEAL PPG webapp.

For every session in MDPIdata/ we compute the same HRV feature vector
twice — once on the ECG-derived RR series, once on each PPG channel's
PPI series — and combine the features into a literature-derived sleepiness
proxy (SPI). Per-session aggregation across PPG channels is quality-weighted
by SSQI / ZSQI; per-site aggregation across sessions is then compared
against the ECG SPI via sqi.ccc.compute_ccc.

The point of the page is *not* to claim a validated drowsiness classifier
— the SPI weights are pre-registered from cardiac autonomic literature
(Burgess 1997, Boudreau 2013, Cabiddu 2012, Yi 2022, Brennan 2001,
Tulppo 1996). The deliverable is *can the PPG-derived HRV vector recover
the same composite sleep-onset autonomic signature that the ECG produces?*

References
----------
Burgess, H. J., et al. (1997). Sleep, 20(6), 390-398.
Boudreau, P., et al. (2013). PLoS ONE, 8(10), e76362.
Cabiddu, R., et al. (2012). Comput Math Methods Med, 2012, 768762.
Yi, C., et al. (2022). Sleep Med, 96, 6-13.
Brennan, M., et al. (2001). IEEE TBME, 48(11), 1342-1347.
Tulppo, M., et al. (1996). Am J Physiol, 271, H244-252.
Lomb, N. R. (1976). Astrophys Space Sci, 39(2), 447-462.
Richman, J. S. & Moorman, J. R. (2000). Am J Physiol, 278, H2039-H2049.
Task Force ESC/NASPE. (1996). Circulation, 93(5), 1043-1065.
"""

import math
import os
from datetime import datetime

import numpy as np
from scipy.signal import lombscargle

from sqi.ccc import (
    compute_ccc,
    detect_ppg_peaks,
    detect_r_peaks,
    peaks_to_intervals,
)

from . import analysis, sessions

# Pre-registered weights — DO NOT FIT. Their sign and magnitude come from
# the cited papers; using cohort-fitted weights here would manufacture
# agreement instead of testing it.
SPI_WEIGHTS = {
    "rmssd":     0.30,    # log(RMSSD) — parasympathetic surge at sleep onset
    "hf_nu":     0.20,    # log(HFnu)  — parasympathetic share
    "log_lf_hf": -0.25,   # log(LF/HF) — sympathovagal shift toward vagal
    "sampen":    -0.15,   # sample entropy — complexity drops at sleep onset
    "sd1_sd2":   0.10,    # Poincaré vagal share
}

LF_BAND = (0.04, 0.15)    # Hz, Task Force 1996
HF_BAND = (0.15, 0.40)    # Hz, Task Force 1996

# Tiered minimum-beat thresholds. Each HRV feature has a different
# variance floor below which the estimate is uninterpretable, so we gate
# them separately instead of using a single MIN_BEATS_FOR_HRV.
#
#   * Time-domain (SDNN, RMSSD, pNN50, Poincaré sd1/sd2): 30 beats —
#     Task Force 1996 floor for short-term HRV.
#   * Sample entropy: 100 beats — Richman & Moorman 2000 note the
#     estimator's variance is large below ~100 templates.
#   * Spectral (LF, HF, derived ratios): 150 beats — covers at least one
#     LF cycle (~25 s @ 60 BPM) with enough beats for Lomb-Scargle.
MIN_BEATS_TIMEDOMAIN = 30
MIN_BEATS_SAMPEN = 100
MIN_BEATS_SPECTRAL = 150

# Backward-compat alias — the old MIN_BEATS_FOR_HRV name is still
# referenced by other modules (analyze_session, etc.) and persists as
# the time-domain gate value.
MIN_BEATS_FOR_HRV = MIN_BEATS_TIMEDOMAIN

# Tiny floor inside log(HFnu) so log(0) doesn't poison the SPI when a
# segment has zero HF power.
EPS = 1e-6

# These caveats are deliberately surfaced on the frontend; they exist so
# the dashboard cannot be misread as a validated drowsiness classifier.
CAVEATS = [
    "SPI is a literature-derived autonomic composite, NOT a validated drowsiness classifier. "
    "Weights are pre-registered from Burgess 1997 / Boudreau 2013 / Cabiddu 2012 / Yi 2022 / "
    "Brennan 2001; they are NOT fitted to this cohort.",
    "Recordings here are ~5 minutes of seated wake — interpret SPI as 'the autonomic state "
    "of this session relative to the cohort', not 'this person is sleepy'. Sleep-onset "
    "calibration (the populations these weights come from) was performed on supine PSG, not "
    "ambulatory seated PPG.",
    "Quality weighting uses SSQI and ZSQI sigmoids on raw PPG (Krishnan 2010; Hartmann/"
    "Charlton 2019). A channel that the peak detector lost (zero matched beats) contributes "
    "the floor weight q=0.05 — it is not dropped entirely, so a bad channel can still pull a "
    "PPG SPI off the ECG reference.",
    "Frequency-domain features (LF, HF) use a Lomb-Scargle periodogram on the raw RR / PPI "
    "time series (Lomb 1976) — no resampling, no interpolation. This is robust to the "
    "missing beats short PPG segments produce, but Lomb-Scargle has different leakage "
    "behaviour than FFT-after-cubic-spline; numbers will not match an FFT pipeline beat-for-"
    "beat.",
    "Absolute LF / HF band powers are reported in ms²·Hz (amplitude² integrated over "
    "frequency), not ms² — scipy.signal.lombscargle(normalize=False) returns amplitude² in "
    "ms², which the trapezoidal integration over the band's Hz axis multiplies by a band "
    "width. The dimensionless ratios (lf_nu, hf_nu, log_lf_hf) are unaffected; they cancel "
    "the bandwidth factor and are the SPI inputs that actually drive the index.",
    "Sample entropy is computed with m=2, r=0.2·SDNN (Richman & Moorman 2000). On series of "
    "fewer than ~100 intervals the estimator's variance is large; per-session SampEn from "
    "5-minute resting recordings is reported but should not be used to compare individuals.",
]


# ── HRV feature helpers ──────────────────────────────────────────────────────

def _nan_features():
    """Sentinel returned when a beat series is too short to score.

    Keeping the keys present (just NaN-valued) lets the per-session and
    per-site aggregators iterate the feature names without branching on
    'were features computed?'.
    """
    return {k: float("nan") for k in
            ("sdnn_ms", "rmssd_ms", "pnn50",
             "lf_power", "hf_power", "lf_nu", "hf_nu",
             "lf_hf_ratio", "log_lf_hf",
             "sd1_ms", "sd2_ms", "sd1_sd2",
             "sampen")}


def _sample_entropy(series, m=2, r=None):
    """Sample entropy (Richman & Moorman 2000), implemented with numpy.

    Returns NaN on series too short for the m+1-length comparison (fewer
    than m+2 points), or when the matching template counts come out zero
    (log of zero would be -inf). Pure numpy keeps us off the nolds
    dependency hop and is fast enough for ~500-beat RR series.

    Per Richman & Moorman 2000 eq. (2), both phi(m) and phi(m+1) iterate
    over the SAME N - m templates so the ratio A / B is unbiased — the
    earlier implementation used N - m + 1 templates for phi(m) and only
    N - m for phi(m+1), inflating B and biasing SampEn upward by a small
    positive amount (most visible on perfectly periodic signals).
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n - m < 2:
        return float("nan")
    if r is None:
        r = 0.2 * float(np.std(x, ddof=1)) if n > 1 else 0.0
    if r <= 0:
        return float("nan")

    def _phi(m_):
        # Templates: rows of length m_ from x. Both phi(m) and phi(m+1)
        # iterate over the SAME N - m templates (Richman & Moorman 2000
        # eq. 2) — `m` here is the outer closure variable, not `m_`.
        if n - m < 2:
            return 0.0
        tmpl = np.array([x[i:i + m_] for i in range(n - m)])
        # Chebyshev (max-abs) distance, exclude self-matches via -1.
        # Vectorised: for every pair of templates compute max-abs diff.
        diff = np.abs(tmpl[:, None, :] - tmpl[None, :, :]).max(axis=2)
        # Exclude self-comparison (diagonal) by subtracting eye-count later.
        matches = int(np.sum(diff <= r) - len(tmpl))
        return matches

    B = _phi(m)
    A = _phi(m + 1)
    if B == 0 or A == 0:
        return float("nan")
    return float(-np.log(A / B))


def _lomb_scargle_band_power(rr_ms, rr_times_s, band):
    """Integrate the Lomb-Scargle periodogram of RR (ms) over a band (Hz).

    `rr_times_s` are the timestamps of the *end* of each RR interval (as
    returned by `peaks_to_intervals`). No resampling — Lomb-Scargle is
    designed for non-uniform sampling and the literature recommends it
    for HRV when missed beats are likely (Cabiddu 2012, Laguna 1998).

    Returns NaN if there are too few intervals, or the requested band has
    zero width / no frequency samples.

    Units. ``scipy.signal.lombscargle(normalize=False)`` returns
    amplitude² (in ms² because the input is RR in ms); integrating that
    over a frequency axis in Hz gives band power in ms²·Hz (i.e.,
    amplitude² integrated over frequency), not ms². The previous
    docstring said ms² — wrong by a factor of band width. The downstream
    SPI consumes only the dimensionless ratios (lf_nu, hf_nu, log_lf_hf)
    which cancel the band-width factor, so this unit clarification is
    documentation-only and does not change any numbers.
    """
    rr_ms = np.asarray(rr_ms, dtype=float)
    rr_times_s = np.asarray(rr_times_s, dtype=float)
    if len(rr_ms) < MIN_BEATS_SPECTRAL or len(rr_ms) != len(rr_times_s):
        return float("nan")
    f_lo, f_hi = band
    if f_hi <= f_lo:
        return float("nan")
    # 64 frequency samples across the band gives smooth band integration
    # without making the O(n*k) Lomb-Scargle blow up on multi-hundred-beat
    # series.
    n_freqs = 64
    freqs = np.linspace(f_lo, f_hi, n_freqs)
    # scipy.signal.lombscargle wants angular frequencies. Mean-subtract
    # the series — the function asks for it explicitly in its docstring,
    # else the DC component dominates the low-frequency bins.
    y = rr_ms - np.mean(rr_ms)
    try:
        pgram = lombscargle(rr_times_s, y, freqs * 2.0 * np.pi,
                            normalize=False)
    except Exception:
        return float("nan")
    # Integrate the periodogram across the band — trapezoidal rule on the
    # freqs grid gives band power in ms²·Hz (amplitude² in ms² times the
    # Hz axis of the integral). Absolute value to be safe against tiny
    # negative numerical noise on near-zero bins.
    # numpy 2.0 renames np.trapz -> np.trapezoid; fall back for older numpy.
    _trap = getattr(np, "trapezoid", None) or np.trapz
    power = float(_trap(np.abs(pgram), freqs))
    return power if np.isfinite(power) else float("nan")


def compute_hrv_features(rr_ms, rr_times_s):
    """Per-channel HRV feature vector for one beat series.

    Inputs are the RR (ECG) or PPI (PPG) intervals in milliseconds, and
    the timestamps of the *end* of each interval in seconds (as returned
    by sqi.ccc.peaks_to_intervals).

    Returns a dict of all 13 features. Each feature has a tiered minimum-
    beat gate (see MIN_BEATS_TIMEDOMAIN / _SAMPEN / _SPECTRAL): below the
    time-domain floor every feature is NaN; above the time-domain floor
    but below the sample-entropy floor sampen is NaN; below the spectral
    floor the LF/HF features are NaN (gate enforced inside
    `_lomb_scargle_band_power`). Keys are always present so downstream
    z-norm and aggregation can iterate without conditional logic.
    """
    rr = np.asarray(rr_ms, dtype=float)
    rr_t = np.asarray(rr_times_s, dtype=float)
    if len(rr) < MIN_BEATS_TIMEDOMAIN:
        return _nan_features()

    # ── Time-domain (Task Force 1996) ────────────────────────────────────
    sdnn = float(np.std(rr, ddof=1))
    diffs = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(diffs ** 2))) if len(diffs) else float("nan")
    pnn50 = float(np.mean(np.abs(diffs) > 50.0)) if len(diffs) else float("nan")

    # ── Frequency-domain (Lomb-Scargle, Lomb 1976) ───────────────────────
    # The MIN_BEATS_SPECTRAL gate lives inside `_lomb_scargle_band_power`,
    # which returns NaN for series below that threshold.
    lf_power = _lomb_scargle_band_power(rr, rr_t, LF_BAND)
    hf_power = _lomb_scargle_band_power(rr, rr_t, HF_BAND)

    total = (lf_power if np.isfinite(lf_power) else 0.0) + \
            (hf_power if np.isfinite(hf_power) else 0.0)
    if total > 0:
        lf_nu = float((lf_power if np.isfinite(lf_power) else 0.0) / total)
        hf_nu = float((hf_power if np.isfinite(hf_power) else 0.0) / total)
    else:
        lf_nu = float("nan")
        hf_nu = float("nan")

    if (np.isfinite(lf_power) and np.isfinite(hf_power) and hf_power > 0):
        lf_hf_ratio = float(lf_power / hf_power)
        log_lf_hf = float(np.log(lf_hf_ratio)) if lf_hf_ratio > 0 else float("nan")
    else:
        lf_hf_ratio = float("nan")
        log_lf_hf = float("nan")

    # ── Poincaré (Brennan 2001 closed form) ──────────────────────────────
    sd1 = rmssd / np.sqrt(2.0) if np.isfinite(rmssd) else float("nan")
    sdnn_sq = sdnn ** 2 if np.isfinite(sdnn) else float("nan")
    var_diff = float(np.var(diffs, ddof=1)) if len(diffs) > 1 else float("nan")
    sd2_sq = 2.0 * sdnn_sq - 0.5 * var_diff if \
        np.isfinite(sdnn_sq) and np.isfinite(var_diff) else float("nan")
    sd2 = float(np.sqrt(sd2_sq)) if np.isfinite(sd2_sq) and sd2_sq > 0 else float("nan")
    sd1_sd2 = (sd1 / sd2) if (np.isfinite(sd1) and np.isfinite(sd2) and sd2 > 0) else float("nan")

    # ── Complexity ────────────────────────────────────────────────────────
    # Sample entropy has its own tier — Richman & Moorman 2000 note the
    # estimator's variance is large below ~100 templates.
    if len(rr) >= MIN_BEATS_SAMPEN:
        sampen = _sample_entropy(rr, m=2, r=0.2 * sdnn if np.isfinite(sdnn) else None)
    else:
        sampen = float("nan")

    return {
        "sdnn_ms":     float(sdnn),
        "rmssd_ms":    float(rmssd) if np.isfinite(rmssd) else float("nan"),
        "pnn50":       float(pnn50) if np.isfinite(pnn50) else float("nan"),
        "lf_power":    float(lf_power) if np.isfinite(lf_power) else float("nan"),
        "hf_power":    float(hf_power) if np.isfinite(hf_power) else float("nan"),
        "lf_nu":       float(lf_nu) if np.isfinite(lf_nu) else float("nan"),
        "hf_nu":       float(hf_nu) if np.isfinite(hf_nu) else float("nan"),
        "lf_hf_ratio": float(lf_hf_ratio) if np.isfinite(lf_hf_ratio) else float("nan"),
        "log_lf_hf":   float(log_lf_hf) if np.isfinite(log_lf_hf) else float("nan"),
        "sd1_ms":      float(sd1) if np.isfinite(sd1) else float("nan"),
        "sd2_ms":      float(sd2) if np.isfinite(sd2) else float("nan"),
        "sd1_sd2":     float(sd1_sd2) if np.isfinite(sd1_sd2) else float("nan"),
        "sampen":      float(sampen) if np.isfinite(sampen) else float("nan"),
    }


# ── Cohort z-norm + SPI ──────────────────────────────────────────────────────

# Features that get a log() applied before z-norm. log_lf_hf is *already*
# logged (because computing log on a ratio is the natural form when the
# ratio can be near zero), so it's not in this list.
_LOG_FEATURES = {"rmssd_ms", "hf_nu"}


def _feature_for_zscore(feature_name, raw_features):
    """Return the cohort-stat key + the value to feed into z-norm.

    log() is applied to RMSSD and HFnu so the resulting cohort means/
    stds (and the per-session contributions) are computed in log-space,
    matching how the SPI formula consumes them. ``hf_nu`` gets an
    epsilon floor because a session with hf_power=0 would otherwise log
    to -inf and poison the cohort std.
    """
    v = raw_features.get(feature_name)
    if v is None or not np.isfinite(v):
        return float("nan")
    if feature_name == "rmssd_ms":
        return float(np.log(v)) if v > 0 else float("nan")
    if feature_name == "hf_nu":
        return float(np.log(v + EPS))
    return float(v)


def _zscore(x, mean, std):
    if not np.isfinite(x) or not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        return float("nan")
    return float((x - mean) / std)


def compute_cohort_stats(ecg_feature_dicts):
    """Cohort-level mean / std for each z-normed feature.

    The reference is the ECG feature distribution across every session;
    the same {mean, std} is applied to both ECG and PPG features so
    PPG-derived SPI is on the same scale as ECG-derived SPI and the two
    can be plotted against each other.
    """
    keys = ("rmssd_ms", "hf_nu", "log_lf_hf", "sampen", "sd1_sd2")
    out = {}
    for k in keys:
        vals = []
        for feats in ecg_feature_dicts:
            v = _feature_for_zscore(k, feats)
            if np.isfinite(v):
                vals.append(v)
        arr = np.asarray(vals, dtype=float)
        if arr.size >= 2:
            out[k] = {"mean": float(np.mean(arr)),
                      "std":  float(np.std(arr, ddof=1)),
                      "n":    int(arr.size)}
        elif arr.size == 1:
            out[k] = {"mean": float(arr[0]), "std": 0.0, "n": 1}
        else:
            out[k] = {"mean": float("nan"), "std": float("nan"), "n": 0}
    return out


def compute_sleepiness_index(features, cohort_stats):
    """Convert one feature dict into an SPI scalar + its per-feature
    z-scored contributions.

    Returns (spi, components) where ``components`` is a dict mapping
    each weighted-feature name to the value ``w * z(x)`` added into the
    SPI. NaN in any contributing feature collapses its term silently
    (set to 0) — the rest of the SPI still computes. ``spi`` is NaN
    only when every contributing feature is NaN.
    """
    def _stat(k):
        s = cohort_stats.get(k) or {}
        return s.get("mean", float("nan")), s.get("std", float("nan"))

    contributions = {}
    spi_total = 0.0
    contributed = 0
    for feat_key, w in SPI_WEIGHTS.items():
        # Map weight key -> raw feature name (most are 1-to-1 except
        # "rmssd" -> "rmssd_ms").
        raw_key = "rmssd_ms" if feat_key == "rmssd" else feat_key
        x = _feature_for_zscore(raw_key, features)
        mu, sd = _stat(raw_key)
        z = _zscore(x, mu, sd)
        if np.isfinite(z):
            contrib = float(w) * z
            contributions[feat_key] = contrib
            spi_total += contrib
            contributed += 1
        else:
            contributions[feat_key] = float("nan")
    spi = spi_total if contributed > 0 else float("nan")
    return float(spi), contributions


# ── Per-session aggregation (channel → site quality weighting) ──────────────

def _quality_weight(ssqi, zsqi_mean):
    """Quality scalar in [0.05, 1.0] from SSQI + ZSQI.

    Uses two sigmoids:
      * SSQI: pivot at 0.5, slope 2 — favours channels whose raw skew is
        positive (i.e., shape-detectable PPG).
      * ZSQI: pivot at 0.05, slope -20 (inverse via the "0.05 - z"
        argument) — favours channels with low zero-crossing rate, the
        clean-PPG range.

    Floored at 0.05 so a channel with zero matched beats still
    contributes negligibly to the weighted average instead of being
    silently dropped — that way one dead channel can't suddenly relabel
    a session's PPG SPI as if the others were the only signal.
    """
    if ssqi is None or not np.isfinite(ssqi):
        return 0.05
    s = 1.0 / (1.0 + math.exp(-2.0 * (ssqi - 0.5)))
    if zsqi_mean is None or not np.isfinite(zsqi_mean):
        z = 1.0
    else:
        z = 1.0 / (1.0 + math.exp(-20.0 * (0.05 - zsqi_mean)))
    q = max(0.05, min(1.0, s * z))
    return float(q)


def _analyze_one_session(session_name, analyzed_payload, cohort_stats):
    """Build the per-session block for the response.

    `analyzed_payload` is the cached output of analysis.analyze_session
    — we read the matched-RR series from it (so peak detection stays
    consistent with the rest of the dashboard) but compute the HRV
    feature vector here.
    """
    ecg = analyzed_payload.get("ecg") or {}
    results = analyzed_payload.get("results") or []

    # ECG feature vector — derive from the R-peak times stored in the
    # cached analysis. peak_times_s is seconds since session t0; convert
    # to ms for peaks_to_intervals (it consumes timestamps in ms).
    ecg_peak_times_s = ecg.get("peak_times_s") or []
    ecg_feats = _nan_features()
    n_rr = 0
    ecg_spi = float("nan")
    ecg_components = {}
    if len(ecg_peak_times_s) >= MIN_BEATS_FOR_HRV + 1:
        # peaks_to_intervals returns (intervals_ms, peak_times_s); but
        # we already have peak times — call np.diff directly with the
        # same semantics (timestamp of the later peak of each pair).
        pt_s = np.asarray(ecg_peak_times_s, dtype=float)
        rr_ms = np.diff(pt_s) * 1000.0
        rr_t_s = pt_s[1:]
        ecg_feats = compute_hrv_features(rr_ms, rr_t_s)
        n_rr = int(len(rr_ms))
        ecg_spi, ecg_components = compute_sleepiness_index(ecg_feats, cohort_stats)

    # Channel-level features. Use matched RR series for the per-channel
    # PPG feature vector? — no, the literature talks about PPI directly.
    # `stats.matched_ppi_ms` exists in the cached analysis but it's the
    # matched subset; we need the *full* PPI series the PPG detector
    # found. We re-detect PPG peaks here using the same functions
    # analysis.analyze_channel calls (detect_ppg_peaks /
    # peaks_to_intervals), so the SPI is computed on a beat series that
    # matches what's plotted on the per-session detail page. Reading
    # the raw PPG CSV is cheap (~7 MB per channel for 5 min @ 400 Hz).
    sdir = sessions.session_path(session_name)
    chan_rows = []
    weighted_spi_num = 0.0
    weighted_q_den = 0.0
    for r in results:
        ch = r.get("channel")
        site = r.get("site") or ""
        ssqi = r.get("ssqi")
        zsqi_mean = r.get("zsqi_mean")
        q = _quality_weight(ssqi, zsqi_mean) if r.get("stats") else 0.05

        ppg_feats = _nan_features()
        ppi_count = 0
        spi_ch = float("nan")
        comp_ch = {}
        try:
            ppg_path = os.path.join(sdir, f"ppg_data_ch{ch}.csv")
            if os.path.isfile(ppg_path):
                ppg_ts_ms, ppg_sig = analysis.load_ppg(ppg_path)
                if len(ppg_ts_ms):
                    fs = analysis.infer_fs(ppg_ts_ms)
                    if np.isfinite(fs) and len(ppg_sig) > 10:
                        peaks = detect_ppg_peaks(ppg_sig, fs)
                        if len(peaks) >= MIN_BEATS_FOR_HRV + 1:
                            ppi_ms, ppi_times_s = peaks_to_intervals(peaks, ppg_ts_ms)
                            ppg_feats = compute_hrv_features(ppi_ms, ppi_times_s)
                            ppi_count = int(len(ppi_ms))
                            spi_ch, comp_ch = compute_sleepiness_index(
                                ppg_feats, cohort_stats)
        except (IOError, OSError, ValueError):
            # Bad CSV / missing column — feature dict stays at NaN, the
            # channel still gets its floor q so the aggregation isn't
            # surprised by a missing row.
            pass

        if not np.isfinite(spi_ch):
            # No usable SPI from this channel -> floor q so it doesn't
            # silently win the weighted average against a partially-NaN
            # cohort.
            q = 0.05
        weighted_spi_num += q * spi_ch if np.isfinite(spi_ch) else 0.0
        weighted_q_den += q if np.isfinite(spi_ch) else 0.0

        chan_rows.append({
            "channel": ch,
            "site": site,
            "features": ppg_feats,
            "spi": spi_ch,
            "components": comp_ch,
            "ssqi": ssqi if (ssqi is not None and np.isfinite(ssqi)) else None,
            "zsqi_mean": zsqi_mean if (zsqi_mean is not None and np.isfinite(zsqi_mean)) else None,
            "q": q,
            "n_ppi": ppi_count,
        })

    ppg_spi_weighted = (weighted_spi_num / weighted_q_den) if weighted_q_den > 0 else float("nan")
    usable = bool(np.isfinite(ppg_spi_weighted) and np.isfinite(ecg_spi))

    participant = analyzed_payload.get("participant") or {}
    return {
        "session_name": session_name,
        "started_at": analyzed_payload.get("started_at") or sessions.parse_timestamp_from_name(session_name),
        "fitzpatrick": participant.get("fitzpatrick"),
        "participant_id": participant.get("participant_id") or "",
        "ecg": {
            "features": ecg_feats,
            "spi": ecg_spi,
            "components": ecg_components,
            "n_rr": n_rr,
        },
        "channels": chan_rows,
        "ppg_spi_weighted": ppg_spi_weighted,
        "usable": usable,
    }


# ── Per-site aggregation (sessions × channel sites) ─────────────────────────

def _aggregate_per_site(per_session):
    """For each site, build (PPG_SPI_site, ECG_SPI) pairs across sessions
    and run sqi.ccc.compute_ccc.

    The per-site PPG SPI is the quality-weighted mean of *just the
    channels at that site*, recomputed per session. So if a session has
    two finger channels (uncommon but possible), they're averaged with
    their per-channel q's; if it has none, the site has no point for
    that session.
    """
    by_site = {}
    for s in per_session:
        ecg_spi = (s.get("ecg") or {}).get("spi")
        if not np.isfinite(ecg_spi) if ecg_spi is not None else True:
            # No usable ECG SPI for this session → no point can be made
            # for any site.
            if ecg_spi is None or not np.isfinite(ecg_spi):
                continue

        # Group channel rows by their site
        ch_by_site = {}
        for ch_row in s.get("channels") or []:
            site = ch_row.get("site") or "unassigned"
            ch_by_site.setdefault(site, []).append(ch_row)

        for site, rows in ch_by_site.items():
            num, den = 0.0, 0.0
            n_ch = 0
            for ch_row in rows:
                q = float(ch_row.get("q") or 0.0)
                spi = ch_row.get("spi")
                if spi is None or not np.isfinite(spi):
                    continue
                num += q * spi
                den += q
                n_ch += 1
            if den <= 0:
                continue
            ppg_spi_site = num / den
            by_site.setdefault(site, []).append({
                "session_name": s["session_name"],
                "ecg_spi": float(ecg_spi),
                "ppg_spi": float(ppg_spi_site),
                "n_ppg_channels": n_ch,
                "fitzpatrick": s.get("fitzpatrick"),
            })

    out = []
    scatter_points = []
    for site, pts in sorted(by_site.items()):
        ecgs = np.asarray([p["ecg_spi"] for p in pts], dtype=float)
        ppgs = np.asarray([p["ppg_spi"] for p in pts], dtype=float)
        # Filter to finite pairs only.
        mask = np.isfinite(ecgs) & np.isfinite(ppgs)
        ecgs, ppgs = ecgs[mask], ppgs[mask]
        row = {
            "site": site,
            "n_sessions": int(len(ecgs)),
            "ccc_spi":  float("nan"),
            "pearson_r": float("nan"),
            "bias":     float("nan"),
            "loa_lower": float("nan"),
            "loa_upper": float("nan"),
            "rmse":     float("nan"),
            "mae":      float("nan"),
            "mean_ecg_spi": float(np.mean(ecgs)) if len(ecgs) else float("nan"),
            "mean_ppg_spi": float(np.mean(ppgs)) if len(ppgs) else float("nan"),
            "n_ppg_channels": int(sum(p["n_ppg_channels"] for p in pts)),
        }
        if len(ecgs) >= 2:
            try:
                stats = compute_ccc(ppgs, ecgs)
                row.update({
                    "ccc_spi":   float(stats["ccc"]),
                    "pearson_r": float(stats["pearson_r"]),
                    "bias":      float(stats["bias"]),
                    "loa_lower": float(stats["loa_lower"]),
                    "loa_upper": float(stats["loa_upper"]),
                    "rmse":      float(stats["rmse"]),
                    "mae":       float(stats["mae"]),
                })
            except (ValueError, ZeroDivisionError):
                pass
        out.append(row)

        for p in pts:
            scatter_points.append({
                "session_name": p["session_name"],
                "site": site,
                "ecg_spi": p["ecg_spi"],
                "ppg_spi": p["ppg_spi"],
                "fitzpatrick": p.get("fitzpatrick"),
            })

    return out, scatter_points


def _aggregate_per_fst_site(per_session):
    """Cross-tab: FST group (I-III vs IV-VI) × site → CCC of SPI.

    Returns the empty list when no session has a Fitzpatrick grade — the
    frontend renders a 'metadata missing' message in that case.
    """
    def _group_of(fst):
        if fst is None:
            return None
        try:
            f = int(fst)
        except (TypeError, ValueError):
            return None
        if 1 <= f <= 3:
            return "I-III"
        if 4 <= f <= 6:
            return "IV-VI"
        return None

    by_key = {}
    for s in per_session:
        group = _group_of(s.get("fitzpatrick"))
        if group is None:
            continue
        ecg_spi = (s.get("ecg") or {}).get("spi")
        if ecg_spi is None or not np.isfinite(ecg_spi):
            continue
        ch_by_site = {}
        for ch_row in s.get("channels") or []:
            site = ch_row.get("site") or "unassigned"
            ch_by_site.setdefault(site, []).append(ch_row)
        for site, rows in ch_by_site.items():
            num, den = 0.0, 0.0
            for ch_row in rows:
                q = float(ch_row.get("q") or 0.0)
                spi = ch_row.get("spi")
                if spi is None or not np.isfinite(spi):
                    continue
                num += q * spi
                den += q
            if den <= 0:
                continue
            by_key.setdefault((group, site), []).append({
                "ecg_spi": float(ecg_spi),
                "ppg_spi": float(num / den),
            })

    out = []
    for (group, site), pts in sorted(by_key.items()):
        ecgs = np.asarray([p["ecg_spi"] for p in pts], dtype=float)
        ppgs = np.asarray([p["ppg_spi"] for p in pts], dtype=float)
        mask = np.isfinite(ecgs) & np.isfinite(ppgs)
        ecgs, ppgs = ecgs[mask], ppgs[mask]
        row = {
            "fst_group": group,
            "site": site,
            "n_sessions": int(len(ecgs)),
            "ccc_spi": float("nan"),
        }
        if len(ecgs) >= 2:
            try:
                stats = compute_ccc(ppgs, ecgs)
                row["ccc_spi"] = float(stats["ccc"])
            except (ValueError, ZeroDivisionError):
                pass
        out.append(row)
    return out


# ── HRV feature contribution summary (per-site mean component) ──────────────

def _feature_contributions_per_site(per_session):
    """For each site, mean of the |weighted z-contribution| per feature,
    averaged across that site's channels across sessions.

    Used by the frontend's grouped-bar chart so we can see which feature
    is dominating each site's SPI. Returns a dict keyed by site, with
    the SPI feature names as inner keys.
    """
    by_site = {}
    for s in per_session:
        for ch in s.get("channels") or []:
            site = ch.get("site") or "unassigned"
            comp = ch.get("components") or {}
            by_site.setdefault(site, {k: [] for k in SPI_WEIGHTS}).update()
            for fk in SPI_WEIGHTS:
                v = comp.get(fk)
                if v is not None and np.isfinite(v):
                    by_site[site].setdefault(fk, []).append(float(v))
    out = {}
    for site, fmap in by_site.items():
        row = {}
        for fk in SPI_WEIGHTS:
            vals = fmap.get(fk) or []
            row[fk] = float(np.mean(vals)) if vals else 0.0
        out[site] = row
    return out


# ── Interpretation text ─────────────────────────────────────────────────────

def _grade_ccc(v):
    """Same bins the rest of the dashboard uses, expressed as a verbal
    grade for the SPI per-site verdict block."""
    if v is None or not np.isfinite(v):
        return "undefined"
    if v > 0.95: return "good"
    if v > 0.90: return "ok"
    if v > 0.50: return "warn"
    return "bad"


def _interpret(per_site, per_session, unusable):
    """Build the interpretation block: headline + per-site verdicts + notes."""
    finite_sites = [s for s in per_site if np.isfinite(s.get("ccc_spi", float("nan")))]
    if not finite_sites:
        return {
            "headline": "No site produced a CCC on the sleepiness proxy "
                        "(SPI) — every site had too few sessions or no usable PPG SPI.",
            "site_summaries": [],
            "notes": ["Sleepiness analysis needs ≥ 2 sessions per site with both "
                      "ECG SPI and PPG SPI usable — re-run after adding more "
                      "recordings or fixing the failed channels."],
        }

    best = max(finite_sites, key=lambda r: r["ccc_spi"])
    n_sess = len({p["session_name"] for p in per_session})
    n_sites = len(per_site)

    head = (
        f"Across {n_sess} session{'s' if n_sess != 1 else ''} and "
        f"{n_sites} site{'s' if n_sites != 1 else ''}, the best agreement "
        f"on the SPI proxy is at {best['site']} "
        f"(CCC={best['ccc_spi']:.3f}, bias={best['bias']:+.3f})."
    )
    summaries = []
    for s in per_site:
        ccc = s.get("ccc_spi")
        grade = _grade_ccc(ccc)
        if grade == "undefined":
            txt = (f"{s['site']} — only {s['n_sessions']} session{'s' if s['n_sessions'] != 1 else ''} "
                   "with paired ECG/PPG SPI; need ≥ 2 to compute CCC.")
        else:
            txt = (
                f"{s['site']} — CCC {ccc:.3f}, Pearson {s.get('pearson_r', float('nan')):.3f}, "
                f"bias {s.get('bias', 0):+.3f}, ±LOA "
                f"[{s.get('loa_lower', 0):+.3f}, {s.get('loa_upper', 0):+.3f}], "
                f"n={s['n_sessions']}."
            )
        summaries.append({"site": s["site"], "grade": grade, "text": txt})

    notes = []
    if unusable:
        names = ", ".join(u["session_name"] for u in unusable[:3])
        more = "" if len(unusable) <= 3 else f", +{len(unusable)-3} more"
        notes.append(
            f"{len(unusable)} session{'s' if len(unusable) != 1 else ''} were unusable "
            f"({names}{more}) — typically not enough beats for HRV features "
            f"(< {MIN_BEATS_FOR_HRV} matched intervals)."
        )
    if any(s.get("ccc_spi", float("nan")) is not None and np.isfinite(s.get("ccc_spi"))
           and s["ccc_spi"] < 0.50 for s in finite_sites):
        notes.append(
            "Sites where PPG SPI is anti-correlated with ECG SPI (CCC ≤ 0.50) "
            "usually mean the PPG peak detector is locking onto the wrong feature "
            "(respiration, harmonic). Inspect those channels' per-session views."
        )
    if any(s.get("fitzpatrick") for s in per_session):
        notes.append(
            "Fitzpatrick stratification available — the FST × site table below shows "
            "the per-skin-type breakdown. Single-session strata have NaN CCC by design."
        )
    else:
        notes.append(
            "Fitzpatrick metadata is missing on every session — the FST × site cross-tab "
            "below is empty until participants are scored."
        )

    return {"headline": head, "site_summaries": summaries, "notes": notes}


# ── Top-level driver ────────────────────────────────────────────────────────

def analyze_sleepiness(weighting="ssqi_zsqi", start_s=None, end_s=None):
    """Run the full SPI pipeline over every session in MDPIdata/.

    Returns the full response dict described in the page spec.
    """
    summaries = sessions.list_sessions()

    # Stage 1: per-session analysis. Reuse the existing analyze_session
    # so peak detection, SSQI, ZSQI etc. match what the per-session
    # detail page shows (and so a session that errors in the rest of the
    # dashboard errors here too).
    analyzed_pairs = []   # (session_name, analyzed_payload)
    unusable = []
    for s in summaries:
        name = s["name"]
        try:
            r = analysis.analyze_session(name, start_s=start_s, end_s=end_s)
        except Exception as e:
            unusable.append({"session_name": name, "reason": f"analyze_session failed: {e}"})
            continue
        if r.get("error"):
            unusable.append({"session_name": name, "reason": r["error"]})
            continue
        # Patch through metadata the per-session block wants.
        r["session_name"] = name
        r["started_at"] = s.get("started_at")
        analyzed_pairs.append((name, r))

    # Stage 2: ECG HRV feature vectors across the cohort → z-norm reference.
    ecg_feat_vectors = []
    for name, payload in analyzed_pairs:
        ecg = payload.get("ecg") or {}
        ecg_peak_times_s = ecg.get("peak_times_s") or []
        if len(ecg_peak_times_s) >= MIN_BEATS_FOR_HRV + 1:
            pt_s = np.asarray(ecg_peak_times_s, dtype=float)
            rr_ms = np.diff(pt_s) * 1000.0
            rr_t_s = pt_s[1:]
            ecg_feat_vectors.append(compute_hrv_features(rr_ms, rr_t_s))
    cohort_stats = compute_cohort_stats(ecg_feat_vectors)

    # Stage 3: per-session SPI (uses cohort stats from stage 2).
    per_session = []
    for name, payload in analyzed_pairs:
        block = _analyze_one_session(name, payload, cohort_stats)
        if not block["usable"]:
            reason = ("no usable ECG SPI (insufficient RR beats)" if
                      not np.isfinite(block["ecg"]["spi"])
                      else "no usable PPG SPI from any channel")
            unusable.append({"session_name": name, "reason": reason})
        per_session.append(block)

    # Stage 4: per-site aggregation + cross-tab + interpretation.
    per_site, scatter = _aggregate_per_site(per_session)
    per_fst_site = _aggregate_per_fst_site(per_session)
    contributions = _feature_contributions_per_site(per_session)
    interp = _interpret(per_site, per_session, unusable)

    return {
        "weighting": weighting,
        "crop_window": {"start_s": start_s, "end_s": end_s},
        "config": {
            "weights": SPI_WEIGHTS,
            "freq_method": "lomb-scargle",
            "lf_band_hz": list(LF_BAND),
            "hf_band_hz": list(HF_BAND),
            "min_beats_for_hrv": MIN_BEATS_FOR_HRV,
        },
        "cohort_stats": cohort_stats,
        "per_session": per_session,
        "per_site": per_site,
        "per_fst_site": per_fst_site,
        "feature_contributions_per_site": contributions,
        "unusable_sessions": unusable,
        "scatter_points": scatter,
        "caveats": list(CAVEATS),
        "interpretation": interp,
        "n_sessions_total": len(summaries),
        "n_sessions_usable": len([s for s in per_session if s["usable"]]),
    }
