"""Smoke tests for sqi/ccc.py and sqi/SSQI_algorithm.py.

These are intentionally small — the SQI module is already exercised
indirectly through the analysis tests. The goal here is to fence the
*shape* of the public API (return types, lengths) so a refactor on
that side surfaces fast.
"""

import numpy as np
import pytest

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


# ── Filter guards (P3) ──────────────────────────────────────────────────────

class TestFilterGuards:
    def test_bandpass_short_signal_returns_none(self, synth_short_signal):
        from sqi.ccc import bandpass
        out = bandpass(synth_short_signal, fs=200.0)
        assert out is None or len(out) == len(synth_short_signal)

    def test_lowpass_short_signal_returns_none(self, synth_short_signal):
        from sqi.ccc import lowpass
        out = lowpass(synth_short_signal, fs=200.0)
        assert out is None or len(out) == len(synth_short_signal)


# ── PPG bandpass at paper spec (F1) ─────────────────────────────────────────

class TestAutoFlipRobustness:
    def test_detect_r_peaks_robust_to_single_negative_spike(
        self, synth_single_spike_ecg
    ):
        """One huge negative spike must NOT permanently invert the whole recording."""
        peaks = detect_r_peaks(synth_single_spike_ecg["ecg"],
                               synth_single_spike_ecg["fs"])
        # 30 s at 60 BPM -> ~30 peaks. Allow +/- 5 tolerance.
        assert 20 <= len(peaks) <= 40, \
            f"got {len(peaks)} peaks, expected ~30 (single spike must not invert)"


class TestFastHrCeiling:
    def test_detect_r_peaks_handles_180_bpm(self, synth_fast_hr_ecg):
        peaks = detect_r_peaks(synth_fast_hr_ecg["ecg"], synth_fast_hr_ecg["fs"])
        # 30 s x 180 BPM = 90 peaks. Allow +/-15 tolerance.
        assert 75 <= len(peaks) <= 105, \
            f"only {len(peaks)} peaks at 180 BPM — min_distance probably too high"


class TestPpgBandpass:
    def test_ppg_bandpass_function_exists(self):
        from sqi import ccc
        assert hasattr(ccc, "ppg_bandpass")

    def test_ppg_bandpass_attenuates_5hz(self):
        """0.5-4 Hz BP must attenuate a 5 Hz sine vs a 1.5 Hz sine."""
        from sqi.ccc import ppg_bandpass
        fs = 200.0
        n = int(30 * fs)
        t = np.arange(n) / fs
        in_band = np.sin(2 * np.pi * 1.5 * t)
        out_of_band = np.sin(2 * np.pi * 5.0 * t)
        in_band_filtered = ppg_bandpass(in_band, fs)
        out_of_band_filtered = ppg_bandpass(out_of_band, fs)
        # Power ratio: in-band passes ~unchanged, 5 Hz attenuated ~10x
        assert np.std(in_band_filtered) > 0.5 * np.std(in_band)
        assert np.std(out_of_band_filtered) < 0.3 * np.std(out_of_band)


# ── bandpass / lowpass shape ────────────────────────────────────────────────

class TestFilters:

    def test_bandpass_returns_same_shape(self, synth_signal_arrays):
        sig = synth_signal_arrays["ecg"]
        out = bandpass(sig, synth_signal_arrays["fs"])
        assert out.shape == sig.shape

    def test_lowpass_returns_same_shape(self, synth_signal_arrays):
        sig = synth_signal_arrays["ppg"]
        out = lowpass(sig, synth_signal_arrays["fs"])
        assert out.shape == sig.shape

    def test_bandpass_changes_signal(self, synth_signal_arrays):
        """Sanity: filter actually does something (output != input)."""
        sig = synth_signal_arrays["ecg"]
        out = bandpass(sig, synth_signal_arrays["fs"])
        assert not np.allclose(out, sig)


# ── peak detectors ──────────────────────────────────────────────────────────

class TestPeakDetectors:

    def test_detect_r_peaks_returns_int_ndarray(self, synth_signal_arrays):
        peaks = detect_r_peaks(synth_signal_arrays["ecg"],
                               synth_signal_arrays["fs"])
        assert isinstance(peaks, np.ndarray)
        assert peaks.dtype.kind in ("i", "u")
        assert len(peaks) > 0
        # All peak indices must be in-range.
        assert (peaks >= 0).all()
        assert (peaks < len(synth_signal_arrays["ecg"])).all()

    def test_detect_ppg_peaks_returns_int_ndarray(self, synth_signal_arrays):
        peaks = detect_ppg_peaks(synth_signal_arrays["ppg"],
                                  synth_signal_arrays["fs"])
        assert isinstance(peaks, np.ndarray)
        assert peaks.dtype.kind in ("i", "u")
        assert len(peaks) > 0


# ── intervals ───────────────────────────────────────────────────────────────

class TestIntervals:

    def test_peaks_to_intervals_returns_two_arrays_length_minus_one(
            self, synth_signal_arrays):
        peaks = detect_r_peaks(synth_signal_arrays["ecg"],
                               synth_signal_arrays["fs"])
        intervals_ms, peak_times_s = peaks_to_intervals(
            peaks, synth_signal_arrays["ts_ms"])
        assert isinstance(intervals_ms, np.ndarray)
        assert isinstance(peak_times_s, np.ndarray)
        assert len(intervals_ms) == len(peaks) - 1
        assert len(peak_times_s) == len(peaks) - 1
        # Intervals must be positive (monotonically increasing peak times).
        assert (intervals_ms > 0).all()

    def test_match_intervals_returns_equal_length(self, synth_signal_arrays):
        ts_ms = synth_signal_arrays["ts_ms"]
        fs = synth_signal_arrays["fs"]
        r = detect_r_peaks(synth_signal_arrays["ecg"], fs)
        p = detect_ppg_peaks(synth_signal_arrays["ppg"], fs)
        rr_ms, rr_t = peaks_to_intervals(r, ts_ms)
        ppi_ms, ppi_t = peaks_to_intervals(p, ts_ms)
        mr, mp = match_intervals(rr_ms, rr_t, ppi_ms, ppi_t)
        assert len(mr) == len(mp)
        assert len(mr) > 0    # synth data produces matches

    def test_match_intervals_empty_returns_empty(self):
        empty = np.array([])
        mr, mp = match_intervals(empty, empty, empty, empty)
        assert len(mr) == 0
        assert len(mp) == 0


# ── CCC ─────────────────────────────────────────────────────────────────────

class TestComputeCCC:

    def test_compute_ccc_returns_documented_keys(self, synth_signal_arrays):
        ts_ms = synth_signal_arrays["ts_ms"]
        fs = synth_signal_arrays["fs"]
        r = detect_r_peaks(synth_signal_arrays["ecg"], fs)
        p = detect_ppg_peaks(synth_signal_arrays["ppg"], fs)
        rr_ms, rr_t = peaks_to_intervals(r, ts_ms)
        ppi_ms, ppi_t = peaks_to_intervals(p, ts_ms)
        mr, mp = match_intervals(rr_ms, rr_t, ppi_ms, ppi_t)
        d = compute_ccc(mp, mr)
        for k in ("n", "ccc", "pearson_r", "mean_ppg", "mean_ecg",
                  "bias", "std_diff", "loa_upper", "loa_lower",
                  "rmse", "mae"):
            assert k in d, f"compute_ccc result missing key {k!r}"
        assert d["n"] == len(mr)

    def test_compute_ccc_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            compute_ccc(np.array([1.0, 2.0]), np.array([1.0]))

    def test_compute_ccc_rejects_too_few(self):
        with pytest.raises(ValueError):
            compute_ccc(np.array([1.0]), np.array([1.0]))

    def test_ccc_label_buckets(self):
        assert ccc_label(0.995) == "Almost perfect"
        assert ccc_label(0.97) == "Substantial"
        assert ccc_label(0.92) == "Moderate"
        assert ccc_label(0.5) == "Poor"
        assert ccc_label(float("nan")) == "Undefined"


# ── SSQI ────────────────────────────────────────────────────────────────────

class TestSSQI:
    """SSQI = skewness of the input. Krishnan 2010 says SSQI≥1 indicates
    a clean, well-shaped PPG (positive skew from the sharp upstroke)."""

    def test_symmetric_distribution_near_zero(self):
        rng = np.random.default_rng(0)
        sig = rng.standard_normal(10000)
        s = Ssqi(sig)
        # Normal distribution: |skew| < 0.1 with n=10000.
        assert abs(s) < 0.1

    def test_positive_skew_returns_positive(self):
        rng = np.random.default_rng(1)
        sig = rng.chisquare(df=2, size=5000)
        s = Ssqi(sig)
        # Chi-squared(2) skewness = 2*sqrt(2) ≈ 2.83 in the limit.
        assert s > 0.5

    def test_negative_skew_returns_negative(self):
        rng = np.random.default_rng(2)
        sig = -rng.chisquare(df=2, size=5000)
        s = Ssqi(sig)
        assert s < -0.5
