"""Tests for sqi/hrv_clean.clean_intervals."""
import numpy as np
import pytest

from sqi.hrv_clean import clean_intervals


class TestRangeGate:

    def test_drops_intervals_below_300ms(self):
        intervals = np.array([800., 250., 800., 800.])
        times = np.array([1.0, 1.25, 2.05, 2.85])
        clean, t_clean, mask = clean_intervals(intervals, times)
        assert list(mask) == [True, False, True, True]
        assert len(clean) == 3
        assert 250.0 not in clean

    def test_drops_intervals_above_2000ms(self):
        intervals = np.array([800., 2500., 800., 800.])
        times = np.array([1.0, 1.8, 4.3, 5.1])
        clean, t_clean, mask = clean_intervals(intervals, times)
        assert list(mask) == [True, False, True, True]

    def test_custom_range(self):
        intervals = np.array([400., 800., 1500.])
        times = np.array([0.4, 1.2, 2.7])
        clean, t_clean, mask = clean_intervals(intervals, times,
                                               range_ms=(500.0, 1000.0))
        assert list(mask) == [False, True, False]


class TestKarlssonRule:

    def test_drops_outlier_vs_local_median(self):
        # Steady 800 ms beats with one 1500 ms outlier
        intervals = np.array([800.]*10 + [1500.] + [800.]*10)
        times = np.cumsum(intervals) / 1000.0
        clean, t_clean, mask = clean_intervals(intervals, times)
        # Outlier dropped
        assert not mask[10]
        # Other beats survive
        assert mask.sum() == 20

    def test_preserves_clean_60bpm_series(self):
        rng = np.random.default_rng(42)
        intervals = 800.0 + 25.0 * rng.standard_normal(60)
        times = np.cumsum(intervals) / 1000.0
        clean, t_clean, mask = clean_intervals(intervals, times)
        # Allow at most 5% accidental drops on a clean series
        assert mask.sum() >= 57

    def test_handles_constant_perfect_series(self):
        intervals = np.array([800.0] * 30)
        times = np.cumsum(intervals) / 1000.0
        clean, t_clean, mask = clean_intervals(intervals, times)
        assert mask.all()
        assert len(clean) == 30


class TestAlignment:

    def test_times_aligned_with_intervals(self):
        intervals = np.array([800., 250., 800., 800.])
        times = np.array([1.0, 1.25, 2.05, 2.85])
        clean, t_clean, mask = clean_intervals(intervals, times)
        # Times of kept intervals
        np.testing.assert_array_equal(t_clean, times[mask])
        assert len(t_clean) == len(clean)


class TestEdgeCases:

    def test_empty_input(self):
        intervals = np.array([])
        times = np.array([])
        clean, t_clean, mask = clean_intervals(intervals, times)
        assert len(clean) == 0
        assert len(t_clean) == 0
        assert len(mask) == 0

    def test_single_interval(self):
        intervals = np.array([800.])
        times = np.array([1.0])
        clean, t_clean, mask = clean_intervals(intervals, times)
        # Single value: passes range gate, no comparable median
        assert len(clean) == 1
        assert mask[0]

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            clean_intervals(np.array([800., 800.]), np.array([1.0]))

    def test_invalid_karlsson_pct_raises(self):
        with pytest.raises(ValueError):
            clean_intervals(np.array([800.]), np.array([1.0]),
                            karlsson_pct=1.5)
