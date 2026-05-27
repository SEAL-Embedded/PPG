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
