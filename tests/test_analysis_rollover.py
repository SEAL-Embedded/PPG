"""Tests for ticks_us wrap-around handling + leads_off span alignment."""
import os
import numpy as np
import pytest

from webapp import analysis


def _make_wrap_csv(path, n_before, n_after, period_us=2000, n_cols=2):
    """Write a synthetic CSV with timestamps that wrap at 2^30 us.

    n_before samples lead up to just before 2^30 (last value =
    ``WRAP - period_us``; the counter never actually emits 2^30 — it
    rolls from 2^30 - 1 to 0). Then n_after samples appear at small
    values (post-wrap).
    """
    WRAP = 1 << 30
    ts_us = np.concatenate([
        WRAP - period_us * n_before + period_us * np.arange(n_before),
        period_us * np.arange(n_after)  # wrapped values
    ]).astype(np.int64)
    samples = np.arange(len(ts_us)).astype(float)
    with open(path, "w", encoding="utf-8", newline="") as f:
        for i in range(len(ts_us)):
            if n_cols == 3:
                f.write(f"{ts_us[i]},{samples[i]},0\n")
            else:
                f.write(f"{ts_us[i]},{samples[i]}\n")
    return ts_us


class TestUnwrapTicksUs:

    def test_load_ppg_unwraps_30bit_wrap(self, tmp_path):
        p = tmp_path / "ppg_data_ch0.csv"
        _make_wrap_csv(str(p), n_before=10, n_after=10, n_cols=2)
        ts_ms, sig = analysis.load_ppg(str(p))
        # All diffs must be positive after unwrap.
        diffs = np.diff(ts_ms)
        assert (diffs > 0).all(), \
            f"ts_ms has backward steps after unwrap: {diffs[diffs <= 0]}"

    def test_load_ecg_unwraps_30bit_wrap(self, tmp_path):
        p = tmp_path / "ecg_data.csv"
        _make_wrap_csv(str(p), n_before=10, n_after=10, n_cols=3)
        ts_ms, sig, leads_off = analysis.load_ecg(str(p))
        diffs = np.diff(ts_ms)
        assert (diffs > 0).all()
