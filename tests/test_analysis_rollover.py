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

    def test_unwrap_helper_no_op_on_monotone_input(self):
        ts = np.array([100, 200, 300, 400], dtype=np.int64)
        result = analysis._unwrap_ticks_us(ts)
        np.testing.assert_array_equal(result, ts.astype(np.float64))

    def test_unwrap_helper_single_sample_returns_unchanged(self):
        ts = np.array([42], dtype=np.int64)
        result = analysis._unwrap_ticks_us(ts)
        np.testing.assert_array_equal(result, ts.astype(np.float64))

    def test_unwrap_helper_multiple_wraps(self):
        # Two wraps in one series.
        WRAP = 1 << 30
        ts = np.array([WRAP - 1000, 0, WRAP - 1000, 0], dtype=np.int64)
        result = analysis._unwrap_ticks_us(ts)
        expected = np.array([WRAP - 1000, WRAP, 2*WRAP - 1000, 2*WRAP],
                            dtype=np.float64)
        np.testing.assert_array_equal(result, expected)


class TestLeadsOffSpansAlignment:

    def test_leads_off_spans_fall_within_downsampled_xs(
        self, tmp_path
    ):
        """Spans must not extend beyond xs[0]..xs[-1]."""
        # Build a synthetic ECG where leads_off=1 over a fraction.
        n = 5000
        ts_us = (np.arange(n) * 3000).astype(np.int64)  # 333 Hz
        sig = np.zeros(n)
        leads_off = np.zeros(n, dtype=int)
        leads_off[1000:2000] = 1  # leads-off in the middle
        p = tmp_path / "ecg_data.csv"
        with open(p, "w") as f:
            for i in range(n):
                f.write(f"{ts_us[i]},{sig[i]},{leads_off[i]}\n")
        # Manually load + downsample to test the alignment.
        # (This test asserts the contract — the actual load happens in
        #  load_session_signals which we exercise via the API tests.)
        ts_ms, sig_arr, lo = analysis.load_ecg(str(p))
        ts_s = (ts_ms - ts_ms[0]) / 1000.0
        spans = analysis._leads_off_spans(ts_s, lo)
        # All spans must be within the full ts_s range.
        for start, end in spans:
            assert ts_s[0] - 1e-6 <= start <= ts_s[-1] + 1e-6
            assert ts_s[0] - 1e-6 <= end <= ts_s[-1] + 1e-6
            assert start <= end

    def test_load_session_signals_leads_off_spans_clamped_to_xs(
        self, isolated_sessions_root
    ):
        """End-to-end: load_session_signals must clamp leads_off spans
        to the downsampled xs extent. The min/max bucket decimator can
        return an xs whose first/last point sits inside ts_s by several
        samples, so unclamped spans would jut past the visible trace.
        """
        # Build a session_<ts>/ecg_data.csv where leads_off=1 from the
        # very first sample to the last — so the span is exactly
        # (ts_s[0], ts_s[-1]) and any narrower xs[0]..xs[-1] from the
        # decimator exposes the misalignment unless clamping happens.
        name = "session_20260101_120000"
        sdir = os.path.join(isolated_sessions_root, name)
        os.makedirs(sdir, exist_ok=True)
        n = 5000
        ts_us = (np.arange(n) * 3000).astype(np.int64)  # 333 Hz
        # Random noise so argmin/argmax in the first/last bucket are
        # almost never at index 0 / n-1, forcing xs[0] > ts_s[0] and
        # xs[-1] < ts_s[-1].
        rng = np.random.default_rng(42)
        sig = rng.standard_normal(n)
        leads_off = np.ones(n, dtype=int)  # leads-off the whole time
        p = os.path.join(sdir, "ecg_data.csv")
        with open(p, "w") as f:
            for i in range(n):
                f.write(f"{ts_us[i]},{sig[i]},{leads_off[i]}\n")

        out = analysis.load_session_signals(name, max_points=200)
        ecg = out["ecg"]
        xs = ecg["time_s"]
        spans = ecg["leads_off_spans"]
        assert len(xs) > 0
        assert len(spans) > 0
        # Confirm the test data actually exposes the misalignment risk —
        # i.e. xs does NOT touch the original endpoints. Otherwise the
        # test would pass trivially.
        ts_s_end_expected = (n - 1) * 3000 / 1_000_000.0
        assert xs[0] > 0.0 or xs[-1] < ts_s_end_expected, (
            "test data did not produce an inset xs; the misalignment "
            "scenario is not being exercised"
        )
        x_lo, x_hi = xs[0], xs[-1]
        eps = 1e-9
        for start, end in spans:
            assert x_lo - eps <= start <= x_hi + eps, (
                f"span start {start} outside xs range "
                f"[{x_lo}, {x_hi}]"
            )
            assert x_lo - eps <= end <= x_hi + eps, (
                f"span end {end} outside xs range "
                f"[{x_lo}, {x_hi}]"
            )
            assert start <= end
