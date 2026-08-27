"""Unit tests for webapp/sleepiness.py.

Same isolation rules as the other test modules: every test that touches
disk goes through the ``isolated_sessions_root`` fixture, so the real
MDPIdata/ tree is never read or written.

The HRV-feature tests work on bare numpy arrays — no file I/O — so
they're fast and don't need the session fixture.
"""

import json
import os

import numpy as np
import pytest

from depereciated import sleepiness
from webapp import sessions


# ── HRV features on bare RR arrays ──────────────────────────────────────────

class TestComputeHrvFeatures:

    def test_short_series_returns_all_nan(self):
        """< MIN_BEATS_FOR_HRV intervals → every feature NaN. We rely on
        the dict still containing all keys so downstream aggregation can
        iterate without conditional logic."""
        feats = sleepiness.compute_hrv_features(np.array([800.0] * 5),
                                                  np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        for k in ("sdnn_ms", "rmssd_ms", "lf_power", "hf_power",
                  "lf_nu", "hf_nu", "sampen", "sd1_sd2", "log_lf_hf"):
            assert not np.isfinite(feats[k]), f"{k} should be NaN"

    def test_clean_rr_series_returns_finite_features(self):
        """A 60-beat synthetic RR series at 800 ms ± 30 ms must produce
        finite values for the time-domain and Poincaré features. (LF/HF
        Lomb-Scargle may still be noisy on synthetic data — we don't
        assert finite there.)"""
        rng = np.random.default_rng(7)
        n = 60
        rr_ms = 800.0 + 30.0 * rng.standard_normal(n)
        # Build cumulative times in seconds for the timestamps.
        rr_t_s = np.cumsum(rr_ms / 1000.0)
        feats = sleepiness.compute_hrv_features(rr_ms, rr_t_s)
        assert np.isfinite(feats["sdnn_ms"])
        assert feats["sdnn_ms"] > 0
        assert np.isfinite(feats["rmssd_ms"])
        assert feats["rmssd_ms"] > 0
        assert np.isfinite(feats["sd1_ms"])
        assert np.isfinite(feats["sd2_ms"])
        assert 0.0 <= feats["pnn50"] <= 1.0
        # Sample entropy on this length should be a real number.
        assert np.isfinite(feats["sampen"])

    def test_poincare_brennan_identities(self):
        """Brennan's closed forms must hold:
            sd1 = rmssd / sqrt(2)
            sd2^2 = 2·sdnn^2 - 0.5·var(diff(RR))
        """
        rng = np.random.default_rng(11)
        rr_ms = 800.0 + 25.0 * rng.standard_normal(80)
        rr_t_s = np.cumsum(rr_ms / 1000.0)
        feats = sleepiness.compute_hrv_features(rr_ms, rr_t_s)
        expected_sd1 = feats["rmssd_ms"] / np.sqrt(2.0)
        assert feats["sd1_ms"] == pytest.approx(expected_sd1, rel=1e-6)
        diffs = np.diff(rr_ms)
        expected_sd2_sq = 2.0 * (feats["sdnn_ms"] ** 2) - 0.5 * float(np.var(diffs, ddof=1))
        assert feats["sd2_ms"] ** 2 == pytest.approx(expected_sd2_sq, rel=1e-6)


# ── Sample entropy ──────────────────────────────────────────────────────────

class TestSampleEntropy:

    def test_constant_series_undefined(self):
        """All-equal series → either A==0 or B==0 → returns NaN."""
        v = sleepiness._sample_entropy(np.ones(40))
        assert not np.isfinite(v)

    def test_random_series_finite(self):
        rng = np.random.default_rng(0)
        v = sleepiness._sample_entropy(rng.standard_normal(80))
        assert np.isfinite(v)
        # SampEn for normal noise typically lands somewhere in (0, 3).
        assert 0.0 < v < 5.0

    def test_short_series_returns_nan(self):
        assert not np.isfinite(sleepiness._sample_entropy(np.array([1.0, 2.0])))


# ── SPI scoring ─────────────────────────────────────────────────────────────

class TestComputeSleepinessIndex:

    def test_zero_cohort_std_yields_nan(self):
        """When the cohort std for a feature is zero (only one session
        contributed), z-score collapses to NaN and the term is excluded.
        With *every* term excluded, SPI is NaN."""
        cohort = {
            "rmssd_ms":  {"mean": 0.0, "std": 0.0, "n": 1},
            "hf_nu":     {"mean": 0.0, "std": 0.0, "n": 1},
            "log_lf_hf": {"mean": 0.0, "std": 0.0, "n": 1},
            "sampen":    {"mean": 0.0, "std": 0.0, "n": 1},
            "sd1_sd2":   {"mean": 0.0, "std": 0.0, "n": 1},
        }
        feats = sleepiness._nan_features()
        spi, comps = sleepiness.compute_sleepiness_index(feats, cohort)
        assert not np.isfinite(spi)
        for k in sleepiness.SPI_WEIGHTS:
            assert not np.isfinite(comps[k])

    def test_features_at_mean_yield_zero_spi(self):
        """If every feature is exactly at the cohort mean, every z is 0
        and the SPI sums to 0 exactly."""
        cohort = {
            "rmssd_ms":  {"mean": np.log(50.0), "std": 0.3, "n": 5},
            "hf_nu":     {"mean": np.log(0.35 + sleepiness.EPS), "std": 0.2, "n": 5},
            "log_lf_hf": {"mean": 0.5, "std": 0.4, "n": 5},
            "sampen":    {"mean": 1.5, "std": 0.2, "n": 5},
            "sd1_sd2":   {"mean": 0.5, "std": 0.1, "n": 5},
        }
        feats = {
            "rmssd_ms": 50.0,
            "hf_nu": 0.35,
            "log_lf_hf": 0.5,
            "sampen": 1.5,
            "sd1_sd2": 0.5,
        }
        # Pad the dict so _feature_for_zscore can read all keys.
        for k in ("sdnn_ms", "pnn50", "lf_power", "hf_power",
                  "lf_nu", "lf_hf_ratio", "sd1_ms", "sd2_ms"):
            feats.setdefault(k, 0.0)
        spi, comps = sleepiness.compute_sleepiness_index(feats, cohort)
        assert spi == pytest.approx(0.0, abs=1e-9)
        for v in comps.values():
            assert v == pytest.approx(0.0, abs=1e-9)


# ── Quality weight sigmoid ──────────────────────────────────────────────────

class TestQualityWeight:

    def test_nan_ssqi_returns_floor(self):
        assert sleepiness._quality_weight(float("nan"), 0.02) == 0.05

    def test_clean_signal_high_weight(self):
        # SSQI=2 (strong positive skew), ZSQI=0.03 (well below 0.05) →
        # both sigmoids saturate near 1.
        q = sleepiness._quality_weight(2.0, 0.03)
        assert q > 0.5

    def test_bad_signal_floors_at_005(self):
        # Negative SSQI (inverted) and high ZSQI (noisy) drive both
        # sigmoids near zero — multiplication clips to floor.
        q = sleepiness._quality_weight(-2.0, 0.30)
        assert q == 0.05

    def test_weight_bounded(self):
        for ssqi, zsqi in [(2.0, 0.01), (0.0, 0.05), (-1.0, 0.5)]:
            q = sleepiness._quality_weight(ssqi, zsqi)
            assert 0.05 <= q <= 1.0


# ── Cohort stats ────────────────────────────────────────────────────────────

class TestCohortStats:

    def test_empty_cohort_yields_zero_n(self):
        stats = sleepiness.compute_cohort_stats([])
        for k in stats:
            assert stats[k]["n"] == 0
            assert not np.isfinite(stats[k]["mean"])

    def test_simple_two_session_cohort(self):
        f1 = {"rmssd_ms": 40.0, "hf_nu": 0.30, "log_lf_hf": 0.2,
              "sampen": 1.2, "sd1_sd2": 0.5}
        f2 = {"rmssd_ms": 60.0, "hf_nu": 0.40, "log_lf_hf": 0.8,
              "sampen": 1.6, "sd1_sd2": 0.7}
        for f in (f1, f2):
            for k in ("sdnn_ms", "pnn50", "lf_power", "hf_power",
                      "lf_nu", "lf_hf_ratio", "sd1_ms", "sd2_ms"):
                f.setdefault(k, 0.0)
        stats = sleepiness.compute_cohort_stats([f1, f2])
        # RMSSD goes through log() before z-norm: mean = (ln40 + ln60)/2
        expected_rmssd_mean = (np.log(40.0) + np.log(60.0)) / 2.0
        assert stats["rmssd_ms"]["mean"] == pytest.approx(expected_rmssd_mean, rel=1e-9)
        assert stats["rmssd_ms"]["n"] == 2


# ── End-to-end analyze_sleepiness on a synthetic session ────────────────────

class TestAnalyzeSleepiness:

    def test_no_sessions_returns_empty_shape(self, isolated_sessions_root):
        result = sleepiness.analyze_sleepiness()
        # Shape contract: every required key is present even when the
        # cohort is empty.
        for k in ("weighting", "crop_window", "config", "cohort_stats",
                  "per_session", "per_site", "per_fst_site",
                  "feature_contributions_per_site",
                  "unusable_sessions", "scatter_points", "caveats",
                  "interpretation"):
            assert k in result
        assert result["per_session"] == []
        assert result["per_site"] == []

    def test_one_synthetic_session_runs(self, synth_session_with_metadata):
        """One session yields a per_session row (usable or not depending on
        the synthetic peak count) and a populated config block."""
        name, _, _, _ = synth_session_with_metadata
        result = sleepiness.analyze_sleepiness()
        assert result["n_sessions_total"] >= 1
        assert any(s["session_name"] == name for s in result["per_session"])
        # The synthetic ECG has ~30 R-peaks in 30 s → exactly at the
        # MIN_BEATS_FOR_HRV boundary. SPI may or may not be finite, but
        # the row must exist.
        row = [s for s in result["per_session"] if s["session_name"] == name][0]
        assert "ecg" in row and "channels" in row
        # Config dump must reflect the pre-registered weights.
        assert result["config"]["weights"] == sleepiness.SPI_WEIGHTS
        # Caveats must be on the response so the frontend can surface them.
        assert len(result["caveats"]) >= 5


# ── Persistence ─────────────────────────────────────────────────────────────

class TestPersistence:

    def test_save_and_load_round_trip(self, isolated_sessions_root):
        payload = {"per_session": [], "weighting": "ssqi_zsqi"}
        saved = sessions.save_sleepiness_run(payload)
        assert "run_id" in saved
        assert sessions.SLEEPINESS_PATTERN.match(saved["run_id"])
        # File must exist on disk.
        path = os.path.join(sessions.sleepiness_runs_root(),
                             f"{saved['run_id']}.json")
        assert os.path.isfile(path)
        # Load by id round-trips the run.
        loaded = sessions.load_sleepiness_run(saved["run_id"])
        assert loaded["run_id"] == saved["run_id"]
        # list_sleepiness_runs sees it.
        rows = sessions.list_sleepiness_runs()
        assert any(r["run_id"] == saved["run_id"] for r in rows)

    def test_load_latest_returns_newest(self, isolated_sessions_root):
        import time
        sessions.save_sleepiness_run({"per_session": [], "weighting": "a"})
        time.sleep(1.05)   # the filename is timestamped to the second
        sessions.save_sleepiness_run({"per_session": [], "weighting": "b"})
        latest = sessions.load_latest_sleepiness_run()
        assert latest is not None
        # Newest first means weighting="b" wins.
        assert latest.get("weighting") == "b"

    def test_load_latest_none_when_empty(self, isolated_sessions_root):
        assert sessions.load_latest_sleepiness_run() is None

    def test_load_rejects_traversal_attempts(self, isolated_sessions_root):
        # Malformed id (path-traversal attempt) must return None, not
        # read a file outside the sleepiness_runs folder.
        assert sessions.load_sleepiness_run("../../../etc/passwd") is None
        assert sessions.load_sleepiness_run("run_invalid") is None
