"""Unit tests for webapp/sessions.py.

Every persistence-helper test runs in an isolated tmp_path via the
``isolated_sessions_root`` fixture from conftest, so no real MDPIdata/
files are touched.
"""

import json
import os
from datetime import datetime

import pytest

from webapp import sessions


# ── Pattern regexes ─────────────────────────────────────────────────────────

class TestPatterns:

    @pytest.mark.parametrize("name", [
        "session_20260101_120000",
        "session_99991231_235959",
        "session_20240615_093045",
    ])
    def test_session_pattern_accepts(self, name):
        assert sessions.SESSION_PATTERN.match(name)

    @pytest.mark.parametrize("name", [
        "session_2026_01_01_12_00_00",
        "Session_20260101_120000",
        "session_20260101",
        "session_2026010_120000",
        "batch_20260101_120000",
        "",
        "session__120000",
    ])
    def test_session_pattern_rejects(self, name):
        assert sessions.SESSION_PATTERN.match(name) is None

    @pytest.mark.parametrize("name", [
        "batch_20260101_120000",
        "batch_19991231_010203",
    ])
    def test_batch_pattern_accepts(self, name):
        assert sessions.BATCH_PATTERN.match(name)

    @pytest.mark.parametrize("name", [
        "session_20260101_120000",
        "batch_2026_01_01",
        "batch_20260101",
        "",
    ])
    def test_batch_pattern_rejects(self, name):
        assert sessions.BATCH_PATTERN.match(name) is None


# ── _with_default_sites ─────────────────────────────────────────────────────

class TestWithDefaultSites:

    def test_empty_meta_fills_all_defaults(self):
        meta = sessions._with_default_sites({})
        assert meta["channel_sites"]["0"] == "finger"
        assert meta["channel_sites"]["1"] == "earlobe"
        assert meta["channel_sites"]["2"] == "shoulder"
        assert meta["channel_sites"]["3"] == "forehead"
        assert meta["channel_sites"]["4"] == "wrist"
        assert "participant_id" in meta
        assert "fitzpatrick" in meta
        assert "notes" in meta

    def test_none_meta_returns_dict(self):
        meta = sessions._with_default_sites(None)
        assert isinstance(meta, dict)
        assert meta["channel_sites"]["0"] == "finger"

    def test_explicit_site_wins_over_default(self):
        meta = sessions._with_default_sites({
            "channel_sites": {"0": "thumb"},
        })
        assert meta["channel_sites"]["0"] == "thumb"
        # The unset channels still get the default fill.
        assert meta["channel_sites"]["3"] == "forehead"

    def test_empty_string_site_treated_as_unset(self):
        """An explicit "" should be treated as "missing" and fall back
        to the default for that mux lane — the dashboard renders these
        as <unassigned>, and the receiver wires the channels the same
        way regardless of metadata."""
        meta = sessions._with_default_sites({
            "channel_sites": {"0": ""},
        })
        assert meta["channel_sites"]["0"] == "finger"

    def test_existing_top_level_fields_preserved(self):
        meta = sessions._with_default_sites({
            "participant_id": "P42",
            "fitzpatrick": 4,
            "notes": "hello",
        })
        assert meta["participant_id"] == "P42"
        assert meta["fitzpatrick"] == 4
        assert meta["notes"] == "hello"


# ── parse_timestamp_from_name ───────────────────────────────────────────────

class TestParseTimestamp:

    def test_valid_name_returns_iso(self):
        ts = sessions.parse_timestamp_from_name("session_20260101_123045")
        # datetime.fromisoformat round-trips.
        parsed = datetime.fromisoformat(ts)
        assert parsed.year == 2026 and parsed.month == 1 and parsed.day == 1
        assert parsed.hour == 12 and parsed.minute == 30 and parsed.second == 45

    def test_invalid_name_returns_none(self):
        assert sessions.parse_timestamp_from_name("not_a_session") is None
        assert sessions.parse_timestamp_from_name("session_20261301_120000") is None


# ── Persistence helpers (history / analysis / receiver log) ─────────────────

class TestPersistenceHistory:

    def test_append_then_read_newest_first(self, synth_session):
        name, _, _ = synth_session
        assert sessions.append_history(name, "first_event", {"k": 1}) is True
        assert sessions.append_history(name, "second_event", {"k": 2}) is True
        events = sessions.read_history(name)
        assert len(events) == 2
        assert events[0]["event"] == "second_event"      # newest first
        assert events[1]["event"] == "first_event"
        assert events[0]["data"]["k"] == 2
        assert "ts" in events[0]

    def test_append_history_creates_session_dir_if_missing(
            self, isolated_sessions_root):
        # No synth session — exercise the os.makedirs path.
        name = "session_20260202_010203"
        assert sessions.append_history(name, "e", {"x": 1}) is True
        events = sessions.read_history(name)
        assert len(events) == 1

    def test_read_history_respects_limit(self, synth_session):
        name, _, _ = synth_session
        for i in range(5):
            sessions.append_history(name, f"evt_{i}", {"i": i})
        events = sessions.read_history(name, limit=2)
        assert len(events) == 2
        # Newest two: evt_4, evt_3.
        assert events[0]["event"] == "evt_4"
        assert events[1]["event"] == "evt_3"

    def test_read_history_missing_file_returns_empty(self, synth_session):
        name, _, _ = synth_session
        assert sessions.read_history(name) == []

    def test_append_history_coerces_nan(self, synth_session):
        """NaN floats are not strict-JSON; the writer must coerce them
        to None so read_history() can parse the file back."""
        name, _, _ = synth_session
        sessions.append_history(name, "with_nan",
                                  {"ccc": float("nan"),
                                   "inf": float("inf"),
                                   "nested": {"x": float("nan")}})
        events = sessions.read_history(name)
        assert len(events) == 1
        assert events[0]["data"]["ccc"] is None
        assert events[0]["data"]["inf"] is None
        assert events[0]["data"]["nested"]["x"] is None


class TestPersistenceAnalysis:

    def test_save_then_load_round_trips(self, synth_session):
        name, _, _ = synth_session
        payload = {"results": [{"channel": 0, "stats": {"ccc": 0.95}}],
                   "ecg": {"n_peaks": 30}}
        assert sessions.save_session_analysis(
            name, payload, {"start_s": 1.0, "end_s": 25.0}) is True
        loaded = sessions.load_session_analysis(name)
        assert loaded is not None
        assert loaded["results"][0]["stats"]["ccc"] == 0.95
        # Persistence layer must add analyzed_at + crop_window.
        assert "analyzed_at" in loaded
        assert loaded["crop_window"]["start_s"] == 1.0
        assert loaded["crop_window"]["end_s"] == 25.0

    def test_save_with_none_crop_window(self, synth_session):
        name, _, _ = synth_session
        sessions.save_session_analysis(name, {"results": []}, None)
        loaded = sessions.load_session_analysis(name)
        assert loaded["crop_window"] == {"start_s": None, "end_s": None}

    def test_load_missing_returns_none(self, synth_session):
        name, _, _ = synth_session
        assert sessions.load_session_analysis(name) is None

    def test_save_coerces_nan_to_null(self, synth_session):
        name, _, _ = synth_session
        sessions.save_session_analysis(
            name, {"stats": {"ccc": float("nan")}}, None)
        # Raw file load — should be parseable strict JSON.
        path = os.path.join(sessions.session_path(name),
                             sessions.ANALYSIS_FILENAME)
        with open(path, "r") as f:
            raw = json.load(f)
        assert raw["stats"]["ccc"] is None


class TestPersistenceReceiverLog:

    def test_save_then_tail(self, synth_session):
        name, _, _ = synth_session
        text = "line1\nline2\nline3\n"
        assert sessions.save_receiver_log(name, text) is True
        out = sessions.tail_receiver_log(name)
        assert "line1" in out and "line3" in out

    def test_tail_respects_n(self, synth_session):
        name, _, _ = synth_session
        text = "\n".join(f"line{i}" for i in range(10)) + "\n"
        sessions.save_receiver_log(name, text)
        out = sessions.tail_receiver_log(name, n=3)
        # Last three lines: line7, line8, line9.
        assert "line9" in out
        assert "line0" not in out

    def test_save_none_writes_empty_string(self, synth_session):
        name, _, _ = synth_session
        assert sessions.save_receiver_log(name, None) is True
        assert sessions.tail_receiver_log(name) == ""

    def test_tail_missing_file_returns_empty(self, synth_session):
        name, _, _ = synth_session
        assert sessions.tail_receiver_log(name) == ""


# ── summarize_session — the new persistence-aware fields ────────────────────

class TestSummarizeSession:

    def test_lightweight_summary_has_new_fields(self, synth_session):
        name, _, _ = synth_session
        summary = sessions.summarize_session(name, lightweight=True)
        # Sanity over the legacy shape.
        assert summary["name"] == name
        assert summary["has_ecg"] is True
        assert sorted(summary["channels"]) == [0, 1, 2]
        # New persistence fields:
        assert "last_analyzed_at" in summary
        assert "analysis_count" in summary
        assert "has_receiver_log" in summary
        assert "history_count" in summary

    def test_counts_zero_for_fresh_session(self, synth_session):
        name, _, _ = synth_session
        summary = sessions.summarize_session(name)
        assert summary["last_analyzed_at"] is None
        assert summary["analysis_count"] == 0
        assert summary["history_count"] == 0
        assert summary["has_receiver_log"] is False

    def test_counts_track_analysis_events(self, synth_session):
        """The history-counter scans for the literal substring
        '"event":"analysis_run"' — make sure the writer emits compact
        JSON in the format the counter expects."""
        name, _, _ = synth_session
        sessions.append_history(name, "metadata_edited", {})
        sessions.append_history(name, "analysis_run", {"n_channels": 3})
        sessions.append_history(name, "analysis_run", {"n_channels": 3})
        summary = sessions.summarize_session(name)
        assert summary["history_count"] == 3
        assert summary["analysis_count"] == 2

    def test_last_analyzed_at_set_after_save(self, synth_session):
        name, _, _ = synth_session
        sessions.save_session_analysis(name, {}, None)
        summary = sessions.summarize_session(name)
        assert summary["last_analyzed_at"] is not None
        # Suffix Z is the persistence layer's convention.
        assert summary["last_analyzed_at"].endswith("Z")

    def test_has_receiver_log_flag(self, synth_session):
        name, _, _ = synth_session
        sessions.save_receiver_log(name, "hello\n")
        summary = sessions.summarize_session(name)
        assert summary["has_receiver_log"] is True


# ── list_sessions ──────────────────────────────────────────────────────────

class TestListSessions:

    def test_lists_only_session_pattern_folders(
            self, isolated_sessions_root, synth_session):
        # Add a non-session folder and a non-folder file alongside.
        os.makedirs(os.path.join(isolated_sessions_root, "scratch"))
        with open(os.path.join(isolated_sessions_root, "stray.txt"), "w") as f:
            f.write("nope")
        out = sessions.list_sessions()
        names = [s["name"] for s in out]
        assert synth_session[0] in names
        assert "scratch" not in names
        assert "stray.txt" not in names


# ── Batch analysis archive ─────────────────────────────────────────────────

class TestBatchAnalyses:

    def test_save_then_list_then_load(self, isolated_sessions_root):
        payload = {"n_sessions_analyzed": 4, "sessions": [],
                    "crop_window": {"start_s": None, "end_s": None}}
        saved = sessions.save_batch_analysis(payload)
        assert saved["batch_id"].startswith("batch_")
        assert "created_at" in saved

        listed = sessions.list_batch_analyses()
        assert any(r["batch_id"] == saved["batch_id"] for r in listed)
        row = next(r for r in listed if r["batch_id"] == saved["batch_id"])
        assert row["n_sessions_analyzed"] == 4
        assert "crop_window" in row

        loaded = sessions.load_batch_analysis(saved["batch_id"])
        assert loaded is not None
        assert loaded["batch_id"] == saved["batch_id"]
        assert loaded["n_sessions_analyzed"] == 4

    def test_load_invalid_id_returns_none(self, isolated_sessions_root):
        assert sessions.load_batch_analysis("not_a_batch") is None
        assert sessions.load_batch_analysis("") is None
        # Path traversal attempts get rejected by the pattern.
        assert sessions.load_batch_analysis("../../etc/passwd") is None

    def test_load_missing_returns_none(self, isolated_sessions_root):
        # Pattern-valid but file doesn't exist.
        assert sessions.load_batch_analysis("batch_20990101_010101") is None

    def test_list_empty_root(self, isolated_sessions_root):
        # batch_analyses_root() creates the dir on demand; empty list.
        assert sessions.list_batch_analyses() == []


# ── load_participant_metadata ──────────────────────────────────────────────

class TestParticipantMetadata:

    def test_missing_file_returns_default_filled(self, synth_session):
        name, _, _ = synth_session
        meta = sessions.load_participant_metadata(name)
        # Even with no participant.json, defaults are filled.
        assert meta["channel_sites"]["0"] == "finger"
        assert meta["participant_id"] == ""

    def test_save_then_load_round_trips(self, synth_session):
        name, _, _ = synth_session
        sessions.save_participant_metadata(name, {
            "participant_id": "P77",
            "fitzpatrick": 5,
            "notes": "test",
            "channel_sites": {"0": "thumb"},
        })
        meta = sessions.load_participant_metadata(name)
        assert meta["participant_id"] == "P77"
        assert meta["fitzpatrick"] == 5
        # Explicit site preserved, others defaulted.
        assert meta["channel_sites"]["0"] == "thumb"
        assert meta["channel_sites"]["1"] == "earlobe"


# ── Per-session "best window" ───────────────────────────────────────────────

class TestSessionWindow:

    def test_missing_window_returns_none_bounds(self, synth_session):
        name, _, _ = synth_session
        assert sessions.get_session_window(name) == {
            "start_s": None, "end_s": None}

    def test_save_then_get_round_trips(self, synth_session):
        name, _, _ = synth_session
        sessions.save_session_window(name, 5.0, 15.0)
        assert sessions.get_session_window(name) == {
            "start_s": 5.0, "end_s": 15.0}
        # File actually written.
        assert os.path.isfile(
            os.path.join(sessions.session_path(name),
                         sessions.WINDOW_FILENAME))

    def test_open_bound_persists_as_none(self, synth_session):
        name, _, _ = synth_session
        sessions.save_session_window(name, None, 20.0)
        assert sessions.get_session_window(name) == {
            "start_s": None, "end_s": 20.0}

    def test_both_none_clears_saved_window(self, synth_session):
        name, _, _ = synth_session
        sessions.save_session_window(name, 5.0, 15.0)
        sessions.save_session_window(name, None, None)
        # File removed, get reverts to None bounds.
        assert not os.path.isfile(
            os.path.join(sessions.session_path(name),
                         sessions.WINDOW_FILENAME))
        assert sessions.get_session_window(name) == {
            "start_s": None, "end_s": None}

    def test_clear_when_none_saved_is_noop(self, synth_session):
        name, _, _ = synth_session
        # Clearing with nothing saved must not raise.
        sessions.save_session_window(name, None, None)
        assert sessions.get_session_window(name) == {
            "start_s": None, "end_s": None}

    def test_corrupt_window_file_returns_none_bounds(self, synth_session):
        name, _, _ = synth_session
        with open(os.path.join(sessions.session_path(name),
                               sessions.WINDOW_FILENAME), "w") as f:
            f.write("{ not json")
        assert sessions.get_session_window(name) == {
            "start_s": None, "end_s": None}

    def test_summarize_session_includes_saved_window(self, synth_session):
        name, _, _ = synth_session
        sessions.save_session_window(name, 3.0, 12.0)
        summary = sessions.summarize_session(name)
        assert summary["window"] == {"start_s": 3.0, "end_s": 12.0}
