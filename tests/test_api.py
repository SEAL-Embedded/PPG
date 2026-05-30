"""FastAPI integration tests via fastapi.testclient.TestClient.

Every test runs under ``isolated_sessions_root`` so endpoints that
write to disk (analyze, metadata, batch) land in tmp_path rather than
in the real MDPIdata/ folder. The TestClient is a plain in-process
HTTP shim — no network, no port binding.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from webapp import api, sessions


@pytest.fixture
def client(isolated_sessions_root):
    """TestClient against the live FastAPI app. The session root is
    already monkeypatched by the fixture dependency."""
    return TestClient(api.app)


# ── /api/sessions ───────────────────────────────────────────────────────────

class TestSessionsList:

    def test_get_sessions_empty(self, client):
        r = client.get("/api/sessions")
        assert r.status_code == 200
        assert r.json() == []

    def test_get_sessions_returns_new_metadata_fields(
            self, client, synth_session):
        name, _, _ = synth_session
        r = client.get("/api/sessions")
        assert r.status_code == 200
        rows = r.json()
        assert len(rows) == 1
        row = rows[0]
        assert row["name"] == name
        for k in ("last_analyzed_at", "analysis_count",
                  "has_receiver_log", "history_count"):
            assert k in row


class TestSingleSession:

    def test_get_single_session(self, client, synth_session):
        name, _, _ = synth_session
        r = client.get(f"/api/sessions/{name}")
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == name
        assert data["has_ecg"] is True
        assert sorted(data["channels"]) == [0, 1, 2]

    def test_invalid_name_returns_400(self, client):
        r = client.get("/api/sessions/not_a_session")
        assert r.status_code == 400

    def test_nonexistent_session_returns_404(self, client):
        # Valid pattern, but no folder.
        r = client.get("/api/sessions/session_99990101_000000")
        assert r.status_code == 404


# ── POST /api/sessions/{name}/analyze ──────────────────────────────────────

class TestAnalyzeEndpoint:

    def test_analyze_writes_analysis_json_and_history(
            self, client, synth_session):
        name, _, _ = synth_session
        r = client.post(f"/api/sessions/{name}/analyze")
        assert r.status_code == 200
        body = r.json()
        assert "results" in body
        assert "ecg" in body
        assert "interpretation" in body

        # Persisted to disk.
        analysis_path = os.path.join(
            sessions.session_path(name), sessions.ANALYSIS_FILENAME)
        assert os.path.isfile(analysis_path)

        # History event appended.
        events = sessions.read_history(name)
        evts = [e["event"] for e in events]
        assert "analysis_run" in evts

    def test_analyze_with_crop_window(self, client, synth_session):
        name, _, _ = synth_session
        r = client.post(
            f"/api/sessions/{name}/analyze",
            params={"start_s": 5.0, "end_s": 15.0})
        assert r.status_code == 200
        body = r.json()
        assert body["ecg"]["duration_s"] == pytest.approx(10.0, abs=0.1)

    def test_analyze_invalid_name(self, client):
        r = client.post("/api/sessions/bogus/analyze")
        assert r.status_code == 400

    def test_analyze_nonexistent(self, client):
        r = client.post("/api/sessions/session_99990101_000000/analyze")
        assert r.status_code == 404


# ── POST /api/analyze_all ───────────────────────────────────────────────────

class TestAnalyzeAllEndpoint:

    def test_writes_batch_snapshot_and_fans_out_history(
            self, client, synth_session):
        name, _, _ = synth_session
        r = client.post("/api/analyze_all")
        assert r.status_code == 200
        body = r.json()
        # batch_id added by the persistence layer.
        assert body.get("batch_id", "").startswith("batch_")
        assert "created_at" in body

        # batch_<ts>.json sitting under MDPIdata/batch_analyses/.
        bdir = sessions.batch_analyses_root()
        assert any(f.startswith("batch_") and f.endswith(".json")
                   for f in os.listdir(bdir))

        # Each analysed session received a batch_analysis_included event.
        events = sessions.read_history(name)
        evts = [e["event"] for e in events]
        assert "batch_analysis_included" in evts

    def test_empty_root_still_returns_200(self, client):
        r = client.post("/api/analyze_all")
        assert r.status_code == 200
        body = r.json()
        assert body["n_sessions_total"] == 0


# ── GET /api/sessions/{name}/history ───────────────────────────────────────

class TestHistoryEndpoint:

    def test_returns_events_newest_first(self, client, synth_session):
        name, _, _ = synth_session
        sessions.append_history(name, "first", {"k": 1})
        sessions.append_history(name, "second", {"k": 2})

        r = client.get(f"/api/sessions/{name}/history")
        assert r.status_code == 200
        events = r.json()
        assert events[0]["event"] == "second"
        assert events[1]["event"] == "first"

    def test_respects_limit(self, client, synth_session):
        name, _, _ = synth_session
        for i in range(5):
            sessions.append_history(name, f"evt_{i}", {"i": i})
        r = client.get(f"/api/sessions/{name}/history", params={"limit": 2})
        assert r.status_code == 200
        events = r.json()
        assert len(events) == 2

    def test_invalid_name(self, client):
        r = client.get("/api/sessions/bad_name/history")
        assert r.status_code == 400


# ── GET /api/sessions/{name}/analysis ───────────────────────────────────────

class TestCachedAnalysisEndpoint:

    def test_returns_cached_payload_when_present(
            self, client, synth_session):
        name, _, _ = synth_session
        sessions.save_session_analysis(
            name, {"results": [], "ecg": {"n_peaks": 30}}, None)
        r = client.get(f"/api/sessions/{name}/analysis")
        assert r.status_code == 200
        body = r.json()
        # We don't expect the "cached: false" sentinel when a file exists.
        assert body.get("cached", True) is not False
        assert body["ecg"]["n_peaks"] == 30

    def test_returns_cached_false_when_absent(self, client, synth_session):
        name, _, _ = synth_session
        r = client.get(f"/api/sessions/{name}/analysis")
        assert r.status_code == 200
        assert r.json() == {"cached": False}


# ── GET /api/sessions/{name}/receiver_log ──────────────────────────────────

class TestReceiverLogEndpoint:

    def test_returns_log_dict(self, client, synth_session):
        name, _, _ = synth_session
        sessions.save_receiver_log(name, "hello\nworld\n")
        r = client.get(f"/api/sessions/{name}/receiver_log")
        assert r.status_code == 200
        body = r.json()
        assert "log" in body
        assert "hello" in body["log"]

    def test_empty_when_missing(self, client, synth_session):
        name, _, _ = synth_session
        r = client.get(f"/api/sessions/{name}/receiver_log")
        assert r.status_code == 200
        assert r.json() == {"log": ""}


# ── /api/batch_analyses round trip ──────────────────────────────────────────

class TestBatchEndpoints:

    def test_list_then_load_round_trip(self, client, synth_session):
        # Trigger a batch run to create a snapshot on disk.
        r = client.post("/api/analyze_all")
        assert r.status_code == 200
        batch_id = r.json()["batch_id"]

        # List endpoint surfaces the new batch.
        r = client.get("/api/batch_analyses")
        assert r.status_code == 200
        rows = r.json()
        assert any(row["batch_id"] == batch_id for row in rows)

        # Single-batch endpoint returns the full payload.
        r = client.get(f"/api/batch_analyses/{batch_id}")
        assert r.status_code == 200
        body = r.json()
        assert body["batch_id"] == batch_id
        assert "sessions" in body

    def test_invalid_batch_id_returns_400(self, client):
        r = client.get("/api/batch_analyses/not_a_batch")
        assert r.status_code == 400

    def test_missing_batch_returns_404(self, client):
        r = client.get("/api/batch_analyses/batch_20990101_010101")
        assert r.status_code == 404


# ── POST /api/sessions/{name}/metadata ──────────────────────────────────────

class TestMetadataEndpoint:

    def test_writes_participant_and_appends_history(
            self, client, synth_session):
        name, _, _ = synth_session
        payload = {
            "participant_id": "P123",
            "fitzpatrick": 4,
            "notes": "hello",
            "channel_sites": {"0": "thumb"},
        }
        r = client.post(f"/api/sessions/{name}/metadata", json=payload)
        assert r.status_code == 200
        assert r.json() == {"ok": True}

        # File on disk.
        meta = sessions.load_participant_metadata(name)
        assert meta["participant_id"] == "P123"
        assert meta["fitzpatrick"] == 4
        # Notes survive the round-trip exactly as posted — guards against
        # any future merge-logic regression that drops the field.
        assert meta["notes"] == "hello"
        # Explicit override survives the default-fill merge.
        assert meta["channel_sites"]["0"] == "thumb"

        # History event appended with before/after pair.
        events = sessions.read_history(name)
        edits = [e for e in events if e["event"] == "metadata_edited"]
        assert len(edits) == 1
        assert "before" in edits[0]["data"]
        assert "after" in edits[0]["data"]
        assert edits[0]["data"]["after"]["participant_id"] == "P123"

    def test_invalid_name(self, client):
        r = client.post("/api/sessions/bad/metadata",
                         json={"participant_id": "X"})
        assert r.status_code == 400

    def test_metadata_endpoint_blank_notes_does_not_erase_saved_notes(
        self, isolated_sessions_root, synth_session
    ):
        from fastapi.testclient import TestClient
        from webapp.api import app
        client = TestClient(app)
        name, _, _ = synth_session
        client.post(f"/api/sessions/{name}/metadata",
                    json={"participant_id": "P001", "fitzpatrick": 3,
                          "notes": "important note", "channel_sites": {"0": "finger"}})
        client.post(f"/api/sessions/{name}/metadata",
                    json={"participant_id": "P001", "fitzpatrick": 3,
                          "notes": "", "channel_sites": {"0": "finger"}})
        r = client.get(f"/api/sessions/{name}")
        assert r.json()["participant"]["notes"] == "important note"


# ── /api/sessions/{name}/window ─────────────────────────────────────────────

class TestWindowEndpoint:

    def test_get_default_is_none_bounds(self, client, synth_session):
        name, _, _ = synth_session
        r = client.get(f"/api/sessions/{name}/window")
        assert r.status_code == 200
        assert r.json() == {"start_s": None, "end_s": None}

    def test_post_saves_window_and_appends_history(
            self, client, synth_session):
        name, _, _ = synth_session
        r = client.post(f"/api/sessions/{name}/window",
                         params={"start_s": 5.0, "end_s": 15.0})
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["window"] == {"start_s": 5.0, "end_s": 15.0}

        # Persisted, and a get reflects it.
        r2 = client.get(f"/api/sessions/{name}/window")
        assert r2.json() == {"start_s": 5.0, "end_s": 15.0}

        # History event appended.
        events = sessions.read_history(name)
        assert "best_window_saved" in [e["event"] for e in events]

    def test_post_no_bounds_clears_window(self, client, synth_session):
        name, _, _ = synth_session
        sessions.save_session_window(name, 5.0, 15.0)
        r = client.post(f"/api/sessions/{name}/window")
        assert r.status_code == 200
        assert r.json()["window"] == {"start_s": None, "end_s": None}

    def test_invalid_name(self, client):
        r = client.get("/api/sessions/bad_name/window")
        assert r.status_code == 400
        r = client.post("/api/sessions/bad_name/window",
                        params={"start_s": 1.0})
        assert r.status_code == 400

    def test_nonexistent_session_returns_404(self, client):
        r = client.get("/api/sessions/session_99990101_000000/window")
        assert r.status_code == 404


# ── POST /api/analyze_all?use_saved_windows ─────────────────────────────────

class TestAnalyzeAllSavedWindows:

    def test_crops_each_session_to_its_saved_window(
            self, client, synth_session):
        name, _, _ = synth_session  # synthetic session is 30 s long
        sessions.save_session_window(name, 5.0, 15.0)
        r = client.post("/api/analyze_all",
                        params={"use_saved_windows": "true"})
        assert r.status_code == 200
        body = r.json()
        assert body["use_saved_windows"] is True
        assert body["crop_window"] == {"per_session": True}

        sx = body["sessions"][0]
        assert sx["crop_window"] == {"start_s": 5.0, "end_s": 15.0}
        assert sx["ecg"]["duration_s"] == pytest.approx(10.0, abs=0.2)

    def test_session_without_window_uses_full_length(
            self, client, synth_session):
        name, _, _ = synth_session
        # No window saved — batch must fall back to full length.
        r = client.post("/api/analyze_all",
                        params={"use_saved_windows": "true"})
        assert r.status_code == 200
        sx = r.json()["sessions"][0]
        assert sx["crop_window"] == {"start_s": None, "end_s": None}
        assert sx["ecg"]["duration_s"] == pytest.approx(30.0, abs=0.5)


# ── GET /api/sessions/{name}/signals (sanity, not deep) ────────────────────

class TestSignalsEndpoint:

    def test_returns_ecg_and_channels(self, client, synth_session):
        name, _, _ = synth_session
        r = client.get(f"/api/sessions/{name}/signals",
                        params={"max_points": 500})
        assert r.status_code == 200
        body = r.json()
        assert "ecg" in body
        assert "channels" in body
        assert len(body["channels"]) == 3


# ── /api/recording/* (Popen mocked) ────────────────────────────────────────

class TestRecordingEndpoints:
    """The recorder is created at api-module import time and is a
    module-level singleton instance. Replacing it per-test keeps state
    out of one test bleeding into another."""

    def test_status_idle(self, client):
        # Replace _recorder with a fresh instance before reading status.
        from webapp import recorder as _r
        with patch.object(api, "_recorder", _r.Recorder()):
            r = client.get("/api/recording/status")
            assert r.status_code == 200
            body = r.json()
            assert body["active"] is False

    def test_start_then_stop(self, client):
        from webapp import recorder as _r
        fresh = _r.Recorder()
        with patch.object(api, "_recorder", fresh), \
             patch("webapp.recorder.subprocess.Popen") as mock_popen, \
             patch("webapp.recorder.threading.Thread"):
            proc = MagicMock()
            proc.poll.return_value = None
            proc.returncode = None
            proc.wait.return_value = 0
            proc.stdout = MagicMock()
            mock_popen.return_value = proc

            r = client.post("/api/recording/start",
                             json={"port": "COM_TEST", "baud": 115200})
            assert r.status_code == 200
            body = r.json()
            assert "session_name" in body and "session_dir" in body

            # Status reflects active.
            r2 = client.get("/api/recording/status")
            assert r2.json()["active"] is True

            # After flipping poll() to "exited cleanly", stop succeeds.
            proc.poll.return_value = 0
            proc.returncode = 0
            r3 = client.post("/api/recording/stop")
            assert r3.status_code == 200
            assert r3.json() == {"stopped": True}

    def test_double_start_returns_409(self, client):
        from webapp import recorder as _r
        fresh = _r.Recorder()
        with patch.object(api, "_recorder", fresh), \
             patch("webapp.recorder.subprocess.Popen") as mock_popen, \
             patch("webapp.recorder.threading.Thread"):
            proc = MagicMock()
            proc.poll.return_value = None
            proc.stdout = MagicMock()
            mock_popen.return_value = proc

            r1 = client.post("/api/recording/start",
                              json={"port": "COM_TEST", "baud": 115200})
            assert r1.status_code == 200
            r2 = client.post("/api/recording/start",
                              json={"port": "COM_TEST", "baud": 115200})
            assert r2.status_code == 409
