"""Unit tests for webapp/recorder.py.

The recorder spawns a child process and reads its stdout. We never let
the real subprocess.Popen fire — every test patches it (and the
background drain thread) so the recorder is exercised entirely
in-memory. Tests run under the isolated_sessions_root fixture so the
pre-created session_<ts>/ folders land in tmp_path.
"""

import os
import signal
from unittest.mock import MagicMock, patch

import pytest

from webapp import recorder, sessions


@pytest.fixture
def patched_recorder(isolated_sessions_root):
    """Yield a fresh Recorder with subprocess.Popen + threading.Thread
    patched. ``mock_popen`` is exposed so the test can drive
    ``returncode`` / ``poll()`` semantics.

    By default the mocked subprocess is "alive" (poll() returns None).
    """
    with patch("webapp.recorder.subprocess.Popen") as mock_popen, \
         patch("webapp.recorder.threading.Thread") as mock_thread:
        proc = MagicMock()
        proc.poll.return_value = None       # alive by default
        proc.returncode = None
        proc.stdout = MagicMock()
        mock_popen.return_value = proc
        # The drain thread mock just records start() was called.
        mock_thread.return_value.start = MagicMock()
        yield {
            "recorder": recorder.Recorder(),
            "popen": mock_popen,
            "proc": proc,
            "thread": mock_thread,
            "sessions_root": isolated_sessions_root,
        }


class TestRecorderLifecycle:

    def test_status_when_idle(self, patched_recorder):
        r = patched_recorder["recorder"]
        st = r.status()
        for k in ("active", "session_name", "session_dir", "started_at",
                  "port", "baud", "sample_counts", "recent_log", "exit_code"):
            assert k in st
        assert st["active"] is False
        assert st["session_name"] is None
        assert st["port"] is None

    def test_start_creates_session_dir(self, patched_recorder):
        r = patched_recorder["recorder"]
        name, sdir = r.start("COM_TEST", 115200)
        assert sessions.SESSION_PATTERN.match(name)
        assert os.path.isdir(sdir)
        # The fixture root was passed through the monkeypatched
        # sessions_root(); confirm the new folder sits inside it.
        assert os.path.dirname(os.path.abspath(sdir)) == \
               os.path.abspath(patched_recorder["sessions_root"])
        # Popen must have been called with our Python entry point.
        patched_recorder["popen"].assert_called_once()

    def test_status_after_start_active(self, patched_recorder):
        r = patched_recorder["recorder"]
        name, _ = r.start("COM_TEST", 115200)
        st = r.status()
        assert st["active"] is True
        assert st["session_name"] == name
        assert st["port"] == "COM_TEST"
        assert st["baud"] == 115200

    def test_double_start_rejected(self, patched_recorder):
        """The Recorder pattern uses an internal lock + alive-check
        rather than an __new__ singleton — two simultaneous start()
        calls on the same instance must raise."""
        r = patched_recorder["recorder"]
        r.start("COM_TEST", 115200)
        with pytest.raises(RuntimeError, match="already active"):
            r.start("COM_TEST", 115200)

    def test_start_appends_history_event(self, patched_recorder):
        r = patched_recorder["recorder"]
        name, _ = r.start("COM_TEST", 9600)
        history = sessions.read_history(name)
        # recording_started should be the only event right after start.
        events = [e["event"] for e in history]
        assert "recording_started" in events
        rec = next(e for e in history if e["event"] == "recording_started")
        assert rec["data"]["port"] == "COM_TEST"
        assert rec["data"]["baud"] == 9600


class TestRecorderStop:

    def test_stop_when_idle_returns_false(self, patched_recorder):
        r = patched_recorder["recorder"]
        assert r.stop() is False

    def test_stop_persists_receiver_log_and_history(self, patched_recorder):
        r = patched_recorder["recorder"]
        proc = patched_recorder["proc"]
        name, _ = r.start("COM_TEST", 115200)
        # Simulate the child writing two log lines before we stop it.
        r._full_log.extend(["boot ok", "starting capture"])
        # The stop path calls proc.wait() — make it succeed cleanly.
        proc.wait.return_value = 0
        proc.returncode = 0

        assert r.stop() is True

        # receiver.log was written.
        log_text = sessions.tail_receiver_log(name)
        assert "boot ok" in log_text
        assert "starting capture" in log_text

        # recording_stopped event appended.
        events = sessions.read_history(name)
        assert any(e["event"] == "recording_stopped" for e in events)

    def test_status_detects_unattended_exit(self, patched_recorder):
        r = patched_recorder["recorder"]
        proc = patched_recorder["proc"]
        name, _ = r.start("COM_TEST", 115200)
        # Subprocess died on its own (poll returns non-None) — the next
        # status() call must persist the artefacts.
        proc.poll.return_value = 137
        proc.returncode = 137
        r._full_log.append("crashed")

        st = r.status()
        # After persistence, the receiver.log + history event should
        # exist even though we never called stop().
        assert os.path.isfile(os.path.join(
            sessions.session_path(name), sessions.RECEIVER_LOG_FILENAME))
        events = sessions.read_history(name)
        assert any(e["event"] == "recording_stopped" for e in events)
        # exit_code surfaced in status.
        assert st["exit_code"] == 137


class TestRecorderInternals:

    def test_count_lines_missing_file(self):
        # _count_lines is a module-level helper; safe to call directly.
        assert recorder._count_lines("/nonexistent/path/file.csv") == 0

    def test_count_lines_with_content(self, tmp_path):
        p = tmp_path / "file.csv"
        p.write_text("a\nb\nc\n", encoding="utf-8")
        assert recorder._count_lines(str(p)) == 3
