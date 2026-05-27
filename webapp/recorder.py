"""
Singleton wrapper around the PPG_ECG_Full_Unpacking.py serial receiver.

The webapp owns one Recorder; ``start()`` spawns the receiver as a child
process and ``stop()`` sends it Ctrl+Break (on Windows) so its
``except KeyboardInterrupt`` handler runs and flushes the CSVs cleanly.

We pre-create the session_<ts>/ folder and hand it to the child via
``SEAL_PPG_SESSION_DIR`` — that way the webapp always knows the exact
output path without parsing the child's stdout.
"""

import glob
import os
import signal
import subprocess
import sys
import threading
from collections import deque
from datetime import datetime

from . import sessions

REPO_ROOT = sessions.repo_root()
RECEIVER_SCRIPT = os.path.join(REPO_ROOT, "PPG_ECG_Full_Unpacking.py")
MAX_LOG_LINES = 200


class Recorder:
    def __init__(self):
        self._proc = None
        self._session_name = None
        self._session_dir = None
        self._started_at = None
        self._port = None
        self._baud = None
        # Two parallel buffers: `_log` is the bounded recent tail surfaced
        # in status() for the live UI; `_full_log` is the unbounded
        # capture we dump to receiver.log when the subprocess stops. The
        # spec requires the full log on disk so a 30-minute recording's
        # boot/error lines don't get rotated out.
        self._log = deque(maxlen=MAX_LOG_LINES)
        self._full_log = []
        # Bookkeeping for the "subprocess exited on its own" path: once
        # we've persisted receiver.log + appended recording_stopped we
        # set this to True so a subsequent status() poll doesn't
        # double-write. start() resets it.
        self._log_persisted = True
        self._lock = threading.Lock()

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self, port, baud):
        with self._lock:
            if self._is_running_locked():
                raise RuntimeError("Recording already active")

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_name = f"session_{stamp}"
            session_dir = os.path.join(sessions.sessions_root(), session_name)
            os.makedirs(session_dir, exist_ok=True)

            env = os.environ.copy()
            env["SEAL_PPG_PORT"] = str(port)
            env["SEAL_PPG_BAUD"] = str(int(baud))
            env["SEAL_PPG_SESSION_DIR"] = session_dir

            # CREATE_NEW_PROCESS_GROUP on Windows lets us deliver CTRL_BREAK
            # later without killing the parent (the FastAPI server).
            flags = 0
            if os.name == "nt":
                flags = subprocess.CREATE_NEW_PROCESS_GROUP

            self._proc = subprocess.Popen(
                [sys.executable, "-u", RECEIVER_SCRIPT],
                cwd=REPO_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                creationflags=flags,
            )
            self._session_name = session_name
            self._session_dir = session_dir
            self._started_at = datetime.now().isoformat()
            self._port = port
            self._baud = int(baud)
            self._log.clear()
            self._full_log = []
            # Re-arm the "persist log on exit" guard. It stays False until
            # stop() or status() handles the transition.
            self._log_persisted = False

            threading.Thread(target=self._drain_stdout, daemon=True).start()

        # Append the recording_started event outside the lock so a slow
        # disk write doesn't hold up subsequent status() polls.
        sessions.append_history(session_name, "recording_started", {
            "port": str(port),
            "baud": int(baud),
            "session_name": session_name,
        })
        return session_name, session_dir

    def stop(self, timeout=5.0):
        with self._lock:
            if self._proc is None:
                return False
            proc = self._proc

        if os.name == "nt":
            try:
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            except (OSError, ValueError):
                proc.terminate()
        else:
            proc.terminate()

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

        # Persist the run's artefacts BEFORE clearing _proc so the
        # in-memory state we read (log, session dir, sample counts, exit
        # code) all refer to the run that just ended.
        self._persist_run_artefacts(exit_code=proc.returncode)

        with self._lock:
            self._proc = None

        return True

    # ── introspection ────────────────────────────────────────────────────────

    def status(self):
        # If the subprocess died on its own since the last poll, persist
        # its log + history event now — otherwise a crash leaves the
        # session without receiver.log even though the CSVs are on disk.
        # Done outside the lock so the I/O doesn't block other callers.
        died_unattended = False
        exit_code_on_death = None
        with self._lock:
            if (self._proc is not None
                    and self._proc.poll() is not None
                    and not self._log_persisted):
                died_unattended = True
                exit_code_on_death = self._proc.returncode
        if died_unattended:
            self._persist_run_artefacts(exit_code=exit_code_on_death)

        with self._lock:
            active = self._is_running_locked()
            return {
                "active": active,
                "session_name": self._session_name,
                "session_dir": self._session_dir,
                "started_at": self._started_at,
                "port": self._port,
                "baud": self._baud,
                "sample_counts": self._sample_counts() if self._session_dir else {},
                "recent_log": list(self._log)[-30:],
                "exit_code": (None if active else
                              (self._proc.returncode if self._proc else None)),
            }

    # ── internals ────────────────────────────────────────────────────────────

    def _is_running_locked(self):
        return self._proc is not None and self._proc.poll() is None

    def _drain_stdout(self):
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for raw in iter(proc.stdout.readline, b""):
                line = raw.decode("utf-8", errors="replace").rstrip()
                if line:
                    self._log.append(line)
                    self._full_log.append(line)
        except (ValueError, OSError):
            pass

    def _persist_run_artefacts(self, exit_code):
        """Write receiver.log + append recording_stopped to history.jsonl.
        Idempotent via the _log_persisted guard — both stop() and the
        status()-detected unattended-exit path call this, and only the
        first call does the work."""
        with self._lock:
            if self._log_persisted:
                return
            session_name = self._session_name
            session_dir = self._session_dir
            started_at = self._started_at
            log_text = "\n".join(self._full_log)
            if log_text and not log_text.endswith("\n"):
                log_text += "\n"
            sample_counts = self._sample_counts() if session_dir else {}
            self._log_persisted = True

        if not session_name:
            return

        # File I/O outside the lock — sessions.* swallows its own errors,
        # so we only need to guard against the unexpected here.
        try:
            sessions.save_receiver_log(session_name, log_text)
        except Exception as e:
            print(f"[recorder] save_receiver_log failed: {e}")

        duration_s = None
        if started_at:
            try:
                t0 = datetime.fromisoformat(started_at)
                duration_s = (datetime.now() - t0).total_seconds()
            except (TypeError, ValueError):
                duration_s = None

        try:
            sessions.append_history(session_name, "recording_stopped", {
                "session_name": session_name,
                "duration_s": duration_s,
                "exit_code": exit_code,
                "sample_counts": sample_counts,
            })
        except Exception as e:
            print(f"[recorder] append_history(recording_stopped) failed: {e}")

    def _sample_counts(self):
        """Count rows in each CSV the receiver is writing.

        Files are opened by the child in 'w' mode; on Windows the default
        share mode still permits read access, so this is safe to call
        while the recording is live."""
        counts = {}
        if self._session_dir is None:
            return counts

        ecg_path = os.path.join(self._session_dir, "ecg_data.csv")
        if os.path.isfile(ecg_path):
            counts["ecg"] = _count_lines(ecg_path)

        for path in sorted(glob.glob(os.path.join(self._session_dir, "ppg_data_ch*.csv"))):
            key = os.path.splitext(os.path.basename(path))[0].replace("ppg_data_", "")
            counts[key] = _count_lines(path)

        return counts


def _count_lines(path):
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except IOError:
        return 0
