"""
Session discovery + participant metadata persistence.

A session is a folder named `session_YYYYMMDD_HHMMSS/` sitting in the
repository root (the same convention PPG_ECG_Full_Unpacking.py uses).
Each session may contain:

    ecg_data.csv              col0=ts_us, col1=sample, col2=leads_off
    ppg_data_ch{N}.csv        col0=ts_us, col1=sample
    participant.json          written by the webapp (FST, site map, ...)

The metadata file is owned by the webapp; the receiver never touches it.
That separation means a session created from the CLI stays valid — it
just has no participant metadata until the webapp adds one.
"""

import glob
import json
import os
import re
import shutil
from datetime import datetime

SESSION_PATTERN = re.compile(r"^session_(\d{8})_(\d{6})$")
BATCH_PATTERN = re.compile(r"^batch_\d{8}_\d{6}$")
PARTICIPANT_FILENAME = "participant.json"
HISTORY_FILENAME = "history.jsonl"
ANALYSIS_FILENAME = "analysis.json"
RECEIVER_LOG_FILENAME = "receiver.log"
SESSIONS_SUBDIR = "MDPIdata"
BATCH_SUBDIR = "batch_analyses"

# Fixed mux-lane → body-site wiring, identical across every session on
# this rig. Used as the fallback whenever a session's participant.json
# leaves a channel's site blank; an explicit non-empty value in the file
# still wins.
DEFAULT_CHANNEL_SITES = {
    "0": "finger",
    "1": "earlobe",
    "2": "shoulder",
    "3": "forehead",
    "4": "wrist",
}


def _with_default_sites(meta):
    """Merge DEFAULT_CHANNEL_SITES into a metadata dict without clobbering
    any site the file set explicitly."""
    meta = dict(meta or {})
    sites = dict(meta.get("channel_sites") or {})
    for ch, site in DEFAULT_CHANNEL_SITES.items():
        if not sites.get(ch):
            sites[ch] = site
    meta["channel_sites"] = sites
    meta.setdefault("participant_id", "")
    meta.setdefault("fitzpatrick", None)
    meta.setdefault("notes", "")
    return meta


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sessions_root():
    """Folder that actually holds the session_*/ recordings.

    Recordings live flat under MDPIdata/ (MDPIdata/session_<ts>/), not the
    repo root. Created on demand so a fresh checkout works first run."""
    root = os.path.join(repo_root(), SESSIONS_SUBDIR)
    os.makedirs(root, exist_ok=True)
    return root


def session_path(name):
    return os.path.join(sessions_root(), name)


def list_sessions():
    """Return summary dicts for every session_*/ folder, newest first."""
    out = []
    for entry in os.listdir(sessions_root()):
        if not SESSION_PATTERN.match(entry):
            continue
        full = os.path.join(sessions_root(), entry)
        if not os.path.isdir(full):
            continue
        out.append(summarize_session(entry, lightweight=True))
    out.sort(key=lambda s: s["name"], reverse=True)
    return out


def summarize_session(name, lightweight=False):
    """Inspect a session folder and report what's in it.

    ``lightweight=True`` skips the duration estimate (which has to read
    the tail of ecg_data.csv) so the session list endpoint stays snappy
    even with hundreds of sessions on disk."""
    full = session_path(name)
    ecg_path = os.path.join(full, "ecg_data.csv")
    has_ecg = os.path.isfile(ecg_path)

    channels = []
    for p in sorted(glob.glob(os.path.join(full, "ppg_data_ch*.csv"))):
        m = re.match(r".*ppg_data_ch(\d+)\.csv$", p.replace("\\", "/"))
        if m:
            channels.append(int(m.group(1)))

    participant = load_participant_metadata(name)
    started_at = parse_timestamp_from_name(name)
    duration_s = None if lightweight else estimate_duration(full)

    # Persistence-layer additions: every session row should expose enough
    # state that the dashboard can render "analysed N times, last on …"
    # without a follow-up fetch per row. Each helper swallows I/O errors
    # so one corrupt file in one session never 500s the whole list.
    last_analyzed_at = _read_analysis_timestamp(name)
    history_count, analysis_count = _summarize_history_counts(name)
    has_receiver_log = os.path.isfile(os.path.join(full, RECEIVER_LOG_FILENAME))

    return {
        "name": name,
        "path": full,
        "started_at": started_at,
        "duration_s": duration_s,
        "has_ecg": has_ecg,
        "channels": channels,
        "participant": participant,
        "last_analyzed_at": last_analyzed_at,
        "analysis_count": analysis_count,
        "has_receiver_log": has_receiver_log,
        "history_count": history_count,
    }


def parse_timestamp_from_name(name):
    m = SESSION_PATTERN.match(name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").isoformat()
    except ValueError:
        return None


def estimate_duration(session_dir):
    """Read first + last timestamps in ecg_data.csv and difference them.

    Streams the file instead of loading it whole — a 10-min @ 400 Hz
    session is ~7 MB, which adds up across hundreds of sessions."""
    ecg_path = os.path.join(session_dir, "ecg_data.csv")
    if not os.path.isfile(ecg_path):
        return None
    try:
        with open(ecg_path, "r") as f:
            first = f.readline().strip()
            last = first
            for line in f:
                line = line.strip()
                if line:
                    last = line
        if not first or not last:
            return None
        t0 = float(first.split(",")[0])
        t1 = float(last.split(",")[0])
        return (t1 - t0) / 1e6
    except (IOError, ValueError, IndexError):
        return None


def load_participant_metadata(name):
    """Always returns a usable metadata dict (never None) with the fixed
    channel→site map filled in, so every consumer — the session list,
    the detail meta grid, and the SQI table — labels channels the same
    way even for sessions captured before participant.json existed."""
    p = os.path.join(session_path(name), PARTICIPANT_FILENAME)
    if not os.path.isfile(p):
        return _with_default_sites({})
    try:
        with open(p, "r") as f:
            return _with_default_sites(json.load(f))
    except (IOError, json.JSONDecodeError):
        return _with_default_sites({})


def save_participant_metadata(name, metadata):
    full = session_path(name)
    os.makedirs(full, exist_ok=True)
    with open(os.path.join(full, PARTICIPANT_FILENAME), "w") as f:
        json.dump(metadata, f, indent=2)


def delete_session(name):
    """Permanently remove a session_*/ folder and everything in it.

    Defensive on purpose: the name must match SESSION_PATTERN and the
    resolved path must sit directly inside sessions_root(), so a crafted
    name can't rmtree something outside the data directory.
    """
    if not SESSION_PATTERN.match(name):
        raise ValueError("invalid session name")
    full = os.path.abspath(session_path(name))
    root = os.path.abspath(sessions_root())
    if os.path.dirname(full) != root:
        raise ValueError("refusing to delete path outside sessions root")
    if not os.path.isdir(full):
        raise FileNotFoundError(name)
    shutil.rmtree(full)


# ── Persistence layer ───────────────────────────────────────────────────────
#
# Everything below this line writes durable per-session artefacts inside
# MDPIdata/session_<ts>/ (and the global MDPIdata/batch_analyses/ folder)
# so a recording, its metadata edits, and every analysis run survive a
# webapp restart. Every helper is wrapped in try/except + warning print:
# a broken file must never raise into the FastAPI handler.

def _utc_now_iso():
    """Single source of truth for timestamps written by the persistence
    layer. Suffix Z so consumers don't mistake the value for naive local
    time."""
    return datetime.utcnow().isoformat() + "Z"


def batch_analyses_root():
    """Folder that holds the saved batch snapshots. Created on demand so
    the first /api/analyze_all call works on a fresh checkout."""
    root = os.path.join(sessions_root(), BATCH_SUBDIR)
    os.makedirs(root, exist_ok=True)
    return root


def _history_path(name):
    return os.path.join(session_path(name), HISTORY_FILENAME)


def _analysis_path(name):
    return os.path.join(session_path(name), ANALYSIS_FILENAME)


def _receiver_log_path(name):
    return os.path.join(session_path(name), RECEIVER_LOG_FILENAME)


def _coerce_jsonable(obj):
    """Strip NaN/Inf out of a JSON tree so the file we write is parseable
    by strict JSON consumers (the spec compact event summary holds CCC /
    ICC values which can legitimately be float('nan'))."""
    import math
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _coerce_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_coerce_jsonable(v) for v in obj]
    return obj


def append_history(name, event, data):
    """Append one event to the session's history.jsonl. Tolerant of
    every realistic failure mode: missing folder, locked file, NaN
    floats in ``data``. Returns True on success, False otherwise."""
    try:
        full = session_path(name)
        os.makedirs(full, exist_ok=True)
        record = {
            "ts": _utc_now_iso(),
            "event": str(event),
            "data": _coerce_jsonable(data) if data is not None else {},
        }
        line = json.dumps(record, separators=(",", ":")) + "\n"
        with open(_history_path(name), "a", encoding="utf-8") as f:
            f.write(line)
        return True
    except (IOError, OSError, ValueError, TypeError) as e:
        print(f"[sessions] append_history({name}, {event}) failed: {e}")
        return False


def read_history(name, limit=500):
    """Return up to ``limit`` parsed events from history.jsonl, newest
    first. Skips lines that don't parse so one corrupt write doesn't
    break the response. Empty list when the file is absent."""
    path = _history_path(name)
    if not os.path.isfile(path):
        return []
    out = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    out.append(json.loads(raw))
                except json.JSONDecodeError:
                    continue
    except (IOError, OSError) as e:
        print(f"[sessions] read_history({name}) failed: {e}")
        return []
    out.reverse()
    if limit and limit > 0:
        out = out[: int(limit)]
    return out


def save_session_analysis(name, result, crop_window):
    """Write analysis.json for one session. The endpoint hands us the
    raw `analyze_session` result plus the crop window — we add the
    `analyzed_at` timestamp and the `crop_window` block itself here so
    the on-disk file is fully self-describing without forcing the
    analysis module to know about persistence concerns."""
    try:
        full = session_path(name)
        os.makedirs(full, exist_ok=True)
        payload = dict(result or {})
        payload["analyzed_at"] = _utc_now_iso()
        payload["crop_window"] = {
            "start_s": (None if crop_window is None else crop_window.get("start_s")),
            "end_s":   (None if crop_window is None else crop_window.get("end_s")),
        }
        with open(_analysis_path(name), "w", encoding="utf-8") as f:
            json.dump(_coerce_jsonable(payload), f, indent=2)
        return True
    except (IOError, OSError, ValueError, TypeError) as e:
        print(f"[sessions] save_session_analysis({name}) failed: {e}")
        return False


def load_session_analysis(name):
    """Return the cached analysis.json payload, or None if absent /
    unparseable. Used by GET /api/sessions/{name}/analysis."""
    path = _analysis_path(name)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (IOError, OSError, json.JSONDecodeError) as e:
        print(f"[sessions] load_session_analysis({name}) failed: {e}")
        return None


def save_receiver_log(name, text):
    """Persist the recorder subprocess's captured stdout/stderr to
    receiver.log in the session folder. Called from Recorder.stop() and
    from Recorder.status() when an unattended subprocess exit is
    detected."""
    try:
        full = session_path(name)
        os.makedirs(full, exist_ok=True)
        with open(_receiver_log_path(name), "w", encoding="utf-8", newline="") as f:
            f.write(text if text is not None else "")
        return True
    except (IOError, OSError, TypeError) as e:
        print(f"[sessions] save_receiver_log({name}) failed: {e}")
        return False


def tail_receiver_log(name, n=500):
    """Return the last ``n`` lines of receiver.log as a single string,
    or "" if the file doesn't exist / can't be read."""
    path = _receiver_log_path(name)
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except (IOError, OSError) as e:
        print(f"[sessions] tail_receiver_log({name}) failed: {e}")
        return ""
    if n and n > 0:
        lines = lines[-int(n):]
    return "".join(lines)


def _read_analysis_timestamp(name):
    """Lightweight: cheap-to-call helper used by summarize_session. Pulls
    only the `analyzed_at` field rather than the full result blob."""
    path = _analysis_path(name)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        ts = payload.get("analyzed_at") if isinstance(payload, dict) else None
        return ts if isinstance(ts, str) else None
    except (IOError, OSError, json.JSONDecodeError):
        return None


def _summarize_history_counts(name):
    """Return (total_events, analysis_run_events) by streaming the file —
    avoids json.loads on each line for what's basically a row-count."""
    path = _history_path(name)
    if not os.path.isfile(path):
        return 0, 0
    total = 0
    analyses = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                total += 1
                if '"event":"analysis_run"' in line:
                    analyses += 1
    except (IOError, OSError):
        return 0, 0
    return total, analyses


# ── Batch analysis archive ──────────────────────────────────────────────────

def save_batch_analysis(payload):
    """Persist a /api/analyze_all snapshot under MDPIdata/batch_analyses/
    as batch_<YYYYMMDD_HHMMSS>.json. Mutates ``payload`` by adding
    ``batch_id`` and ``created_at`` so the returned object and the
    on-disk one stay identical."""
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    batch_id = f"batch_{stamp}"
    payload = dict(payload or {})
    payload["batch_id"] = batch_id
    payload["created_at"] = _utc_now_iso()
    try:
        path = os.path.join(batch_analyses_root(), f"{batch_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_coerce_jsonable(payload), f, indent=2)
    except (IOError, OSError, TypeError) as e:
        print(f"[sessions] save_batch_analysis({batch_id}) failed: {e}")
    return payload


def list_batch_analyses():
    """List saved batch snapshots, newest first. Returns shallow
    `{batch_id, created_at, n_sessions_analyzed, crop_window}` rows,
    not the full payloads (those go through load_batch_analysis)."""
    root = batch_analyses_root()
    out = []
    try:
        entries = os.listdir(root)
    except OSError:
        return out
    for entry in entries:
        if not entry.endswith(".json"):
            continue
        stem = entry[:-5]
        if not BATCH_PATTERN.match(stem):
            continue
        full = os.path.join(root, entry)
        try:
            with open(full, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (IOError, OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        out.append({
            "batch_id": payload.get("batch_id", stem),
            "created_at": payload.get("created_at"),
            "n_sessions_analyzed": payload.get("n_sessions_analyzed", 0),
            "crop_window": payload.get("crop_window") or {"start_s": None, "end_s": None},
        })
    out.sort(key=lambda r: r.get("batch_id") or "", reverse=True)
    return out


def load_batch_analysis(batch_id):
    """Load one saved batch snapshot by id. Returns None if the id is
    malformed, points outside the archive folder, or the file can't be
    read."""
    if not BATCH_PATTERN.match(batch_id or ""):
        return None
    root = os.path.abspath(batch_analyses_root())
    full = os.path.abspath(os.path.join(root, f"{batch_id}.json"))
    # Defensive: same path-confinement check delete_session uses.
    if os.path.dirname(full) != root:
        return None
    if not os.path.isfile(full):
        return None
    try:
        with open(full, "r", encoding="utf-8") as f:
            return json.load(f)
    except (IOError, OSError, json.JSONDecodeError) as e:
        print(f"[sessions] load_batch_analysis({batch_id}) failed: {e}")
        return None
