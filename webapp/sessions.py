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
from datetime import datetime

SESSION_PATTERN = re.compile(r"^session_(\d{8})_(\d{6})$")
PARTICIPANT_FILENAME = "participant.json"


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def session_path(name):
    return os.path.join(repo_root(), name)


def list_sessions():
    """Return summary dicts for every session_*/ folder, newest first."""
    out = []
    for entry in os.listdir(repo_root()):
        if not SESSION_PATTERN.match(entry):
            continue
        full = os.path.join(repo_root(), entry)
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

    return {
        "name": name,
        "path": full,
        "started_at": started_at,
        "duration_s": duration_s,
        "has_ecg": has_ecg,
        "channels": channels,
        "participant": participant,
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
    p = os.path.join(session_path(name), PARTICIPANT_FILENAME)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r") as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError):
        return None


def save_participant_metadata(name, metadata):
    full = session_path(name)
    os.makedirs(full, exist_ok=True)
    with open(os.path.join(full, PARTICIPANT_FILENAME), "w") as f:
        json.dump(metadata, f, indent=2)
