"""
FastAPI surface for the SEAL PPG webapp.

Routes are intentionally thin — every endpoint forwards to one function
in webapp/sessions.py, webapp/analysis.py, or the Recorder singleton.
"""

import os
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import serial.tools.list_ports

from . import analysis, recorder, sessions, sleepiness

app = FastAPI(title="SEAL PPG Acquisition & Analysis")

_recorder = recorder.Recorder()

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Schemas ──────────────────────────────────────────────────────────────────

class ParticipantMetadata(BaseModel):
    participant_id: str = ""
    fitzpatrick: Optional[int] = None   # 1-6
    notes: str = ""
    channel_sites: Dict[str, str] = Field(default_factory=dict)


class StartRecordingRequest(BaseModel):
    port: str = "COM3"
    baud: int = 115200
    participant: Optional[ParticipantMetadata] = None


# ── Root / static ────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# ── Ports ────────────────────────────────────────────────────────────────────

@app.get("/api/ports")
def list_ports() -> List[Dict[str, str]]:
    return [
        {"device": p.device, "description": p.description or ""}
        for p in serial.tools.list_ports.comports()
    ]


# ── Sessions ─────────────────────────────────────────────────────────────────

def _require_session(name: str):
    if not sessions.SESSION_PATTERN.match(name):
        raise HTTPException(400, "invalid session name")
    if not os.path.isdir(sessions.session_path(name)):
        raise HTTPException(404, f"session not found: {name}")


@app.get("/api/sessions")
def get_sessions():
    return sessions.list_sessions()


@app.get("/api/sessions/{name}")
def get_session(name: str):
    _require_session(name)
    return sessions.summarize_session(name)


@app.post("/api/sessions/{name}/metadata")
def post_metadata(name: str, meta: ParticipantMetadata):
    _require_session(name)
    # Capture the previous metadata before overwriting so the
    # `metadata_edited` history event carries a diff-friendly
    # before/after pair (matches the spec's history schema).
    before = sessions.load_participant_metadata(name)
    after = meta.dict()
    sessions.save_participant_metadata(name, after)
    sessions.append_history(name, "metadata_edited", {
        "before": before,
        "after": after,
    })
    return {"ok": True}


@app.delete("/api/sessions/{name}")
def delete_session(name: str):
    _require_session(name)
    status = _recorder.status()
    if status.get("active") and status.get("session_name") == name:
        raise HTTPException(409, "cannot delete the session that is currently recording")
    try:
        sessions.delete_session(name)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@app.get("/api/sessions/{name}/signals")
def get_signals(name: str, max_points: int = 5000, tail_seconds: Optional[float] = None,
                start_s: Optional[float] = None, end_s: Optional[float] = None):
    _require_session(name)
    return analysis.load_session_signals(
        name, max_points=max_points, tail_seconds=tail_seconds,
        start_s=start_s, end_s=end_s,
    )


def _compact_analysis_summary(result):
    """Per-channel {ccc, icc, matched} dict used inside the history
    `analysis_run` event. Keeps history.jsonl tiny while the full result
    lives in analysis.json. NaN floats become None for strict-JSON safety."""
    import math
    summary = {}
    for row in (result.get("results") or []):
        ch = row.get("channel")
        if ch is None:
            continue
        stats = row.get("stats") or {}
        def _clean(v):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return v
        summary[str(ch)] = {
            "ccc": _clean(stats.get("ccc")),
            "icc": _clean(stats.get("icc")),
            "matched": int(row.get("n_matched_beats") or 0),
        }
    return summary


def _persist_session_analysis(name, result, start_s, end_s):
    """Write analysis.json + append the analysis_run history event.
    Errors are swallowed by the sessions.* helpers so a write failure
    never makes the analyse endpoint return 500 — the caller still gets
    the live result they asked for."""
    crop = {"start_s": start_s, "end_s": end_s}
    sessions.save_session_analysis(name, result, crop)
    sessions.append_history(name, "analysis_run", {
        "crop_window": crop,
        "n_channels": len(result.get("results") or []),
        "summary": _compact_analysis_summary(result),
    })


@app.post("/api/sessions/{name}/analyze")
def analyze_session(name: str, start_s: Optional[float] = None,
                    end_s: Optional[float] = None):
    _require_session(name)
    result = analysis.analyze_session(name, start_s=start_s, end_s=end_s)
    _persist_session_analysis(name, result, start_s, end_s)
    return result


# Batch over every session_*/ folder under MDPIdata/. Same crop window
# semantics as the single-session endpoint — start_s/end_s are seconds
# since each session's ECG t0, applied independently per session.
@app.post("/api/analyze_all")
def analyze_all(start_s: Optional[float] = None, end_s: Optional[float] = None):
    result = analysis.analyze_all_sessions(start_s=start_s, end_s=end_s)
    # Save the batch snapshot under MDPIdata/batch_analyses/. The helper
    # mutates result with batch_id + created_at so what we return and
    # what's on disk are byte-identical.
    result = sessions.save_batch_analysis(result)
    batch_id = result.get("batch_id")
    sessions_list = result.get("sessions") or []
    total = len(sessions_list)
    for i, s in enumerate(sessions_list, start=1):
        sname = s.get("session_name")
        if not sname:
            continue
        sessions.append_history(sname, "batch_analysis_included", {
            "batch_id": batch_id,
            "session_position": i,
            "total": total,
        })
    return result


# ── Persistence read endpoints ───────────────────────────────────────────────

@app.get("/api/sessions/{name}/history")
def get_session_history(name: str, limit: int = 500):
    _require_session(name)
    return sessions.read_history(name, limit=limit)


@app.get("/api/sessions/{name}/analysis")
def get_session_analysis(name: str):
    _require_session(name)
    cached = sessions.load_session_analysis(name)
    if cached is None:
        return {"cached": False}
    return cached


@app.get("/api/sessions/{name}/receiver_log")
def get_session_receiver_log(name: str, tail: int = 500):
    _require_session(name)
    return {"log": sessions.tail_receiver_log(name, n=tail)}


@app.get("/api/batch_analyses")
def get_batch_analyses():
    return sessions.list_batch_analyses()


@app.get("/api/batch_analyses/{batch_id}")
def get_batch_analysis(batch_id: str):
    if not sessions.BATCH_PATTERN.match(batch_id):
        raise HTTPException(400, "invalid batch id")
    payload = sessions.load_batch_analysis(batch_id)
    if payload is None:
        raise HTTPException(404, f"batch not found: {batch_id}")
    return payload


# ── Cohort sleepiness summary (SPI) ─────────────────────────────────────────

@app.post("/api/sleepiness_summary")
def post_sleepiness_summary(weighting: str = "ssqi_zsqi",
                            start_s: Optional[float] = None,
                            end_s: Optional[float] = None):
    """Compute the cohort sleepiness proxy (SPI) across MDPIdata/.

    Same crop-window semantics as `/api/analyze_all`. Persists the full
    payload to ``MDPIdata/sleepiness_runs/run_<ts>.json`` and appends a
    ``sleepiness_analysis_included`` event to each contributing session's
    ``history.jsonl`` so the per-session timeline knows the session
    contributed to a cohort run.
    """
    result = sleepiness.analyze_sleepiness(weighting=weighting,
                                            start_s=start_s, end_s=end_s)
    result = sessions.save_sleepiness_run(result)
    run_id = result.get("run_id")
    per_session_list = result.get("per_session") or []
    total = len(per_session_list)
    for i, s in enumerate(per_session_list, start=1):
        sname = s.get("session_name")
        if not sname:
            continue
        sessions.append_history(sname, "sleepiness_analysis_included", {
            "run_id": run_id,
            "session_position": i,
            "total": total,
            "ppg_spi_weighted": s.get("ppg_spi_weighted"),
            "ecg_spi": (s.get("ecg") or {}).get("spi"),
            "usable": s.get("usable"),
        })
    # Coerce NaN/Inf to None so the JSON we return is parseable by
    # strict consumers (the frontend uses fetch().json() which rejects
    # bare NaN tokens). The on-disk file is already coerced by
    # save_sleepiness_run.
    return sessions._coerce_jsonable(result)


@app.get("/api/sleepiness_summary/latest")
def get_sleepiness_summary_latest():
    """Return the most-recent saved sleepiness-SPI run, or
    ``{"cached": false}`` when no run has been saved yet."""
    payload = sessions.load_latest_sleepiness_run()
    if payload is None:
        return {"cached": False}
    return sessions._coerce_jsonable(payload)


@app.get("/api/sleepiness_runs")
def get_sleepiness_runs():
    """List saved sleepiness runs (shallow rows, newest first)."""
    return sessions.list_sleepiness_runs()


@app.get("/api/sleepiness_runs/{run_id}")
def get_sleepiness_run(run_id: str):
    if not sessions.SLEEPINESS_PATTERN.match(run_id):
        raise HTTPException(400, "invalid sleepiness run id")
    payload = sessions.load_sleepiness_run(run_id)
    if payload is None:
        raise HTTPException(404, f"sleepiness run not found: {run_id}")
    return sessions._coerce_jsonable(payload)


# ── Recording lifecycle ──────────────────────────────────────────────────────

@app.post("/api/recording/start")
def start_recording(req: StartRecordingRequest):
    try:
        session_name, session_dir = _recorder.start(req.port, req.baud)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    if req.participant is not None:
        sessions.save_participant_metadata(session_name, req.participant.dict())
    return {"session_name": session_name, "session_dir": session_dir}


@app.post("/api/recording/stop")
def stop_recording():
    stopped = _recorder.stop()
    return {"stopped": stopped}


@app.get("/api/recording/status")
def recording_status():
    return _recorder.status()
