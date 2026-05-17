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

from . import analysis, recorder, sessions

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
    sessions.save_participant_metadata(name, meta.dict())
    return {"ok": True}


@app.get("/api/sessions/{name}/signals")
def get_signals(name: str, max_points: int = 5000, tail_seconds: Optional[float] = None):
    _require_session(name)
    return analysis.load_session_signals(
        name, max_points=max_points, tail_seconds=tail_seconds,
    )


@app.post("/api/sessions/{name}/analyze")
def analyze_session(name: str):
    _require_session(name)
    return analysis.analyze_session(name)


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
