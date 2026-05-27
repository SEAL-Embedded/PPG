# API reference

Every HTTP endpoint the dashboard serves. Implementation lives in `webapp/api.py`. All paths are mounted under the same FastAPI app started by `python app.py` (defaults to `http://127.0.0.1:8000/`).

Pair with [`DATA_FORMATS.md`](DATA_FORMATS.md) for the on-disk schema of every persistence file an endpoint reads or writes.

## Table of contents

- [Static + ports](#static--ports)
- [Sessions](#sessions)
- [Analysis](#analysis)
- [Persistence reads](#persistence-reads)
- [Recording lifecycle](#recording-lifecycle)
- [Batch + archive](#batch--archive)
- [Sleepiness summary](#sleepiness-summary)
- [Error codes](#error-codes)

---

## Static + ports

### `GET /`
Returns `webapp/static/index.html`. Single-page entry point.

### `GET /api/ports`
List enumerated serial ports for the recorder.
Response: `[{"device": "COM3", "description": "USB Serial Device (COM3)"}, …]`

---

## Sessions

### `GET /api/sessions`
List every `session_*/` folder under `MDPIdata/`, newest first.
Response: array of session-summary objects:
```json
{
  "name": "session_20260516_174059",
  "path": "C:\\…\\MDPIdata\\session_20260516_174059",
  "started_at": "2026-05-16T17:40:59",
  "duration_s": null,
  "has_ecg": true,
  "channels": [0, 1, 2, 3, 4],
  "participant": {...},
  "last_analyzed_at": "2026-05-25T08:29:57.354731Z" | null,
  "analysis_count": 3,
  "history_count": 5,
  "has_receiver_log": false
}
```
This is the lightweight summary — `duration_s` is null (computing it requires reading the tail of every `ecg_data.csv`).

### `GET /api/sessions/{name}`
Full summary for one session (includes `duration_s`). Same shape as above with `duration_s` populated.
Errors: `400` on invalid name regex, `404` if folder missing.

### `POST /api/sessions/{name}/metadata`
Save participant metadata. Body schema:
```json
{
  "participant_id": "P001",
  "fitzpatrick": 3,
  "notes": "ambient light dim",
  "channel_sites": {"0": "finger", "1": "earlobe", ...}
}
```
Response: `{"ok": true}`.
**Side effects:** overwrites `MDPIdata/session_<name>/participant.json`; appends a `metadata_edited` event to `history.jsonl` with full `before`/`after` blocks.

### `DELETE /api/sessions/{name}`
Permanently remove the session folder and every file in it.
Response: `{"ok": true}`. Errors: `409` if this session is currently recording.

### `GET /api/sessions/{name}/signals`
Downsampled ECG + per-channel PPG traces for plotting. Query params:
- `max_points` (int, default 5000) — per-trace point budget
- `tail_seconds` (float, optional) — return only the last N seconds
- `start_s`, `end_s` (float, optional) — crop window (seconds since session t0)

Response: `{"ecg": {...}, "channels": [{...}]}` — each trace includes `time_s`, `signal`, `fs_hz`, `n_samples`. ECG also carries `leads_off_spans`; PPG channels also carry `time_bp_s`/`signal_bp` for the 0.6–3.3 Hz bandpass overlay.

---

## Analysis

### `POST /api/sessions/{name}/analyze`
Run the full per-session SQI + CCC + ICC + Bland-Altman + interpretation pipeline. Query params: `start_s`, `end_s` (crop window).

Response: see [`DATA_FORMATS.md`](DATA_FORMATS.md#analysisjson) — the full `analysis.json` content (with `analyzed_at`, `crop_window`, `ecg`, `results[]`, `participant`, `interpretation`).

**Side effects:**
- Writes (overwrites) `MDPIdata/session_<name>/analysis.json`.
- Appends an `analysis_run` event to `history.jsonl` (compact per-channel `{ccc, icc, matched}` summary).

---

## Persistence reads

### `GET /api/sessions/{name}/history`
Return parsed `history.jsonl` events. Query param: `limit` (int, default 500). Newest first.
Response: array of `{ts, event, data}` objects.

### `GET /api/sessions/{name}/analysis`
Return cached `analysis.json` if present. Does **NOT** re-run the analysis.
Response: full payload, or `{"cached": false}` if the file is missing.

### `GET /api/sessions/{name}/receiver_log`
Return the tail of `receiver.log`. Query param: `tail` (int, default 500 lines).
Response: `{"log": "...string..."}`.

---

## Recording lifecycle

### `POST /api/recording/start`
Start a recording subprocess. Body:
```json
{
  "port": "COM3",
  "baud": 115200,
  "participant": {...}    /* optional ParticipantMetadata */
}
```
Response: `{"session_name": "session_20260516_174059", "session_dir": "..."}`.

**Side effects:** creates the session folder, spawns `PPG_ECG_Full_Unpacking.py` as a child process with `SEAL_PPG_PORT`/`SEAL_PPG_BAUD`/`SEAL_PPG_SESSION_DIR` env vars; if `participant` is non-null, writes `participant.json`; appends a `recording_started` event to `history.jsonl`. Errors: `409` if a recording is already active.

### `POST /api/recording/stop`
Stop the active recording. No body.
Response: `{"stopped": bool}` — `true` if a recording was active and has been terminated.

**Side effects:** sends `Ctrl+Break` to the subprocess; dumps the full in-memory log buffer to `MDPIdata/session_<name>/receiver.log`; appends a `recording_stopped` event to `history.jsonl` with duration, exit code, and per-channel sample counts.

### `GET /api/recording/status`
Poll-friendly recording status.
Response:
```json
{
  "active": true,
  "session_name": "session_20260516_190103",
  "session_dir": "...",
  "started_at": "2026-05-16T19:01:03Z",
  "exit_code": null,
  "sample_counts": {"ecg": 12345, "0": 4521, "1": 4503, ...},
  "recent_log": ["line", "line", ...]
}
```
When `active=false` the next poll after the subprocess exits also dumps the receiver log + appends `recording_stopped` (so an unattended exit still ends up on disk).

---

## Batch + archive

### `POST /api/analyze_all`
Run `analyze_session` on every `session_*/` folder, aggregate per body site, produce a batch interpretation. Query params: `start_s`, `end_s`.

Response: see [`DATA_FORMATS.md`](DATA_FORMATS.md#batch_analysesbatch_tsjson) — full batch payload with `batch_id`, `created_at`, `n_sessions_analyzed`, `failed_sessions[]`, `sessions[]`, `per_site[]`, `fst_unavailable`, `interpretation`.

**Side effects:**
- Writes `MDPIdata/batch_analyses/batch_<YYYYMMDD_HHMMSS>.json`.
- Appends a `batch_analysis_included` event to every successfully-analyzed session's `history.jsonl` (1-indexed `session_position`).

### `GET /api/batch_analyses`
List saved batch runs. Newest first. Response: array of `{batch_id, created_at, n_sessions_analyzed, crop_window}`.

### `GET /api/batch_analyses/{batch_id}`
Load one saved batch run. Errors: `400` if the batch id doesn't match `^batch_\d{8}_\d{6}$`, `404` if the file is missing.

---

## Sleepiness summary

The "Total summary" page (top-bar `Σ` button) computes a Sleepiness Proxy Index across the cohort. See [`INTERPRETATION_GUIDE.md`](INTERPRETATION_GUIDE.md) for the SPI formula and caveats.

### `POST /api/sleepiness_summary`
Run the full SPI pipeline across every session. Query params:
- `weighting` (string, default `"ssqi_zsqi"`) — channel quality-weighting scheme
- `start_s`, `end_s` (float, optional) — crop window applied to every session

Response: see the saved `MDPIdata/sleepiness_runs/run_<ts>.json` schema (covered in `DATA_FORMATS.md` once the implementation lands). Includes `run_id`, `cohort_stats`, `per_session`, `per_site`, `per_fst_site`, `unusable_sessions`, `scatter_points`, `caveats`, `interpretation`.

**Side effects:** writes `MDPIdata/sleepiness_runs/run_<YYYYMMDD_HHMMSS>.json`; appends `sleepiness_analysis_included` events to every contributing session's `history.jsonl`.

### `GET /api/sleepiness_summary/latest`
Return the most recently saved SPI run, or `{"cached": false}` if none exist.

---

## Error codes

| Code | Meaning |
|------|---------|
| `400` | Invalid input — usually a session name or batch id that fails the regex check. |
| `404` | Session or batch id not found on disk. |
| `409` | Conflict — recording already active (on Start), or session is currently recording (on Delete). |
| `500` | Unhandled exception — should never happen; persistence helpers swallow I/O failures so the endpoint still returns the live result. |
