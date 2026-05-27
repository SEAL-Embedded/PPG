# Testing

`tests/` is a pytest suite covering the dashboard backend (analysis, sessions, recorder, API) and the underlying SQI algorithms. As of the last run: **137 tests, 137 passing** in ~5 s.

## Install + run

```bash
pip install pytest httpx fastapi pandas numpy scipy pyserial
pip install pingouin   # optional — ICC tests skip gracefully without it
pytest tests/ -v
```

`pytest.ini` enables verbose tracebacks and silences the bundled deprecation warnings.

## What's covered

| File | Count | Scope |
|------|-------|-------|
| `tests/test_sqi.py` | 15 | `sqi/ccc.bandpass / lowpass / detect_r_peaks / detect_ppg_peaks / peaks_to_intervals / match_intervals / compute_ccc / ccc_label` shape + correctness; `sqi/SSQI_algorithm.Ssqi` sign on symmetric vs skewed distributions. |
| `tests/test_sessions.py` | 48 | `SESSION_PATTERN` / `BATCH_PATTERN` regex; `_with_default_sites` defaults + overrides; `parse_timestamp_from_name`; every persistence helper (`append_history`, `read_history`, `save_session_analysis`, `load_session_analysis`, `save_receiver_log`, `tail_receiver_log`, `save_batch_analysis`, `list_batch_analyses`, `load_batch_analysis`) round-trip with NaN coercion + path traversal rejection; `summarize_session` returns the new persistence-aware fields. |
| `tests/test_analysis.py` | 37 | `load_ecg` / `load_ppg` parsing tolerance; `ppg_bandpass` guard paths; `infer_fs`; `_downsample` extreme-preservation; `interpret_channel` / `interpret_session` / `interpret_batch` shape + grade buckets; `_safe_icc` minimum-pairs guard + pingouin-importorskip valid path; `_aggregate_per_site` mean/std/NaN handling; end-to-end `analyze_session` with crop windows + default site map + explicit metadata override. |
| `tests/test_recorder.py` | 10 | Idle status shape; `start()` creates session dir + appends `recording_started`; double `start()` rejected; `stop()` when idle returns False; receiver-log persistence on stop; `status()` detects unattended subprocess exit + dumps log. Subprocess.Popen is mocked — no real serial port required. |
| `tests/test_api.py` | 27 | Every endpoint via `fastapi.testclient.TestClient`: sessions list/single, analyze (with/without crop), analyze_all (batch fanout), history (newest-first + limit), cached analysis (`{cached: false}` path), receiver_log, batch list/load round-trip, metadata write + `metadata_edited` event, signals, recording start/stop/double-start; 400/404 paths for invalid/missing identifiers. |

If `pingouin` is missing, the `test_safe_icc_valid_input` test is skipped via `pytest.importorskip`; everything else still runs.

## Test data strategy

`tests/conftest.py` builds tiny **synthesised** ECG + PPG CSVs per test — a 30 s pure cosine with an R-peak comb on the ECG, smoothed copy on the PPG, sample rate ≈ 200 Hz to keep tests fast. The conftest exposes a `synthetic_session` fixture that monkeypatches `webapp.sessions.SESSIONS_SUBDIR` (and the underlying `sessions_root()` resolution) at a `tmp_path`, then writes one session folder there. End-to-end persistence tests run against this isolated tree so the real `MDPIdata/` is never touched.

## Adding a new test

1. Pick the right file (algorithm → `test_sqi.py`; persistence helper → `test_sessions.py`; HTTP route → `test_api.py`).
2. Reuse the `synthetic_session` (or `synthetic_session_with_analysis`) fixture from `tests/conftest.py` for anything that needs a session folder.
3. For new HTTP routes, use the `client` fixture — it returns a `TestClient(app)` with the synthesised root already in place.

```python
def test_my_new_endpoint(client, synthetic_session):
    r = client.get(f"/api/sessions/{synthetic_session.name}/my-endpoint")
    assert r.status_code == 200
    assert r.json()["expected_field"] == ...
```

4. When testing persistence-aware functions, assert against the **file on disk** as well as the return value — the persistence layer is "fire and forget" inside the endpoints (errors are swallowed) so the disk artefact is the ground truth.

## CI tip

This suite has no network or hardware dependencies. A minimal GitHub Actions step is:

```yaml
- uses: actions/setup-python@v5
  with: { python-version: "3.11" }
- run: pip install pytest httpx fastapi pandas numpy scipy pyserial pingouin
- run: pytest tests/ -q
```

(Pin `pyserial` even though no real port is touched — `import serial.tools.list_ports` happens at module import time in `webapp.api`.)
