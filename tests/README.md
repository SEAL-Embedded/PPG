# SEAL PPG test bench

`pytest` suite covering the FastAPI webapp (`webapp/`) and the signal-
quality / agreement primitives in `sqi/`.

## Install

The suite assumes Python 3.10+ and the same dependency stack the app
itself runs against. From the repo root:

```
pip install pytest httpx fastapi pandas numpy scipy pyserial
# optional — only used by the ICC test, which skips when missing
pip install pingouin
```

`httpx` is what `fastapi.testclient.TestClient` uses under the hood.

## Run

```
pytest tests/ -v
```

A specific module:

```
pytest tests/test_sessions.py -v
```

Or one test:

```
pytest tests/test_api.py::TestAnalyzeEndpoint::test_analyze_writes_analysis_json_and_history -v
```

## What's covered

| File                     | Coverage                                                                                           |
| ------------------------ | -------------------------------------------------------------------------------------------------- |
| `tests/test_sqi.py`      | filters, peak detectors, intervals, `compute_ccc` shape + error paths, SSQI sign behaviour          |
| `tests/test_sessions.py` | regex patterns, default-site merge, persistence helpers, `summarize_session` new fields, batch archive |
| `tests/test_analysis.py` | loaders, `ppg_bandpass` guard rails, `infer_fs`, peak-preserving downsample, interpretation dicts, `_aggregate_per_site` math, end-to-end `analyze_session` |
| `tests/test_recorder.py` | lifecycle with mocked `subprocess.Popen`, history events, log persistence on stop and on unattended exit |
| `tests/test_api.py`      | every endpoint via `TestClient` — list / detail / analyze / analyze_all / history / cached analysis / receiver log / batch round-trip / metadata / signals / recording (mocked subprocess) |

## Isolation strategy

`tests/conftest.py` monkeypatches `webapp.sessions.sessions_root` to a
`tmp_path` for every test that touches disk. The synthetic ECG + PPG
fixtures live in the same file — a 30 s recording at 200 Hz with
realistic RR jitter (~30 R-peaks, ~28 matched beats, finite CCC /
Pearson / bias). Nothing in this suite reads from `MDPIdata/` or
touches a real serial port.
