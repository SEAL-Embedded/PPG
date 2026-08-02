# SEAL PPG — Multi-site PPG vs ECG signal-quality dashboard

SEAL Lab project for the MDPI Sensors manuscript on multi-site PPG (finger, forehead, earlobe, shoulder, wrist) validated against a 3-lead ECG ground truth and stratified by Fitzpatrick skin type. A Pi Pico drives up to eight MAX30102 PPG sensors behind a TCA9548A I²C mux and an AD8232 ECG front-end; this repo is the data-acquisition, persistence, and analysis side that turns those samples into the per-site agreement tables the paper is built around.

The whole loop — recording, metadata editing, single-session SQI/CCC/ICC/Bland-Altman, the every-session batch view, plain-English interpretations, persisted run history, receiver-log capture, and the past-batch archive — runs from one dashboard:

```bash
pip install fastapi uvicorn pyserial numpy pandas scipy pingouin pytest httpx
python app.py
```

Opens `http://127.0.0.1:8000/` in your browser. On first launch you'll see a 6-step onboarding modal walking you through the layout.

---

## Documentation map

| Doc | What it covers |
|-----|----------------|
| [`webapp/README.md`](webapp/README.md) | Operator's dashboard guide — every button, every panel, every modal |
| [`docs/DASHBOARD_GUIDE.md`](docs/DASHBOARD_GUIDE.md) | Goal-oriented walkthroughs ("I want to …") with screenshots |
| [`docs/INTERPRETATION_GUIDE.md`](docs/INTERPRETATION_GUIDE.md) | What SSQI / ZSQI / KSQI / CCC / ICC / Bland-Altman mean, plus the grade thresholds |
| [`docs/DATA_FORMATS.md`](docs/DATA_FORMATS.md) | On-disk schema for every CSV, JSON, JSONL, and log file |
| [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md) | Every HTTP endpoint with request / response / side-effects |
| [`docs/TESTING.md`](docs/TESTING.md) | How to run the pytest suite and add new tests |
| [`MDPIdata/README.md`](MDPIdata/README.md) | What lives in a session folder and how to consume it from Python |

---

## Repository map

```
PPG/
├── app.py                          single-command dashboard launcher
├── PPG_ECG_Full_Unpacking.py       serial receiver (spawned by dashboard or run standalone)
├── pipico_code/                    MicroPython firmware for the Pi Pico
├── picofix/                        nuke / reflash helpers if the Pico misbehaves
├── webapp/                         FastAPI dashboard
│   ├── api.py                        HTTP routes
│   ├── analysis.py                   per-session + batch analysis pipeline + interpretations
│   ├── sessions.py                   session discovery + persistence helpers
│   ├── recorder.py                   subprocess wrapper for the receiver
│   └── static/                       vanilla JS + Plotly single-page UI
├── sqi/                            signal-quality algorithms imported by the dashboard
│   ├── ccc.py                        CCC + Bland-Altman driver
│   ├── SSQI_algorithm.py             skewness SQI
│   ├── zcr_sqi.py                    zero-crossing rate SQI
│   ├── KSQI_algorithm.py             kurtosis SQI (Pearson / non-excess)
│   ├── ICC.py                        intraclass correlation (cross-subject form)
│   └── bland_altman.py               legacy raw-signal Bland-Altman (not wired — see note below)
├── signal_visualization/           legacy HRV + visualisation scripts (vis.py, fullvis.py, ppgvis.py)
├── tests/                          pytest suite (137 tests covering analysis, sessions, API, recorder, sqi)
├── MDPIdata/                       study data, one folder per session_<UTC_timestamp>/
│   ├── session_<ts>/                 ECG + PPG CSVs, participant.json, analysis.json, history.jsonl, receiver.log
│   └── batch_analyses/               saved snapshots of every "Analyze all sessions" run
├── invalidSessions/                recordings excluded from the manuscript
├── old/                            prior PPG + PVT study, kept for reference
└── depreciated/                    early single-PPG + no-ECG prototypes
```

Why `bland_altman.py` isn't wired into the dashboard: it operates on raw ECG/PPG samples directly, but the two have completely different waveform shapes (sharp QRS spike vs smooth pulse), so sample-by-sample Bland-Altman has no physiological meaning. The dashboard does the physiologically valid thing — computes Bland-Altman on the matched RR / PPI **intervals** instead, inside `compute_ccc` in `sqi/ccc.py`. Every Bland-Altman card in the UI plots those.

---

## Citation / license

Cite as: SEAL Lab, *Multi-site PPG vs ECG agreement stratified by Fitzpatrick skin type*, manuscript in preparation for MDPI Sensors, 2026.

License: research / academic use. Add a `LICENSE` file when releasing publicly.
