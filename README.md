# signal-processing-testing
initial testing of signal processing from MAX30102

## Webapp (acquisition + analysis dashboard)

One-command launcher for capturing a session, viewing the signals, and
running the SQI / agreement metrics against the simultaneous ECG:

```
pip install fastapi uvicorn pyserial
python app.py
```

Opens `http://127.0.0.1:8000` and your default browser. The dashboard
drives `PPG_ECG_Full_Unpacking.py` as a subprocess for capture, then
runs every PPG channel through `sqi/SSQI_algorithm.py`, `sqi/zcr_sqi.py`
and `sqi/ccc.py` (CCC + Bland-Altman vs ECG). Participant metadata
(`participant_id`, Fitzpatrick grade, channel→site map) is persisted as
`session_<ts>/participant.json`.

The original CLI scripts continue to work standalone — `python
PPG_ECG_Full_Unpacking.py`, `python vis.py`, `python fullvis.py`,
`python sqi/ccc.py`.

See [`webapp/README.md`](webapp/README.md) for a full dashboard user
guide: recording workflow, what each plot and table column means,
sidebar collapse, file layout, and troubleshooting.
