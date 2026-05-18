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
explaining the messy folder structure:

.claude - idk but this is how claude works.
.vscode - ignore this
_pychahe - ignore this too

old - all that old code i wrote for other ppg stuff, you can ignore this

depreciated - here I put the code and files that aren't particularly old, but are for older prototypes of our work. things like taking port input for just one ppg (now outdated since we use the multiplexer to collect many), taking port input for multiple ppg but no ecg, etc.


picofix - the nuke files for the raspberry pi if it tweaks out again. the instructions on how to use it are inside the picofix folder itself as instructions.md

invalidSessions - Any session that isn't going to be considered as a true recording in our paper is put into invalidSessions. parlty for additional testing purposes, partly because im scared to delete stuff.

MDPIdata - where we will store the actual data. within it there are folders, one for each subject. for ourselves, i have our names on it so we can easily test/debug, but dont forget that all volunteer subjects must remain anonymous and use an ID.
