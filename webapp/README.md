# SEAL PPG — Dashboard guide

A single-page FastAPI dashboard that drives the full SEAL PPG capture
and analysis loop: start a recording, watch the ECG + PPG waveforms
stream live, then review SSQI / ZSQI / Lin's CCC / Bland-Altman against
the simultaneous ECG. Runs entirely on `127.0.0.1`.

The webapp does not re-implement any signal processing — it imports
functions from `sqi/ccc.py`, `sqi/SSQI_algorithm.py`, `sqi/zcr_sqi.py`,
and `sqi/bland_altman.py`. Numbers match a direct `python sqi/ccc.py`
run on the same channel + ECG pair.

---

## 1. Install

One-time, in the project root:

```
pip install fastapi uvicorn pyserial
```

`numpy`, `pandas`, `scipy`, and `matplotlib` are already used elsewhere
in the project and assumed present.

## 2. Run

```
python app.py
```

The script prints the URL, opens your default browser, and starts
serving on `http://127.0.0.1:8000`. `Ctrl+C` in the terminal stops it.

Optional environment knobs:

| Variable             | Default       | Purpose                       |
|----------------------|---------------|-------------------------------|
| `SEAL_WEBAPP_HOST`   | `127.0.0.1`   | Bind address                  |
| `SEAL_WEBAPP_PORT`   | `8000`        | Bind port                     |
| `SEAL_WEBAPP_NOOPEN` | unset         | If set, don't open browser    |

---

## 3. Recording a participant

The **Record** panel (left sidebar) is where new captures start.

1. **Port** — pick the COM port the Pi Pico is on. The `⟳` button
   rescans (handy if you replug the cable while the dashboard is
   open).
2. **Baud** — defaults to `115200`, matching the firmware.
3. **Participant ID** — free-form. Suggested format `P001`, `P002` …
4. **Fitzpatrick** — I–VI. Required for the FST stratified analysis;
   leave blank if not graded.
5. **Notes** — anything memorable about the session (ambient light,
   posture quirks, electrode replacement).
6. **Channel → Site** — assign each active mux lane to one of
   `finger / forehead / earlobe / shoulder / wrist / other`. The
   assignment persists in your browser between participants, so you
   only set it up once per rig.
7. Click **● Start**.

Once recording begins:

- The status pill at the top right pulses red `RECORDING`.
- A new `session_<YYYYMMDD_HHMMSS>/` folder is created in the project
  root. CSVs land there as the Pico streams.
- The right pane shows the **live signal traces** — ECG on top in red,
  each PPG channel below in its own card. Plots refresh roughly once
  per second from a 30-second tail window so they stay snappy even
  after a 10-minute capture.
- The **counter strip** under the form shows per-channel sample counts
  ticking up; ECG count is highlighted red.
- The receiver's stdout (mux discovery, throughput, errors) appears in
  the small log panel beneath the counters.

When you're done, click **■ Stop**. The receiver gets `Ctrl+Break` and
flushes its CSVs cleanly, then the dashboard:

1. Re-loads the full (untrimmed) signals.
2. Runs the SQI + CCC pipeline on every PPG channel against ECG.
3. Displays the SQI table and per-channel Bland-Altman cards.

No "Run analysis" click is required — the analysis auto-fires on
session stop and on every session click.

---

## 4. Reviewing past sessions

The **Sessions** sidebar lists every `session_*/` folder in the project
root, newest first. Each card shows:

- Participant ID (or *unassigned* if none was saved).
- FST grade if set.
- Recording timestamp.
- The channels actually present, with the `ecg` chip in red.

Click any card to load it. The right pane fills in immediately:

- **Header bar** — date/time, the meta-grid (`PID / FST / DURATION /
  CHANNELS / ECG / STARTED`), action buttons.
- **ECG reference** — the full ECG waveform with R-peak markers and
  red shading over any leads-off intervals. The summary line shows
  sample count, sample rate, R-peak count, mean HR, and leads-off
  sample count.
- **PPG channels** — one card per `ppg_data_ch{N}.csv`, with the site
  label (from the metadata) and detected systolic peak markers
  overlaid.
- **SQI & agreement vs ECG** — a row per channel:

  | Column     | Meaning                                                    |
  |------------|------------------------------------------------------------|
  | fs (Hz)    | Effective sample rate, inferred from timestamp deltas      |
  | SSQI       | Skewness of the raw PPG signal (Krishnan et al.)           |
  | ZSQI μ / σ | Mean and std of windowed zero-crossing rate (5-s windows)  |
  | Matched    | Number of RR–PPI pairs after nearest-neighbour matching    |
  | CCC        | Lin's concordance correlation coefficient                  |
  | Pearson    | Pearson r on the matched interval pairs                    |
  | Bias       | Mean (PPI − RR) in milliseconds                            |
  | LOA±       | Lower / upper 1.96·SD limits of agreement, in ms           |
  | RMSE, MAE  | Root-mean-square / mean-absolute error in ms               |

  The first row of the table is the **ECG reference** — fs, R-peak
  count, mean HR, leads-off samples. CCC values are color-coded:
  green ≥ 0.95 (substantial), amber 0.90–0.95 (moderate), red below.

- **Bland-Altman per channel** — one scatter card per matched-beat
  channel showing PPI − RR (y) against the mean (x), with the bias
  line in grey and the LOA± lines in red dashes.

---

## 5. Editing metadata after the fact

If you started a recording without filling in the participant form (or
need to correct a typo), fill the form on the left while the session
is selected and click **Save metadata** at the top of the detail
pane. The values are persisted into
`session_<...>/participant.json`, the session list updates, and the
PPG cards re-label themselves with the new site names.

`Run analysis` re-runs the SQI + CCC pipeline (useful if you change
the site map and want the table to relabel without a hard reload).
`Reload signals` reads the CSVs again — useful if you copied a session
from another machine while the dashboard was open.

---

## 6. Sidebar collapse

Each sidebar has a `«` button at its top-right corner. Click it to
collapse that sidebar to a thin 36 px rail showing a `»` button to
re-expand. Both sidebars can be collapsed independently; the main
column expands to fill the freed space, and the Plotly traces resize
in place. Your collapse choices persist across reloads
(`localStorage`).

This is the recommended way to maximise the plot area when reviewing a
recording.

---

## 7. Where things live

```
PPG/
├── app.py                          ← single-command entry point
├── PPG_ECG_Full_Unpacking.py       ← serial receiver, spawned as child
├── session_20260516_142931/        ← per-capture folders (gitignored)
│   ├── ecg_data.csv                  col0=ts_us, col1=sample, col2=leads_off
│   ├── ppg_data_ch0.csv              col0=ts_us, col1=sample
│   ├── ppg_data_ch1.csv
│   ├── ...
│   └── participant.json              webapp-owned: pid / fst / sites / notes
├── sqi/                            ← analysis functions imported by webapp
│   ├── ccc.py
│   ├── SSQI_algorithm.py
│   ├── zcr_sqi.py
│   └── bland_altman.py
└── webapp/
    ├── api.py                      FastAPI routes
    ├── sessions.py                 fs/metadata discovery
    ├── analysis.py                 per-channel SQI + CCC driver
    ├── recorder.py                 subprocess wrapper for receiver
    └── static/                     vanilla JS + Plotly single-page UI
        ├── index.html
        ├── app.js
        └── style.css
```

**Timestamp unit.** Session CSVs store col 0 in **microseconds** (Pico
`ticks_us`). `sqi/ccc.py` expects col 0 in **milliseconds** per its
docstring. The webapp converts µs → ms before calling any `ccc.py`
function. Running `python sqi/ccc.py` directly on a session CSV
without converting will misinterpret `fs` (≈ 0.16 Hz) and produce
garbage.

---

## 8. The original scripts still work

The dashboard is additive — none of the existing CLI tools were
replaced. You can still:

- `python PPG_ECG_Full_Unpacking.py` — record without the dashboard.
- `python vis.py` — quick-look the most recent PPG CSVs.
- `python fullvis.py` — combined ECG + PPG plot of one session.
- `python sqi/ccc.py` — CCC + Bland-Altman on a single channel pair,
  with prompts for paths.
- `python sqi/zcr_sqi.py --folder <dir>` — windowed ZSQI batch.

The dashboard just orchestrates them through one UI.

---

## 9. Troubleshooting

**Dashboard won't start, "address already in use".**
Another instance is still running. Either find and kill the old
`python.exe` via Task Manager, or set `SEAL_WEBAPP_PORT=8001` and use
that port.

**"No ports detected" in the Port dropdown.**
The Pico isn't enumerating. Plug it in *before* opening the
dashboard, or click the `⟳` button to rescan. If still empty, check
Device Manager — the Pico should show as a COM port.

**`Receiver exited unexpectedly (code N)`.**
The receiver subprocess died on its own. The alert shows the tail of
its stdout; common causes:
- Wrong COM port → check the port picker, then `⟳`.
- Pico unplugged mid-run → reconnect and start a new session.
- Firmware not running / wrong firmware → reflash via `picofix/` if
  needed.

**Plots are empty for a channel.**
That mux lane didn't enumerate. Check `Active PPG channels: [...]` in
the receiver log panel. Reseat the MAX30102, then start a new
session.

**`fs` shown is far from 750 Hz claimed in the manuscript.**
This is a known firmware/paper divergence — `fullpipico.py` configures
the MAX30102 at `set_sample_rate(3200)` with `fifo_average=8`, giving
≈ 400 Hz per channel, further reduced by the round-robin servicing
across active lanes. See `memory/project_code_paper_gaps.md`.

**CCC is near zero on a channel that "looks fine".**
Usually the PPG peak detector matched the wrong features (e.g.
respiration baseline rather than systolic peaks), producing PPI
values an order of magnitude off RR. Inspect the `n PPI intervals`
vs `n RR intervals` for that channel and look at the systolic peak
markers in its PPG card — if the markers are off, the underlying
peak threshold in `sqi/ccc.py:detect_ppg_peaks` needs tuning for that
sensor placement.
