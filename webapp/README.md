# SEAL PPG — Dashboard guide

The dashboard is the operating tool for the SEAL multi-site PPG vs ECG study: one FastAPI app that records a session from the Pi Pico, lists every recording in `MDPIdata/`, runs the SQI / CCC / ICC / Bland-Altman pipeline on each PPG channel, scores every channel and body site with a plain-English verdict, and archives every analysis on disk so the folder is fully self-describing.

This is the operator-facing manual. If you'd rather see goal-oriented walkthroughs first ("I want to record", "I want to load yesterday's batch run", …), read [`../docs/DASHBOARD_GUIDE.md`](../docs/DASHBOARD_GUIDE.md). For the algorithm reasoning behind every grade colour, see [`../docs/INTERPRETATION_GUIDE.md`](../docs/INTERPRETATION_GUIDE.md). For the on-disk schemas, [`../docs/DATA_FORMATS.md`](../docs/DATA_FORMATS.md). For HTTP details, [`../docs/API_REFERENCE.md`](../docs/API_REFERENCE.md).

## Table of contents

1. [Install + run](#1-install--run)
2. [Layout](#2-layout)
3. [Recording a participant](#3-recording-a-participant)
4. [Reviewing a session](#4-reviewing-a-session)
5. [Plain-English interpretation](#5-plain-english-interpretation)
6. [Batch analysis across MDPIdata](#6-batch-analysis-across-mdpidata)
7. [Run history + receiver log](#7-run-history--receiver-log)
8. [Past batch archive](#8-past-batch-archive)
9. [Editing metadata](#9-editing-metadata)
10. [Sidebar collapse + keyboard](#10-sidebar-collapse--keyboard)
11. [Where files live](#11-where-files-live)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Install + run

```bash
pip install fastapi uvicorn pyserial numpy pandas scipy pingouin
python app.py
```

`pingouin` is optional — without it the ICC column shows `—` everywhere but the rest of the dashboard still works.

Optional environment knobs:

| Variable             | Default       | Purpose                       |
|----------------------|---------------|-------------------------------|
| `SEAL_WEBAPP_HOST`   | `127.0.0.1`   | Bind address                  |
| `SEAL_WEBAPP_PORT`   | `8000`        | Bind port                     |
| `SEAL_WEBAPP_NOOPEN` | unset         | If set, don't auto-open browser |

A 6-step onboarding modal pops up the first time you load the page. Dismiss it ("Don't show again" toggles the `seal_ppg_onboarded` localStorage flag) or reopen any time via the `?` button in the top-right of the header.

## 2. Layout

Three columns:

| Column      | Purpose                                                              |
|-------------|----------------------------------------------------------------------|
| **Record**  | Port picker, baud, participant form (PID / FST / notes / channel→site), Start / Stop, live counters, receiver log tail. |
| **Sessions**| Every `MDPIdata/session_*/` folder, filterable. Each card shows PID, FST pill, timestamp, channel chips, and persistence badges (`analysed Xs ago` / `×N analyses` / `log only`). Below the filter input: the `▦ Analyze all sessions` button and a `Past batch runs / Browse archive` row. |
| **Detail**  | Either per-session view (signals + SQI + interpretation + history) or batch view (per-site aggregate + per-session × per-channel + batch interpretation). |

Each sidebar has a `«` toggle that collapses it to a 36 px rail (`»` to expand). Collapse choices persist in `localStorage`.

## 3. Recording a participant

1. **Port** — pick the COM port the Pi Pico is on. `⟳` rescans if you replug mid-session.
2. **Baud** — default `115200`, matches the firmware.
3. **Participant ID** — free-form, suggested `P001`, `P002` …
4. **Fitzpatrick** — I–VI. Required to unlock FST stratification in the batch view; blank if not graded.
5. **Notes** — anything memorable (ambient light, posture quirks, electrode replacement).
6. **Channel → Site** — assign each active mux lane to `finger / forehead / earlobe / shoulder / wrist / other`. Persists in your browser between participants (localStorage `seal_ppg_sites`), so you only set it once per rig.
7. **● Start.**

While recording:

- Status pill pulses red `RECORDING`.
- A new `MDPIdata/session_<YYYYMMDD_HHMMSS>/` folder is created; CSVs land there as the Pico streams.
- Live signal traces refresh ~1 Hz from a 30-second tail window so they stay snappy after a 10-minute capture.
- Counters under the form tick per-channel sample counts; ECG count is red.
- Receiver stdout (mux discovery, throughput, errors) appears in the small log panel.
- A `recording_started` event is appended to `history.jsonl` in the session folder.

**■ Stop** sends `Ctrl+Break` to the receiver so its `finally f.close()` block runs, then the dashboard:

1. Dumps the full receiver stdout/stderr buffer to `MDPIdata/session_<ts>/receiver.log`.
2. Appends a `recording_stopped` event to `history.jsonl` (duration, sample counts, exit code).
3. Reloads the full untrimmed signals.
4. Automatically runs the SQI + CCC + ICC pipeline and shows the SQI table, Bland-Altman cards, and the plain-English interpretation block.

If the receiver dies on its own (lost USB, crash), the next status poll detects the unattended exit, dumps the log, and appends `recording_stopped` with the exit code — so the session still ends up fully described on disk even when nobody clicked Stop.

## 4. Reviewing a session

Click any session card. The detail column fills with:

- **Header bar** — date/time, meta-grid (`PID / FST / DURATION / CHANNELS / ECG / STARTED`), action buttons.
- **ECG reference** — full waveform with R-peak markers, red shading over any leads-off intervals. Summary line: sample count, sample rate, R-peak count, mean HR, leads-off samples.
- **PPG channels** — one card per `ppg_data_ch{N}.csv` with the site label, sample count, fs, detected systolic peak markers (Elgendi 2013 TERMA detector), and a `bandpass` checkbox toggling the 0.5–8 Hz cardiac-band Butterworth overlay that detection runs on.
- **Plain-English interpretation** (described in detail in [§5](#5-plain-english-interpretation)).
- **SQI & agreement vs ECG** — one row per channel:

  | Column     | Meaning                                                    |
  |------------|------------------------------------------------------------|
  | fs (Hz)    | Effective sample rate, inferred from timestamp deltas      |
  | SSQI       | Skewness of the raw PPG signal (Krishnan et al. 2010)      |
  | ZSQI μ / σ | Mean and std of windowed zero-crossing rate (5-s windows)  |
  | Matched    | Number of RR–PPI pairs after nearest-neighbour matching    |
  | CCC        | Lin's concordance correlation coefficient                  |
  | ICC        | Intraclass correlation, pingouin `ICC(A,1)` — two-way mixed, absolute agreement |
  | Pearson    | Pearson r on the matched interval pairs                    |
  | Bias       | Mean (PPI − RR) in milliseconds                            |
  | LOA±       | Lower / upper 1.96·SD limits of agreement, in ms           |
  | RMSE, MAE  | Root-mean-square / mean-absolute error in ms               |

  CCC / ICC cells are colour-coded: green ≥ 0.95 (substantial), amber 0.90–0.95 (moderate), red below.

- **Bland-Altman per channel** — one scatter card per matched-beat channel showing PPI − RR (y) against mean (x), with the bias line in grey and the LOA± lines in red dashes. Plotted on the matched **intervals**, not the raw waveform samples (those have different shapes — see [§9 of the top-level README](../README.md)).
- **Run history** — collapsible timeline; see [§7](#7-run-history--receiver-log).

The **Window (s)** controls in the header crop every signal to a [start, end] window (seconds since the session's ECG t0) before any peak detection — so SSQI, ZSQI, R-peaks, RR/PPI, and CCC are all computed on the selected window only. **Full** restores the whole recording.

`Run analysis` re-runs the pipeline (useful after editing the site map). `Reload signals` refetches the CSVs if you copied a session in from another machine. Both write `analysis.json` and append an `analysis_run` history event.

## 5. Plain-English interpretation

After every analysis run, the dashboard renders a block above the SQI table that translates the numbers into sentences. It comes from `webapp/analysis.py:interpret_session` and is structured as:

- **Headline** — one-line cohort summary, e.g. "All 5 PPG channels reached moderate-or-better agreement with ECG." or "No PPG channel reached moderate (CCC>0.90) agreement with ECG in this window."
- **ECG line** — "ECG reference: 451 R-peaks, mean HR 81 bpm over 334 s @ 323 Hz."
- **Verdict cards** — one per channel, colour-coded by grade:

  | Grade | Colour | When |
  |-------|--------|------|
  | **good** | green | CCC > 0.95 and SSQI good/very-good |
  | **ok**   | teal  | CCC > 0.90 (moderate agreement, usable with caveats) |
  | **warn** | amber | CCC 0.50–0.90 (poor agreement, inspect) or few matched beats |
  | **bad**  | red   | CCC ≤ 0.50, zero matched beats, or bias orders-of-magnitude past PTT |

  Each card carries the verdict headline, a bullet per metric in plain English, and a "next step" advice line.

- **Notes** — best/weakest channel callouts, fs warnings, anything that doesn't fit a verdict card.

Read [`../docs/INTERPRETATION_GUIDE.md`](../docs/INTERPRETATION_GUIDE.md) for the exact thresholds (Lin 1989 for CCC, Cicchetti 1994 for ICC, Krishnan 2010-style SSQI bands, ZSQI heuristics for sensor contact).

## 6. Batch analysis across MDPIdata

Click **▦ Analyze all sessions** in the sessions sidebar. The dashboard runs `analyze_session` against every `MDPIdata/session_*/` folder, aggregates the per-channel results by body site, computes the cross-channel mean ± std, and produces a batch-level interpretation. The result replaces the detail column with a three-block batch view:

1. **Batch interpretation** — headline ("Across 4 sessions and 5 body sites: best site is finger (mean CCC 0.20, only poor agreement)."), per-site verdicts (grade-coloured rows), and notes (performance gap, failed sessions, FST availability).
2. **Per-site aggregate** — one row per body site (`finger`, `earlobe`, `forehead`, `shoulder`, `wrist`, …) with mean ± std for SSQI, ZSQI, CCC, ICC, Pearson, bias, LOA span, RMSE, MAE. Click any column header to sort (▴ ▾ tri-state). The first column has a stronger border and rows alternate-tint for scanning.
3. **Per-session × per-channel** — every channel of every session, grouped under a sticky session-header row showing the full `session_YYYYMMDD_HHMMSS` id (clickable — drills into the per-session view), PID, FST tag, mean HR, duration, and channel count.

Top of the batch view:

- **● LIVE RUN** badge for a fresh run, **ARCHIVE · BATCH_<ts>** for an archive load.
- **Re-run** — re-runs against the same crop window.
- **⤓ Export CSV** — downloads the per-site aggregate as a CSV (`per_site_<batch_id>.csv`).
- **Close batch view** — returns to the per-session detail column.

The active crop window from the per-session header is reused for every session in the batch, so the cropped numbers stay comparable to whatever you were last looking at.

**Fitzpatrick stratification** is shown as `available` / `unavailable` in the meta-grid. When `unavailable`, none of the included sessions has an FST grade — save metadata on at least one session to start unlocking the FST I-III vs IV-VI subgroup tables the manuscript needs.

The batch is also persisted to `MDPIdata/batch_analyses/batch_<ts>.json`, and a `batch_analysis_included` event is appended to every included session's `history.jsonl` (with the `batch_id` and the session's 1-indexed position in the batch).

## 7. Run history + receiver log

Each session's detail view has a **Run history** panel near the bottom — collapsible (closed by default; click "Show history (N)" to expand). It renders the events in `MDPIdata/session_<ts>/history.jsonl`, newest first, as a vertical timeline. Event types:

| Event                       | Pill colour | What it records |
|-----------------------------|-------------|-----------------|
| `recording_started`         | blue        | Port, baud, session name |
| `recording_stopped`         | blue        | Duration, sample counts, subprocess exit code |
| `metadata_edited`           | amber       | `before` and `after` full participant dicts |
| `analysis_run`              | green       | Crop window, n channels, compact `{ch: {ccc, icc, matched}}` summary |
| `batch_analysis_included`   | muted       | `batch_id`, session position, total |

The **Reload history** button refetches in case the session was edited from another tab.

The **View receiver log** button opens a modal showing the last 500 lines of `MDPIdata/session_<ts>/receiver.log` (mono, scrollable, with **Refresh**). Disabled when `has_receiver_log` is false (older recording or capture failed).

## 8. Past batch archive

Every batch run is saved to `MDPIdata/batch_analyses/batch_<YYYYMMDD_HHMMSS>.json`. The Sessions sidebar shows the count (`Past batch runs N`) and a **Browse archive** button — opens a modal listing past runs (timestamp, n sessions analyzed, crop window). Select one to reload it into the batch view without re-running the pipeline. The header shows `ARCHIVE · BATCH_<ts>` so you always know whether you're looking at fresh or saved data.

This is the path for "I want to see how the cohort scored after last Tuesday's three new recordings landed" without recomputing.

## 9. Editing metadata

Fill the participant form on the left (PID / FST / notes / channel→site) while a session is selected and click **Save metadata**. The dashboard:

1. Reads the existing `participant.json` so we have a `before` snapshot.
2. Writes the new one.
3. Appends a `metadata_edited` event to `history.jsonl` with both `before` and `after` blocks — so you have a full audit trail of every edit (no work is lost when you correct a typo).
4. Re-renders the session list and the meta-grid; the PPG cards relabel themselves with the new site names.

## 10. Sidebar collapse + keyboard

- Each sidebar has a `«` button at its top-right corner. Click to collapse to a 36 px rail (`»` to expand). Both sidebars are independent; the main column fills the freed space; Plotly traces resize in place. Persists across reloads (`localStorage`).
- All modals close on **Esc** and on backdrop click.
- The **Window (s)** inputs accept **Enter** as a shortcut for the Apply button.
- The `?` button in the header reopens the onboarding modal.

## 11. Where files live

```
PPG/
├── app.py                                  single-command entry point
├── PPG_ECG_Full_Unpacking.py               serial receiver, spawned as child
├── MDPIdata/
│   ├── session_<YYYYMMDD_HHMMSS>/
│   │   ├── ecg_data.csv                      col0=ts_us, col1=sample, col2=leads_off
│   │   ├── ppg_data_ch{0..N}.csv             col0=ts_us, col1=sample
│   │   ├── participant.json                  webapp-owned: pid / fst / sites / notes
│   │   ├── analysis.json                     latest analysis result (overwritten on each Run analysis)
│   │   ├── history.jsonl                     append-only event log
│   │   └── receiver.log                      captured receiver stdout/stderr
│   └── batch_analyses/
│       └── batch_<YYYYMMDD_HHMMSS>.json      saved batch snapshots
├── sqi/                                    analysis functions imported by the dashboard
└── webapp/
    ├── api.py                              FastAPI routes
    ├── analysis.py                         per-session + batch analysis + interpretations
    ├── sessions.py                         session discovery + persistence helpers
    ├── recorder.py                         subprocess wrapper for receiver
    └── static/                             vanilla JS + Plotly UI (no build step)
        ├── index.html
        ├── app.js
        └── style.css
```

**Timestamp unit.** Session CSVs store col 0 in **microseconds** (Pico `ticks_us`); `sqi/ccc.py` expects col 0 in **milliseconds** per its docstring. `webapp/analysis.py:load_ppg / load_ecg` divides by 1000 before calling any `ccc.py` function. Running `python sqi/ccc.py` directly on a session CSV without converting will misinterpret `fs` (≈ 0.16 Hz) and produce garbage.

For the full per-file schema with example payloads, see [`../docs/DATA_FORMATS.md`](../docs/DATA_FORMATS.md).

## 12. Troubleshooting

**Dashboard won't start, "address already in use".**
Another instance is still running. Find and kill the old `python.exe`, or set `SEAL_WEBAPP_PORT=8001`.

**"No ports detected" in the Port dropdown.**
The Pico isn't enumerating. Plug it in before opening the dashboard, or click `⟳`. If still empty, check Device Manager — the Pico should show as a COM port.

**`Receiver exited unexpectedly (code N)`.**
The receiver subprocess died. The alert shows the tail of its stdout; common causes: wrong COM port, Pico unplugged mid-run, firmware not running. The receiver log is still saved to `MDPIdata/session_<ts>/receiver.log` so you can inspect afterwards.

**Plots are empty for a channel.**
That mux lane didn't enumerate. Check `Active PPG channels: [...]` in the receiver log panel. Reseat the MAX30102, then start a new session.

**ICC column shows `—` everywhere.**
`pingouin` is not installed. `pip install pingouin` and reload.

**View receiver log button is greyed out.**
That session was recorded before this dashboard version (or the capture itself failed before any output was buffered). The button is disabled with a tooltip explaining.

**`fs` shown is far from 750 Hz claimed in the manuscript.**
Known firmware/paper divergence — `fullpipico.py` configures the MAX30102 at `set_sample_rate(3200)` with `fifo_average=8`, giving ≈ 400 Hz per channel, further reduced by the round-robin servicing across active lanes. See `memory/project_code_paper_gaps.md` and [`../docs/INTERPRETATION_GUIDE.md`](../docs/INTERPRETATION_GUIDE.md).

**CCC is near zero on a channel that "looks fine".**
Usually the PPG peak detector matched the wrong features (e.g. respiration baseline rather than systolic peaks), producing PPI values an order of magnitude off RR. Inspect the `n PPI intervals` vs `n RR intervals` for that channel and look at the systolic peak markers in its PPG card. The interpretation block flags this with a "peak detector probably matched respiration or a harmonic" advice line.

**The onboarding modal won't go away.**
Tick "Don't show again" before clicking Done — that sets `localStorage["seal_ppg_onboarded"]`. The header `?` button can always reopen it.
