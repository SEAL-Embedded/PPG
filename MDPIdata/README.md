# MDPIdata — recording archive

This is where every SEAL PPG session lands. One folder per recording, named `session_<YYYYMMDD_HHMMSS>` (UTC, derived from the Pico's start time). Each folder is **fully self-describing** — once a session is here, you have everything: the raw signals, the participant metadata, the latest analysis, the full edit/run history, and the receiver-subprocess log.

```
MDPIdata/
├── session_<UTC_timestamp>/
│   ├── ecg_data.csv             ECG samples
│   ├── ppg_data_ch{0..N}.csv    one CSV per active mux lane
│   ├── participant.json         dashboard-owned metadata
│   ├── analysis.json            latest analysis result (overwritten each run)
│   ├── history.jsonl            append-only event log
│   └── receiver.log             captured receiver stdout/stderr
└── batch_analyses/
    └── batch_<UTC_timestamp>.json   one snapshot per "Analyze all sessions" run
```

For the full authoritative schema (every column, every JSON field, every event payload), see [`../docs/DATA_FORMATS.md`](../docs/DATA_FORMATS.md). What follows is the *quick* reference + Python recipes.

---

## Per-file reference

| File | Format | Written by | Read by |
|------|--------|-----------|---------|
| `ecg_data.csv` | Headerless CSV: `timestamp_us, sample, leads_off` | `PPG_ECG_Full_Unpacking.py` (subprocess) | dashboard, `sqi/ccc.py` |
| `ppg_data_ch{N}.csv` | Headerless CSV: `timestamp_us, sample` | same | same |
| `participant.json` | JSON object | dashboard (`POST /api/sessions/{name}/metadata`) | dashboard |
| `analysis.json` | JSON object | dashboard (`POST /api/sessions/{name}/analyze`) | dashboard |
| `history.jsonl` | JSONL (one event per line) | dashboard | dashboard |
| `receiver.log` | Plain text | dashboard (on Stop or detected exit) | operator + dashboard log viewer |
| `batch_analyses/batch_*.json` | JSON object | dashboard (`POST /api/analyze_all`) | dashboard archive view |

---

## Timestamp unit (important)

Both CSVs store column 0 in **microseconds** (Pico `ticks_us()`). `sqi/ccc.py` expects col 0 in **milliseconds** per its docstring, so the dashboard divides by 1000 before calling any `ccc.py` function. If you load a session CSV in your own script, do the same:

```python
import pandas as pd
df = pd.read_csv("session_20260516_174059/ecg_data.csv",
                 header=None, names=["ts_us", "sample", "leads_off"])
df["ts_ms"] = df["ts_us"] / 1000.0          # ms, matches sqi/ccc.py contract
df["ts_s"]  = df["ts_us"] / 1_000_000.0     # seconds since session origin (after subtracting t0)
```

The PPG and ECG share the same Pico clock, so subtracting `t0 = ecg_data.iloc[0, 0]` from both lines up R-peaks and PPG pulses across files for the same session.

---

## Worked example

After recording one session and clicking **Run analysis**, then later including it in an **Analyze all sessions** batch run:

```
session_20260516_174059/
├── ecg_data.csv            (~2 MB / ~100k rows over 5 min)
├── ppg_data_ch0.csv        (~0.7 MB / ~40k rows)
├── ppg_data_ch1.csv
├── ppg_data_ch2.csv
├── ppg_data_ch3.csv
├── ppg_data_ch4.csv
├── participant.json        {"participant_id":"P001","fitzpatrick":3,"notes":"","channel_sites":{"0":"finger",...}}
├── receiver.log            full stdout from PPG_ECG_Full_Unpacking.py (mux init, throughput, exit notice)
├── analysis.json           the full analyze_session() payload with interpretations
└── history.jsonl
    ├─ {"ts":"2026-05-16T17:40:59.123Z","event":"recording_started","data":{...}}
    ├─ {"ts":"2026-05-16T17:46:01.456Z","event":"recording_stopped","data":{...}}
    ├─ {"ts":"2026-05-16T17:50:00.001Z","event":"metadata_edited","data":{"before":{...},"after":{...}}}
    ├─ {"ts":"2026-05-25T08:29:57.357Z","event":"analysis_run","data":{...}}
    └─ {"ts":"2026-05-25T08:30:15.012Z","event":"batch_analysis_included","data":{"batch_id":"batch_20260525_083015","session_position":4,"total":4}}
```

---

## Consuming the persistence files from Python

```python
import json, pathlib

session = pathlib.Path("MDPIdata/session_20260516_174059")

# Participant metadata
meta = json.loads((session / "participant.json").read_text())
fst = meta.get("fitzpatrick")        # 1-6 or None
sites = meta["channel_sites"]        # {"0": "finger", "1": "earlobe", ...}

# Latest analysis (full result with interpretations)
analysis = json.loads((session / "analysis.json").read_text())
when = analysis["analyzed_at"]                    # ISO timestamp with trailing Z
crop = analysis["crop_window"]                    # {"start_s": ..., "end_s": ...}
headline = analysis["interpretation"]["headline"]  # plain English
for row in analysis["results"]:
    ch = row["channel"]
    site = row["site"]
    ccc = (row.get("stats") or {}).get("ccc")
    print(f"ch{ch} ({site}): CCC={ccc}, verdict={row['interpretation']['verdict']}")

# History — newest first
with (session / "history.jsonl").open() as f:
    events = [json.loads(line) for line in f if line.strip()]
events.sort(key=lambda e: e["ts"], reverse=True)
analysed_count = sum(1 for e in events if e["event"] == "analysis_run")

# Batch archive
batches = sorted(pathlib.Path("MDPIdata/batch_analyses").glob("batch_*.json"))
latest = json.loads(batches[-1].read_text())
for site_row in latest["per_site"]:
    print(site_row["site"], site_row["ccc"]["mean"], site_row["ccc"]["std"])
```

---

## Anonymisation reminder

Volunteer subjects must remain anonymous. Use participant IDs (`P001`, `P002`, …) — never put names or other PII in `participant_id`, `notes`, or session folder names. The dashboard's `Save metadata` form accepts free-form text, so check `participant.json` before sharing the folder.
