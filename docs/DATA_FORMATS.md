# Data formats — authoritative on-disk schema

Every file the dashboard reads or writes inside `MDPIdata/`, with example payloads. Pair with [`API_REFERENCE.md`](API_REFERENCE.md) (the HTTP-level view) and [`INTERPRETATION_GUIDE.md`](INTERPRETATION_GUIDE.md) (what the metrics mean).

## Table of contents

1. [Timestamp conventions](#timestamp-conventions)
2. [`ecg_data.csv`](#ecg_datacsv)
3. [`ppg_data_ch{N}.csv`](#ppg_data_chncsv)
4. [`participant.json`](#participantjson)
5. [`analysis.json`](#analysisjson)
6. [`history.jsonl`](#historyjsonl)
7. [`receiver.log`](#receiverlog)
8. [`batch_analyses/batch_<ts>.json`](#batch_analysesbatch_tsjson)

---

## Timestamp conventions

| Where | Unit | Origin | Format |
|-------|------|--------|--------|
| CSV column 0 | microseconds (`int`) | Pico boot (`ticks_us()`) | bare integer |
| `sqi/ccc.py` function inputs | milliseconds (`float`) | same | bare float; dashboard divides CSV col 0 by 1000 before calling |
| `analysis.json["analyzed_at"]`, `history.jsonl` events, `batch_*.json["created_at"]` | ISO 8601 (UTC) | wall clock | string like `"2026-05-25T08:29:57.354731Z"` |
| Frontend Plotly traces, batch CSV export | seconds since session t0 | first ECG sample | float |

---

## `ecg_data.csv`

Headerless CSV, three columns. Written by `PPG_ECG_Full_Unpacking.py`.

| Col | Name           | Type   | Notes |
|-----|----------------|--------|-------|
| 0   | `timestamp_us` | int    | Pico `ticks_us()`. Sample period ≈ 3.1 ms (~323 Hz inferred). |
| 1   | `sample`       | int    | AD8232 ADC, 0–65535. `0` when `leads_off == 1`. |
| 2   | `leads_off`    | int    | 0 = leads attached, 1 = either leg lead disconnected. |

Example (first 5 rows):

```csv
31895407,33192,0
31898620,31703,0
31901758,31239,0
31904977,32087,0
31908208,32647,0
```

---

## `ppg_data_ch{N}.csv`

Headerless CSV, two columns. Written by `PPG_ECG_Full_Unpacking.py`, one file per mux lane `N ∈ {0, 1, ..., 7}` that enumerated at recording start.

| Col | Name           | Type | Notes |
|-----|----------------|------|-------|
| 0   | `timestamp_us` | int  | Same Pico clock as `ecg_data.csv`. |
| 1   | `sample`       | int  | MAX30102 IR / Red ADC value (0–262143). |

Example:

```csv
31898316,30453
31906386,30444
31914167,30448
31922018,30450
31929595,30453
```

---

## `participant.json`

JSON object. Owned and written by the dashboard (`POST /api/sessions/{name}/metadata`); the receiver never touches it. Always returned by the dashboard's session-summary endpoints even if the file is missing — defaults from `webapp/sessions._with_default_sites` are merged in (the fixed mux-lane → body-site map identical across every session on this rig).

```json
{
  "participant_id": "P001",
  "fitzpatrick": 3,
  "notes": "ambient light dim; posture seated",
  "channel_sites": {
    "0": "finger",
    "1": "earlobe",
    "2": "shoulder",
    "3": "forehead",
    "4": "wrist"
  }
}
```

| Field | Type | Notes |
|-------|------|-------|
| `participant_id` | string | Free-form. Use anonymous IDs (`P001`, `P002`, …). |
| `fitzpatrick` | int 1–6 or null | Required for the batch view's FST × site stratification. |
| `notes` | string | Free-form. |
| `channel_sites` | `{str_channel_index: site_label}` | Defaults filled from `DEFAULT_CHANNEL_SITES` in `webapp/sessions.py`; any explicit non-empty value wins. Sites: `finger / forehead / earlobe / shoulder / wrist / other`. |

---

## `analysis.json`

JSON object. Written by `POST /api/sessions/{name}/analyze` via `webapp/sessions.save_session_analysis`. **Overwritten on each Run analysis**, but every run is summarised in `history.jsonl` so the full timeline survives.

Top-level shape:

```json
{
  "analyzed_at": "2026-05-25T08:29:57.354731Z",
  "crop_window": { "start_s": null, "end_s": null },
  "participant": { ... },
  "ecg": {
    "fs_hz": 323.0,
    "n_samples": 105956,
    "duration_s": 333.5,
    "n_peaks": 451,
    "mean_hr_bpm": 81.1,
    "leads_off_samples": 0,
    "peak_times_s": [0.41, 1.13, ...]
  },
  "results": [
    {
      "channel": 0,
      "site": "finger",
      "ppg_fs_hz": 130.0,
      "ecg_fs_hz": 323.0,
      "n_ppg_samples": 42138,
      "n_ecg_samples": 105956,
      "ssqi": 0.855,
      "zsqi_mean": 0.030,
      "zsqi_std": 0.008,
      "zsqi_max": 0.052,
      "ksqi": 2.141,
      "n_rr_intervals": 450,
      "n_ppi_intervals": 449,
      "n_matched_beats": 448,
      "ppg_peak_times_s": [0.55, 1.27, ...],
      "stats": {
        "ccc": 0.7928, "ccc_label": "Poor",
        "pearson_r": 0.8112,
        "mean_ppg_ppi_ms": 740.1, "mean_ecg_rr_ms": 740.1,
        "bias_ms": -0.0, "loa_upper_ms": 55.2, "loa_lower_ms": -55.4,
        "rmse_ms": 28.0, "mae_ms": 22.9,
        "icc": 0.7931, "icc_ci_low": 0.756, "icc_ci_high": 0.825,
        "matched_rr_ms": [720, 738, ...],
        "matched_ppi_ms": [722, 740, ...]
      },
      "error": null,
      "interpretation": {
        "verdict": "ch0 (finger) — poor agreement, inspect before using",
        "grade": "warn",
        "lines": [
          "SSQI +0.85 — positive skew, pulse shape is recognisable.",
          "ZSQI 0.030 (σ 0.008) — low, stable zero-crossing rate, sensor contact looks consistent.",
          "CCC 0.793, ICC 0.793 — poor agreement (Lin 1989: <0.90).",
          "Bland-Altman bias -0.0 ms — tracks ECG with no meaningful offset; ±LOA span 110 ms, RMSE 28.0 ms."
        ],
        "advice": "Look at the PPG trace for ectopic beats, missed peaks, or motion bursts."
      }
    }
  ],
  "interpretation": {
    "headline": "No PPG channel reached moderate (CCC>0.90) agreement with ECG in this window.",
    "ecg_text": "ECG reference: 451 R-peaks, mean HR 81 bpm over 334 s @ 323 Hz.",
    "channel_summaries": [{ "channel": 0, "site": "finger", "verdict": "...", "grade": "warn", "lines": [...], "advice": "..." }, ...],
    "best_channel": 0,
    "worst_channel": 3,
    "notes": [
      "Best channel: ch0 (CCC 0.793).",
      "Weakest channel: ch3 (CCC 0.000).",
      "Channels with zero matched beats are highlighted — investigate sensor contact or peak-detection threshold."
    ]
  }
}
```

Non-finite floats (NaN, ±Inf) are coerced to JSON `null` before writing so the file is parseable by every strict-JSON reader.

---

## `history.jsonl`

Append-only event log, one JSON object per line, UTF-8. Written by `webapp/sessions.append_history`. Newest line is the most recent event; readers should sort by `ts` (the file is appended, not rewritten, so on-disk order matches chronological order).

Five event types. Each has the same wrapper:

```json
{"ts": "2026-05-25T08:29:57.357138Z", "event": "<type>", "data": { ... }}
```

### `recording_started`
```json
{"ts":"2026-05-16T17:40:59.001Z","event":"recording_started",
 "data":{"port":"COM3","baud":115200,"session_name":"session_20260516_174059"}}
```

### `recording_stopped`
```json
{"ts":"2026-05-16T17:46:01.456Z","event":"recording_stopped",
 "data":{"session_name":"session_20260516_174059",
         "duration_s":302.4,
         "exit_code":0,
         "sample_counts":{"ecg":97524,"0":38109,"1":38201,"2":38057,"3":38114,"4":38182}}}
```

### `metadata_edited`
```json
{"ts":"2026-05-16T17:50:00.001Z","event":"metadata_edited",
 "data":{
   "before":{"participant_id":"","fitzpatrick":null,"notes":"","channel_sites":{"0":"finger",...}},
   "after" :{"participant_id":"P001","fitzpatrick":3,"notes":"first acquisition","channel_sites":{"0":"finger",...}}
 }}
```

### `analysis_run`
```json
{"ts":"2026-05-25T08:29:57.357Z","event":"analysis_run",
 "data":{
   "crop_window":{"start_s":null,"end_s":null},
   "n_channels":5,
   "summary":{
     "0":{"ccc":0.793,"icc":0.793,"matched":448},
     "1":{"ccc":0.062,"icc":0.062,"matched":435},
     "2":{"ccc":null,"icc":null,"matched":0},
     "3":{"ccc":0.000,"icc":0.000,"matched":10},
     "4":{"ccc":null,"icc":null,"matched":0}
   }
 }}
```

Per-channel `summary` is intentionally compact — the full `analysis.json` carries the rest, and this keeps `history.jsonl` tiny across hundreds of re-runs.

### `batch_analysis_included`
```json
{"ts":"2026-05-25T08:30:15.012Z","event":"batch_analysis_included",
 "data":{"batch_id":"batch_20260525_083015","session_position":4,"total":4}}
```

`session_position` is 1-indexed.

---

## `receiver.log`

Plain UTF-8 text — the complete stdout/stderr buffer of the receiver subprocess, captured by `webapp/recorder.Recorder._persist_run_artefacts` when the recording is stopped (or when the subprocess is detected to have exited unattended). One line per receiver `print()`.

Typical content: mux discovery (`Active PPG channels: [0, 1, 2, 3, 4]`), per-100-sample throughput notices (`Freq:NHz`), error messages, final exit notice. Use the dashboard's **View receiver log** button to tail the last 500 lines without opening the file directly.

---

## `batch_analyses/batch_<ts>.json`

JSON object. Written by `POST /api/analyze_all` via `webapp/sessions.save_batch_analysis`. One per Analyze-all run; never overwritten (the timestamp in the filename guarantees uniqueness).

Top-level shape:

```json
{
  "batch_id": "batch_20260525_083015",
  "created_at": "2026-05-25T08:30:15.012Z",
  "n_sessions_total": 4,
  "n_sessions_analyzed": 4,
  "failed_sessions": [],
  "crop_window": { "start_s": null, "end_s": null },
  "fst_unavailable": true,
  "sessions": [
    {
      "session_name": "session_20260516_174059",
      "started_at": "2026-05-16T17:40:59",
      "participant": { ... },
      "ecg": { ... },
      "results": [ ... full analyze_session result per channel ... ],
      "interpretation": { ... full session interpretation ... }
    },
    ...
  ],
  "per_site": [
    {
      "site": "finger",
      "n_channels": 4,
      "n_sessions": 4,
      "matched_beats_total": 1198,
      "ssqi":       {"mean": 0.060,  "std": 0.666,  "n": 4},
      "zsqi_mean":  {"mean": 0.034,  "std": 0.008,  "n": 4},
      "zsqi_std":   {"mean": 0.012,  "std": 0.004,  "n": 4},
      "ksqi":       {"mean": 2.180,  "std": 0.412,  "n": 4},
      "ccc":        {"mean": 0.202,  "std": 0.394,  "n": 4},
      "icc":        {"mean": 0.202,  "std": 0.394,  "n": 4},
      "pearson_r":  {"mean": 0.238,  "std": 0.383,  "n": 4},
      "bias_ms":    {"mean": 687.4,  "std": 1081.2, "n": 4},
      "loa_span_ms":{"mean": 5160.0, "std": 5841.0, "n": 4},
      "rmse_ms":    {"mean": 1499.6, "std": 1812.9, "n": 4},
      "mae_ms":     {"mean": 704.6,  "std": 1076.3, "n": 4}
    },
    ...
  ],
  "stratified_by_skin": [
    {
      "group": "light",
      "fst_range": "I-II",
      "n_sessions": 12,
      "per_site":         [ ... same shape as the top-level per_site ... ],
      "hr_per_channel":   [ ... same shape as the top-level hr_per_channel ... ],
      "sdnn_per_channel": [ ... ],
      "lfhf_per_channel": [ ... ]
    },
    { "group": "medium", "fst_range": "III-IV", "n_sessions": 4,  ... },
    { "group": "dark",   "fst_range": "V-VI",   "n_sessions": 4,  ... }
  ],
  "interpretation": {
    "headline": "Across 4 sessions and 5 body sites: best site is finger (mean CCC 0.20, only poor agreement).",
    "site_summaries": [
      { "site": "finger",   "grade": "bad", "text": "finger (n=4 channels, Σ matched 1198 beats); mean CCC 0.202 — failing agreement at this site; SSQI +0.06 (borderline shape); large mean bias +687 ms — peak detector likely off for several channels." },
      ...
    ],
    "notes": [
      "Performance gap: finger (0.202) → wrist (0.001). Expect a similar ranking in the paper's site-level table.",
      "Fitzpatrick stratification unavailable — no session in this batch carries an FST grade in participant.json. Save metadata on each session (left sidebar) to unlock the FST I-III vs IV-VI subgroup tables the manuscript expects."
    ]
  }
}
```

Each `per_site[i].{metric}` block is `{mean, std, n}` where `n` is the count of finite values that went into the aggregate (matches the `_mean_std` helper in `webapp/analysis.py`). `ssqi`, `zsqi_mean`, `zsqi_std`, `ksqi` are aggregated over every channel of every session at that site; `ccc`, `icc`, `pearson_r`, `bias_ms`, `loa_span_ms`, `rmse_ms`, `mae_ms` only over channels that produced matched-interval `stats` (≥ 2 matched beats).

`stratified_by_skin` repeats the four cohort aggregations (`per_site`, `hr_per_channel`, `sdnn_per_channel`, `lfhf_per_channel`) within each Fitzpatrick skin-tone band — light I–II, medium III–IV, dark V–VI (`analysis._SKIN_GROUPS`). Sessions with no FST grade are dropped from every band, so the three `n_sessions` need not sum to `n_sessions_analyzed`. All three bands are always present, with empty tables when a band has no graded sessions, so the frontend layout stays stable. Batch archives saved before this field existed simply omit the key.
