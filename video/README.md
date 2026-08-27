# Live-acquisition video

Remotion replay of a recorded session, rendered to look like the signals are
arriving live. 1× real time, so the video runs as long as the recording did.

Currently built from **`session_20260715_164725`** (Pushkal, FST 4, 414 s) —
the session where all five PPG sites recover heart rate accurately:

| site | windowed HR MAE vs ECG | r | beat-to-beat CCC |
|---|---|---|---|
| finger | 0.72 bpm | 0.924 | 0.904 |
| earlobe | 0.87 bpm | 0.956 | 0.726 |
| shoulder | 1.38 bpm | 0.903 | 0.527 |
| forehead | 1.13 bpm | 0.929 | 0.492 |
| wrist | 1.90 bpm | 0.587 | 0.300 |

## Rebuild

```bash
../.venv/bin/python export_session.py     # writes public/session.json
npm install
npx remotion render LiveAcquisition out/live_acquisition.mp4 --concurrency=10
npx remotion studio                        # to scrub/tweak interactively
```

To use a different session, change `NAME` at the top of `export_session.py`
and re-export — the composition reads its length and channel count from the
JSON, so nothing else needs touching.

## Sidebar trackers

All three mini charts plot every stream (ECG + 5 PPG sites) against elapsed
time, growing as the video plays. The headline value on each card is the ECG
reference.

| Tracker | Window | How it's computed |
|---|---|---|
| Heart rate | 20 s trailing | Median of beat-to-beat intervals |
| HRV SDNN | 60 s trailing | Sample std (ddof=1) of cleaned NN intervals, ≥10 beats required |
| LF/HF | 120 s trailing | NN tachogram → 4 Hz interpolation → linear detrend → Welch PSD; LF 0.04–0.15 Hz over HF 0.15–0.40 Hz, ≥20 beats required |

NN intervals are cleaned with the repo's own `sqi/hrv_clean.clean_intervals`
(physiological range gate + Karlsson 1987 ±20% rule), so these match the
pipeline's definitions rather than inventing new ones.

Both HRV windows are **short-term estimates, not the Task Force 5-minute
standard** — the recording is only 6:54 total. The window length is on every
card label for that reason. LF/HF is on a log axis because it ranges from
~0.3 to ~10 here and a linear axis buries the sub-1 half.

## Privacy

`export_session.py` masks the participant name to asterisks as it writes the
JSON, so the real name never reaches the bundle or the rendered video. The
session timestamp ID and Fitzpatrick type are kept.

## Notes

- PPG traces are shown in the **0.6–3.3 Hz cardiac band** the systolic peak
  detector actually runs on; the raw signal is dominated by DC drift and
  would show almost nothing. The ECG trace is raw.
- Markers are detected beats (R-peaks on ECG, systolic peaks on PPG), taken
  from `webapp/analysis.py` — the same detector the dashboard uses.
- Per-row BPM is the median of beat-to-beat intervals over a trailing 20 s
  window, which is how a bedside monitor derives it.
- Palette validated for colour-vision deficiency against the dark surface;
  see the note in `src/theme.ts`.
