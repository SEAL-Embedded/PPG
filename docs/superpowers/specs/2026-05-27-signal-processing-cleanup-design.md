# Signal-processing cleanup + JSON persistence — design spec

**Date:** 2026-05-27
**Branch:** `bugfix/signal-processing-cleanup-and-persistence`

## Goal

Fix 19 bugs across the SEAL PPG vs ECG validation pipeline so the numbers the dashboard produces — and the metadata the dashboard persists — are trustworthy. Lab lead's diagnosis: "fix signal processing — clearly very messy right now." User-reported persistence bug: saved notes don't survive across machines.

## Source of truth

Four parallel codebase audits + one JSON-persistence trace + one supplemental bug-hunt produced the catalog below. Every item cites `file:line`.

## Bug catalog (19 items)

### Persistence + data integrity (4)

- **P1.** `webapp/sessions.py:188-192` — `save_participant_metadata` is unconditional overwrite. Recording-start sends blank notes → erases prior notes. Fix: read-merge-write. Plus zero `participant.json` files exist in MDPIdata on this machine (primary evidence). [HIGH user impact]
- **P2.** MicroPython `ticks_us` on RP2040 wraps at 2^30 ≈ 17.9 min. No rollover detection in `PPG_ECG_Full_Unpacking.py:149` or `webapp/analysis.py:load_ppg/load_ecg`. Any recording > 17.9 min corrupts every later timestamp. Fix: unwrap in loader. [HIGH]
- **P3.** `sqi/ccc.py:46-57` — `bandpass`/`lowpass` lack padlen guards; short crops raise `ValueError` swallowed by `analyze_channel`'s try/except, producing 0 peaks silently. Fix: length guard mirroring `analysis.py:101`. [HIGH]
- **P4.** `webapp/analysis.py:214-215` — `leads_off_spans` computed on pre-downsample axis, returned alongside downsampled `time_s`. Frontend marks misalign. Fix: compute on downsampled grid. [LOW]

### PPG filtering — unify on paper spec (4)

- **F1.** Three divergent PPG filters: `sqi/ccc.py:53-57` (8 Hz LP, no HP), `webapp/analysis.py:83-103` (0.6-3.3 Hz BP order 2, display only), `signal_visualization/ppgvis.py:198-207` (0.6-3.3 Hz BP order 2). Paper spec: **0.5-4.0 Hz, 4th-order, zero-phase Butterworth.** Fix: add `ppg_bandpass()` to `sqi/ccc.py` at paper spec; both peak detection AND display use it. [HIGH]
- **F2.** `sqi/ccc.py:110` — `detect_ppg_peaks` uses lowpass-only; replace with the unified bandpass. [HIGH]
- **F3.** `webapp/analysis.py:83-103` — `ppg_bandpass()` parameters updated to call the canonical `sqi.ccc.ppg_bandpass`; remove duplicate butter() call. [MED]
- **F4.** `signal_visualization/ppgvis.py` — Welch PSD has a `scaling='spectrum'` × `df` double-counting bug (off by ~250×). Plus duplicate-code burden. Replace with a deprecation stub that points to the webapp pipeline. [LOW]

### Peak detection — adaptive thresholds + robust auto-flip (6)

- **D1.** `sqi/ccc.py:84-87` — `height_thresh` is computed from `filtered` *before* the conditional negation on line 85, so threshold derives from pre-flip while peak search runs on post-flip. Fix: compute threshold AFTER flip. [MED]
- **D2.** `sqi/ccc.py:87` — global p90 height threshold; a motion burst inflates p90 across the recording. Fix: rolling-window adaptive threshold (8-s window, 2-s stride). [HIGH]
- **D3.** `sqi/ccc.py:112` — global std prominence threshold for PPG; same failure mode as D2. Fix: rolling-window adaptive prominence. [HIGH]
- **D4.** `sqi/ccc.py:84` — ECG auto-flip uses single-sample extremum; one electrode pop permanently inverts whole recording. Fix: trimmed-extrema comparison (drop top/bottom 0.5%). [MED]
- **D5.** No PPG auto-flip at all. Fix: same trimmed-extrema flip in `detect_ppg_peaks`. [LOW]
- **D6.** `sqi/ccc.py:86,111` — `min_distance = int(0.4*fs)` caps detection at 150 BPM. Fix: `int(0.25*fs)` for 240 BPM ceiling. [MED]

### Interval matching + ectopic filtering (2)

- **M1.** `sqi/ccc.py:177` — symmetric `|ppi_time - rr_time| <= tol` accepts negative offsets that are physiologically impossible (PPG always lags ECG by PTT). Fix: asymmetric `0 <= ppi_time - rr_time <= max_ptt_s` (default `max_ptt_s = 0.40 s`). [MED]
- **M2.** No ectopic/outlier rejection anywhere between peak detection and HRV computation. Fix: new module `sqi/hrv_clean.py` with `clean_intervals(rr_ms, rr_times_s)` — drop intervals outside [300, 2000] ms; drop intervals differing >20% from 5-beat rolling median (Karlsson 1987). Called by both `analyze_channel` and `sleepiness._analyze_one_session`. [HIGH]

### HRV computation hardening (5)

- **H1.** `webapp/sleepiness.py:63` — `MIN_BEATS_FOR_HRV = 30` covers ~1 LF cycle. Fix: tiered — `MIN_BEATS_TIMEDOMAIN=30`, `MIN_BEATS_SAMPEN=100`, `MIN_BEATS_SPECTRAL=150`. [MED]
- **H2.** `webapp/sleepiness.py:132` — sample entropy template count differs by 1 from Richman & Moorman 2000. Fix: range `n - m` for both m and m+1, padded correctly. [LOW]
- **H3.** `webapp/sleepiness.py:175-184` — Lomb-Scargle units. With `normalize=False`, `lombscargle` returns amplitude² (ms²); `trapezoid` over `freqs` integrates over Hz → ms²·Hz, but docstring says ms². Ratios (lf_nu, hf_nu, log_lf_hf) unaffected; absolute powers off by band-width. Fix: document units OR pass `normalize=True` to get a density. [MED]
- **H4.** No per-HRV-feature CCC aggregation. The lead's original ask. Fix: new function `_aggregate_per_feature(per_session)` in `webapp/sleepiness.py`. One (ECG, PPG) point per (channel, session); two scopes (overall + per_site); n<4 → null; surface as `per_feature` top-level key. [MED]
- **H5.** Redundant ECG-feature computation in `webapp/sleepiness.py:407-412` + duplicate PPG peak detection in `:438-451` (drift risk vs `webapp/analysis.py`). Fix: cache + reuse. [LOW]

## Out of scope (deferred to lab lead per their finding #1)

- Full Pan-Tompkins R-peak detector (replace `find_peaks`)
- Foot-detection (Elgendi TERMA) for PPG instead of systolic peak
- Per-HRV-feature CCC on the per-session detail view (Phase 2 — backend table first)
- Firmware-side `ticks_us` rollover fix in `pipico_code/` (loader-side fix handles new + old CSVs)
- ECGvis.py / fullvis.py raw-signal plotting (low-impact dead code)
- LF/HF integral normalization (paper-side methodology decision)

## Architecture

Three modules carry the weight:

1. **`sqi/ccc.py`** — gains `ppg_bandpass()`, hardens `detect_r_peaks` + `detect_ppg_peaks` with rolling-window adaptive thresholds + trimmed-extrema auto-flip + 240 BPM ceiling + asymmetric PTT match window. Adds padlen guards.
2. **`sqi/hrv_clean.py`** (new) — single source of NN-interval cleaning (range gate + Karlsson rule). Called by every HRV consumer.
3. **`webapp/sleepiness.py`** — tiered MIN_BEATS, sample-entropy off-by-one fix, Lomb-Scargle docstring, per-HRV-feature CCC aggregator, deduplicated ECG-feature computation.

Plus the persistence fix in `webapp/sessions.py` (read-merge-write) and the rollover unwrap in `webapp/analysis.py` (`load_ppg` + `load_ecg`).

## Acceptance criteria

- All existing tests pass
- New tests for each bug: failing test → minimal fix → passing test
- Adversarial fixtures added to `tests/conftest.py`: motion burst, ectopic beat, missed beat, inverted signal, fast HR (180 BPM), >17.9-min recording (synthetic rollover)
- `python app.py` launches; loading a session shows new per-feature CCC table on Σ page; notes saved then "recorded over" by a recording-start still survive
- `pytest tests/` exits 0
- Branch `bugfix/signal-processing-cleanup-and-persistence` is reviewable with squashed commits per group
