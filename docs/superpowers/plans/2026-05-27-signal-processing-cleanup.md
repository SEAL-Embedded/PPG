# Signal-processing cleanup + JSON persistence — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax. **Each subagent owns one file family — no cross-file edits between subagents.**

**Goal:** Fix 19 cataloged bugs (see `docs/superpowers/specs/2026-05-27-signal-processing-cleanup-design.md`) across signal processing, HRV computation, and metadata persistence. TDD throughout.

**Architecture:** Three new/expanded modules — `sqi/ccc.py` (hardened), `sqi/hrv_clean.py` (new), `webapp/sleepiness.py` (extended) — plus targeted fixes in `webapp/sessions.py`, `webapp/analysis.py`, and `signal_visualization/ppgvis.py`.

**Tech Stack:** Python 3, numpy, scipy.signal (filtfilt, butter, find_peaks, lombscargle), FastAPI, pytest, pingouin (optional).

---

## Dispatch wave plan

### Wave 1 — Parallel (5 subagents on disjoint files)

| Subagent | Owns | Bugs |
|---|---|---|
| W1-A | `webapp/sessions.py` + `tests/test_sessions.py` + `tests/test_api.py` | P1 (notes persistence) |
| W1-B | `webapp/analysis.py` + `tests/test_analysis.py` | P2 (rollover unwrap), P4 (leads_off alignment), B3 (wire unified filter) |
| W1-C | `sqi/ccc.py` + `tests/test_sqi.py` + `tests/conftest.py` | P3, F1, F2, D1–D6, M1 (the entire `ccc.py` cleanup) |
| W1-D | NEW `sqi/hrv_clean.py` + `tests/test_hrv_clean.py` | M2 (ectopic filter as a new module) |
| W1-E | `webapp/sleepiness.py` + `tests/test_sleepiness.py` | H1–H5 (the entire HRV cleanup including per-feature CCC) |

### Wave 2 — Sequential after Wave 1 (integration + cleanup)

| Subagent | Owns | Bugs |
|---|---|---|
| W2-F | `signal_visualization/ppgvis.py` | F4 (deprecation stub) |
| W2-G | `webapp/static/app.js` + `webapp/static/index.html` | Frontend per-feature CCC table |
| W2-H | `docs/INTERPRETATION_GUIDE.md` | Update "known gaps" section to reflect fixes |

---

## Task instructions per subagent

Each subagent's full prompt is constructed by the controller (this session). The high-level contract:

1. Read the design spec (`docs/superpowers/specs/2026-05-27-signal-processing-cleanup-design.md`) for the bug catalog
2. Write failing tests for each bug in scope (TDD)
3. Verify tests fail with the current code
4. Implement minimal fixes
5. Verify tests pass
6. Commit each (bug, test, fix) trio separately
7. Report DONE / DONE_WITH_CONCERNS / NEEDS_CONTEXT / BLOCKED

---

## Wave 1 task details

### W1-A — Notes persistence (P1)

**Files:** Modify `webapp/sessions.py:188-192` (`save_participant_metadata`). Modify `tests/test_sessions.py` and `tests/test_api.py` to assert `notes` round-trip.

- [ ] Test: `test_notes_round_trip_through_disk_and_survive_blank_save` — write notes, then re-save with blank notes, GET — notes preserved.
- [ ] Test: `test_save_metadata_preserves_existing_keys_not_in_payload` — write {pid, fst, notes, sites}, then save {pid only} — fst/notes/sites preserved.
- [ ] Run, expect FAIL.
- [ ] Fix `save_participant_metadata` to read-merge-write: load existing, shallow merge incoming, write merged.
- [ ] Run, expect PASS.
- [ ] Add 2 assertions to existing `test_post_metadata_round_trip` in `test_api.py` to check `meta["notes"]`.
- [ ] Commit: `fix(sessions): merge participant.json on save so notes survive overwrites`

### W1-B — Rollover unwrap + leads_off alignment (P2, P4, B3)

**Files:** Modify `webapp/analysis.py` (`load_ppg`, `load_ecg`, `leads_off_spans`, `ppg_bandpass`). Add `tests/test_analysis_rollover.py`.

- [ ] Test: `test_load_ppg_unwraps_30bit_ticks_us` — synth CSV crossing 2^30 boundary, assert `np.diff(ts_ms) > 0` everywhere.
- [ ] Test: `test_load_ecg_unwraps_30bit_ticks_us` — same for ECG (3-column).
- [ ] Run, expect FAIL.
- [ ] In `load_ppg` and `load_ecg`: after parsing `ts_us` array, detect backward steps > 2^29 µs and add 2^30 µs cumulatively. Implement as helper `_unwrap_ticks_us(ts_us)`.
- [ ] Run, expect PASS.
- [ ] Test: `test_leads_off_spans_aligned_with_downsampled_axis` — verify spans fall within `xs` range.
- [ ] Fix: compute `leads_off_spans` after downsampling, OR pass the original `ts_s` separately. Pick the second (less code change).
- [ ] In `webapp/analysis.py:ppg_bandpass`: re-import `ppg_bandpass` from `sqi.ccc` after W1-C lands; delegate. (Order with W1-C: W1-B includes a small comment-only refactor and a `from sqi.ccc import ppg_bandpass` import that will land *after* W1-C's commit lands. Document this as a dependency.)
- [ ] Commit each test+fix pair separately.

### W1-C — `sqi/ccc.py` overhaul (P3, F1, F2, D1–D6, M1)

**Files:** Modify `sqi/ccc.py` heavily. Add adversarial fixtures to `tests/conftest.py`. Add tests to `tests/test_sqi.py`.

- [ ] Add adversarial fixtures to `conftest.py`: `synth_motion_burst_ecg` (clean 60 BPM + 5-s motion burst at +5× amplitude), `synth_inverted_ecg`, `synth_fast_hr_ecg` (180 BPM), `synth_short_signal` (1-s signal, too short for filtfilt).
- [ ] Test: `test_bandpass_short_signal_returns_none_not_raises` (P3) — for `<= 3·padlen` samples, return None or pass through silently.
- [ ] Fix: add padlen guard to both `bandpass` and `lowpass`.
- [ ] Test: `test_ppg_bandpass_at_paper_spec_attenuates_above_4hz` (F1) — feed 5 Hz sine + 2 Hz sine, assert 5 Hz attenuated >20 dB.
- [ ] Fix: add `def ppg_bandpass(sig, fs, low=0.5, high=4.0, order=4)` at paper spec.
- [ ] Test: `test_detect_ppg_peaks_uses_paper_bandpass` (F2) — patch the internal filter call, assert it's the new `ppg_bandpass`.
- [ ] Fix: replace `lowpass(ppg, fs, cutoff=8.0)` with `ppg_bandpass(ppg, fs)` in `detect_ppg_peaks`.
- [ ] Test: `test_detect_r_peaks_threshold_computed_after_flip` (D1) — feed an inverted-ECG fixture, assert peaks are correctly detected with the flipped-signal threshold.
- [ ] Fix: move `height_thresh = np.percentile(filtered, 90) * 0.5` AFTER the `filtered = -filtered` line.
- [ ] Test: `test_detect_r_peaks_handles_motion_burst` (D2) — `synth_motion_burst_ecg` fixture; assert detected beats both before and after the burst.
- [ ] Fix: replace global p90 with rolling-window threshold via `scipy.signal.medfilt` or a manual sliding p90 over 8-s windows with 2-s stride. Keep result as a per-sample threshold array.
- [ ] Test: `test_detect_ppg_peaks_handles_amplitude_drift` (D3) — synth PPG with amplitude doubling halfway; assert beats in both halves detected.
- [ ] Fix: same rolling-window approach for `prominence_thresh`.
- [ ] Test: `test_detect_r_peaks_robust_to_single_negative_spike` (D4) — clean 60 BPM ECG + one −10× spike at t=5s; assert no auto-flip triggered.
- [ ] Fix: replace `np.abs(centered.min()) > centered.max()` with trimmed-extrema comparison (top/bottom 0.5%).
- [ ] Test: `test_detect_ppg_peaks_auto_flips_inverted` (D5) — inverted PPG fixture; assert peaks detected normally.
- [ ] Fix: add the same trimmed-extrema flip to `detect_ppg_peaks`.
- [ ] Test: `test_detect_r_peaks_handles_180_bpm` (D6) — `synth_fast_hr_ecg`; assert ≥150 detected R-peaks (60 s × 3 Hz).
- [ ] Fix: `min_distance = int(0.25 * fs)` for both detectors.
- [ ] Test: `test_match_intervals_rejects_negative_ptt` (M1) — synth pair where ppi_time precedes rr_time by 100 ms; assert no match.
- [ ] Test: `test_match_intervals_accepts_400ms_ptt` — synth pair where ppi_time follows rr_time by 380 ms; assert match.
- [ ] Fix: replace symmetric `np.abs(...)` window with asymmetric `0 <= dt <= max_ptt_s` where `max_ptt_s = 0.40` (configurable via kwarg, default 0.40).
- [ ] Commit each (test, fix) pair separately. Aim for 9-10 small commits.

### W1-D — `sqi/hrv_clean.py` new module (M2)

**Files:** Create `sqi/hrv_clean.py`. Create `tests/test_hrv_clean.py`.

- [ ] Test: `test_clean_intervals_drops_out_of_range` — input [800, 200, 800, 3000, 800]; expect [800, NaN, 800, NaN, 800] (or filtered: [800, 800, 800] with mask).
- [ ] Test: `test_clean_intervals_karlsson_drops_outliers_vs_local_median` — input [800]×10 + [1600] + [800]×10; the 1600 ms beat dropped.
- [ ] Test: `test_clean_intervals_preserves_clean_series` — input clean 60-beat sequence at 800±30 ms; expect no drops.
- [ ] Test: `test_clean_intervals_returns_aligned_timestamps` — when dropping intervals, the time-array stays aligned (same length filter).
- [ ] Test: `test_clean_intervals_handles_empty_array`.
- [ ] Implement `clean_intervals(intervals_ms, times_s, range_ms=(300, 2000), karlsson_pct=0.20, window=5) -> (intervals_clean, times_clean, mask)`.
- [ ] Implementation strategy: two-pass — (1) range gate; (2) rolling median over `window` neighbors, drop if `|x - med| / med > karlsson_pct`.
- [ ] Commit per (test, fix) pair.

### W1-E — `webapp/sleepiness.py` overhaul (H1-H5)

**Files:** Modify `webapp/sleepiness.py`. Modify `tests/test_sleepiness.py`.

- [ ] Test: `test_min_beats_tiered_time_vs_spectral` — feed 50-beat series; expect SDNN finite, LF/HF NaN, SampEn NaN.
- [ ] Fix H1: split `MIN_BEATS_FOR_HRV` into `MIN_BEATS_TIMEDOMAIN=30`, `MIN_BEATS_SAMPEN=100`, `MIN_BEATS_SPECTRAL=150`. Apply each gate in `compute_hrv_features`.
- [ ] Test: `test_sample_entropy_matches_richman_reference` — known input → expected SampEn within tolerance vs reference value. Use a published example (e.g., constant-perturbed series).
- [ ] Fix H2: change `range(n - m_ + 1)` to `range(n - m)` for both `_phi(m)` and `_phi(m+1)`.
- [ ] Test: `test_lomb_scargle_units_documented` — assert the function's docstring says "ms²·Hz" OR returns ms² after normalization correction.
- [ ] Fix H3: docstring update for `_lomb_scargle_band_power` to explicitly say "ms²·Hz" units (least-invasive fix), AND add a note in `CAVEATS`.
- [ ] Test: `test_aggregate_per_feature_overall_finite_n` — synth `per_session` with 5 sessions × 5 channels each having known feature values; assert overall CCC table has all 13 features, each with n=25 (or appropriate).
- [ ] Test: `test_aggregate_per_feature_per_site_groups_correctly` — assert per-site grouping uses `channel_sites`.
- [ ] Test: `test_aggregate_per_feature_returns_null_below_min_n` — synth 3 sessions; assert ccc=null.
- [ ] Test: `test_aggregate_per_feature_in_response_payload` — full `analyze_sleepiness` call; assert `result["per_feature"]` exists with `overall` and `per_site` keys.
- [ ] Fix H4: add `_aggregate_per_feature(per_session)` returning `{"overall": {feat: stats}, "per_site": {site: {feat: stats}}}`; wire into `analyze_sleepiness` response. Use the existing `compute_ccc` from `sqi.ccc`. n<4 → all stats null + `caveat: "small n"` flag.
- [ ] (Optional) Test: `test_no_redundant_ecg_feature_recomputation` — count calls to compute_hrv_features for one session; assert ECG features computed exactly once. Use unittest.mock to patch.
- [ ] (Optional) Fix H5: cache ECG features in Stage 2's loop, read back in Stage 3.
- [ ] Commit per (test, fix) pair.

---

## Wave 2 (sequential after Wave 1 lands and passes)

### W2-F — Deprecate `signal_visualization/ppgvis.py`

- [ ] Replace `ppgvis.py` content with a 30-line deprecation stub:
  ```python
  """DEPRECATED — see webapp/sleepiness.py for the maintained HRV pipeline.

  This file's Welch PSD has a scaling × df double-counting bug (off ~250×).
  Kept as a stub for any external script that imports it.
  """
  raise ImportError(
      "signal_visualization.ppgvis is deprecated. Use webapp.sleepiness "
      "for HRV features and the dashboard's Σ page for visualisation."
  )
  ```
- [ ] Verify no other file imports `ppgvis` (grep).
- [ ] Commit.

### W2-G — Frontend per-feature CCC table

- [ ] Add `<section id="per-feature-ccc-table">` to `webapp/static/index.html` inside the sleepiness summary page.
- [ ] Add render function `renderPerFeatureCCC(per_feature)` in `webapp/static/app.js` — two tables (overall, per-site), columns: feature | n | CCC | Pearson | bias | LOA span | RMSE.
- [ ] Apply same colour coding as existing CCC table (green ≥0.95, amber 0.90-0.95, red <).
- [ ] Show "—" for null CCC; amber n-pill for 4 ≤ n < 10.
- [ ] Manual verification: load Σ page in browser, confirm table renders with synthetic / real data.
- [ ] Commit.

### W2-H — Documentation

- [ ] Update `docs/INTERPRETATION_GUIDE.md` "Known code-vs-manuscript gaps" section to remove items now fixed.
- [ ] Add a new section "Per-HRV-feature agreement (Σ page)" explaining the new table.
- [ ] Commit.

---

## Final integration

- [ ] Run `pytest tests/` from repo root — all pass.
- [ ] Start `python app.py`, navigate to a session, run analysis, navigate to Σ page, verify new table renders.
- [ ] Squash + rebase per-bug commits into 5-6 logical commits.
- [ ] Open PR for human review.
