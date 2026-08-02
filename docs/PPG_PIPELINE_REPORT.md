# PPG Processing Pipeline Report: Raw CSV to HRV Metrics

Purpose: a self-contained description of the SEAL PPG/ECG pipeline as it is
actually implemented in code, written so an independent reviewer can check
each stage against the published literature. Every parameter quoted here is
taken directly from the source files named in each section. A final section
lists code-vs-documentation discrepancies that the reviewer should resolve
before citing any of these methods.

Source files:
- Acquisition firmware: `pipico_code/ppgcode/main.py`, `pipico_code/fullpipico.py`
- On-disk schema: `docs/DATA_FORMATS.md`
- Core signal processing: `sqi/ccc.py`
- NN-interval cleaning: `sqi/hrv_clean.py`
- SQI metrics: `sqi/SSQI_algorithm.py`, `sqi/zcr_sqi.py`
- Per-session driver and HRV/aggregation: `webapp/analysis.py`

---

## 0. Acquisition (hardware front end)

| Item | Value | Source |
|------|-------|--------|
| PPG sensor | MAX30102 (IR/Red), one per TCA9548A mux lane, up to 8 lanes | `main.py:24-33` |
| PPG sensor config | `set_sample_rate(3200)`, `set_fifo_average(8)`, `set_pulse_width(69)`, `set_adc_range(16384)`, LED mode 1 (Red only) | `main.py:59-69` |
| Nominal per-sensor rate | 3200 / 8 = 400 Hz FIFO output | `fullpipico.py:46-51` |
| Effective per-channel rate (recorded) | **400.0 Hz** measured (median dt = 2500 us, every channel/session). The firmware drains each sensor's whole FIFO per round-robin pass and back-dates samples to the nominal 2500 us spacing, so mux sharing does **not** reduce the per-channel timestamp rate | measured from `MDPIdata/session_*`; `fullpipico.py:46-51` back-dating |
| ECG sensor | AD8232, sampled at `ECG_INTERVAL_US = 2500` (400 Hz target) | `fullpipico.py:44` |
| Effective ECG rate (recorded) | **~376 Hz** typical (median dt ~2661 us); one session at ~342 Hz | measured from `MDPIdata/session_*` |
| Timestamp clock | Pico `ticks_us()` (30-bit, wraps every 2^30 us ~ 17.9 min) | firmware + `analysis._unwrap_ticks_us` |
| Sample value | Red ADC count; PPG 0-262143, ECG 0-65535 | `DATA_FORMATS.md` |

Note: the `fs_hz` example values in `docs/DATA_FORMATS.md` (~130 Hz PPG, ~323 Hz
ECG) are from a since-deleted 20260516 session and are **stale**. The current
on-disk sessions measure 400 Hz PPG and ~376 Hz ECG as in the table above.

The PPG value stored is the **Red** channel (`pop_red_from_storage`), not IR.
A reviewer validating against PPG literature should note most finger-PPG HRV
work uses IR or green; Red is acceptable but worth confirming against the
specific prior studies being matched.

On-disk format (headerless CSV, written by `PPG_ECG_Full_Unpacking.py`):
- `ecg_data.csv`: `timestamp_us, sample, leads_off`
- `ppg_data_ch{N}.csv`: `timestamp_us, sample`

---

## 1. Loading and timestamp conditioning

Functions: `analysis.load_ppg`, `analysis.load_ecg`, `analysis._unwrap_ticks_us`.

1. Read CSV with `on_bad_lines="skip"` (tolerates a partially flushed last line during live view).
2. Coerce both columns to numeric; drop rows where either is NaN.
3. **Tick unwrap**: detect backward jumps `< -2^29` in the us timestamp and add
   `2^30` to all subsequent samples (`_unwrap_ticks_us`). Handles the Pico's
   30-bit clock wrap for recordings longer than ~17.9 min.
4. Convert us to ms (`/1000.0`). All downstream interval math is in ms.

`infer_fs` = `1000 / median(diff(ts_ms))`. Sampling rate is never assumed; it
is measured per channel from recorded timestamps.

---

## 2. Per-channel PPG conditioning (before peak detection)

Function: `analysis.analyze_channel` (lines ~765-830), helpers `_drop_non_monotonic`,
`_resample_uniform`, `_remove_outliers`.

Order of operations on each PPG channel:

1. **Drop non-monotonic timestamps** (`_drop_non_monotonic`): keep only samples
   whose timestamp strictly exceeds the running max. Required because the
   drain-all firmware back-dates burst samples, which would otherwise produce
   negative PPIs and break cubic interpolation.
2. **Cubic resample onto a uniform grid** at the measured `fs` (`_resample_uniform`,
   `scipy.interpolate.interp1d kind="cubic"`). Butterworth `filtfilt` assumes
   uniform sampling, so this precedes any filtering.
3. **Rolling-median outlier scrub** (`_remove_outliers`): flag samples that sit
   more than `n_sigma = 4.25` standard deviations from a centered rolling
   median (`window = 15` samples), replace by linear interpolation. This is the
   motion-spike removal.

The result, `clean_sig`, is the **resampled, outlier-removed, pre-bandpass**
trace. SSQI, ZSQI and KSQI are computed on `clean_sig`. The bandpass is applied
only inside the peak detector (SSQI/KSQI on a bandpassed signal would distort
the skew and the tail weight the two moments measure).

---

## 3. Filtering

Function: `sqi.ccc.ppg_bandpass` (also re-exported via `analysis.ppg_bandpass`).

- **PPG bandpass**: Butterworth, `low=0.5 Hz`, `high=8.0 Hz`, `order=2`,
  zero-phase (`filtfilt`). *(See Discrepancy D1 — a recent commit message
  claims 0.6-3.3 Hz; the code default is 0.5-8.0 Hz.)*
- Rationale in code: 0.5 Hz HP strips baseline wander; 8 Hz LP keeps the
  cardiac fundamental (~1 Hz) plus harmonics for systolic-peak shape.
- Returns `None` (channel skipped) when fs invalid, cutoffs outside (0, Nyquist),
  or signal shorter than `filtfilt` padlen.
- **ECG**: the R-peak detector (`detect_r_peaks`) does **not** band-pass the
  ECG; it is prominence-based and runs on the raw signal directly (see §4).

---

## 4. Peak detection

### ECG R-peaks: `sqi.ccc.detect_r_peaks`
Prominence-based detection on the **raw** ECG (no band-pass, no derivative /
squaring / integration). The R-peaks on these seated, ~5-min, single-lead
recordings are high-SNR and very distinct, so detection is `scipy.signal.find_peaks`
with three scale/rate-relative guards:
- **Polarity**: orient the trace so R-peaks point up, decided from the robust
  1st/99th percentiles (not raw min/max, so one artifact spike can't flip the
  whole recording). Handles an inverted lead. The flip only fires when the
  margin is decisive -- `_R_ORIENTATION_MIN_RATIO = 1.2` between the larger and
  smaller percentile distance. Below that, R and S amplitudes are effectively
  tied, so the code defaults to upright and emits a `UserWarning` naming the
  ratio. Rationale: a wrong flip does not fail loudly, it silently relocates
  every peak onto the S-wave. Real case in this dataset:
  `session_20260715_153323` decided polarity on a **1.0%** margin (dist_low
  12131 vs dist_high 12011) and was being detected upside-down; the guard
  corrects it, cutting that session's raw RR standard deviation from 61.4 ms to
  54.1 ms. It is the only one of the 20 sessions whose orientation changes.
- **Relative prominence**: `prominence = 0.35 * (p99 - p40)` of the oriented
  signal — 35% of the recording's own R-amplitude. Prominence measures rise
  above the surrounding troughs, so it is immune to baseline wander and DC
  offset; there is no absolute ADC count anywhere.
- **Refractory spacing**: `distance = int(0.28*fs)` ~ 280 ms (caps ~214 BPM).
  Because `find_peaks` drops the smaller of any two peaks inside `distance`,
  this also removes the T-wave trailing each R-peak.

Returned indices are local maxima of the (oriented) raw signal — the actual
peak tips — so plotted markers sit exactly on each R-peak. Parameters live in
the module constants `_R_REFRACTORY_S = 0.28` and `_R_PROMINENCE_FRAC = 0.35`.

### PPG systolic peaks (webapp path)
The active webapp detector is selected by the module constant
`analysis.PPG_PEAK_DETECTOR` and dispatched through `analysis._detect_ppg_peaks`.
Both options first band-pass with `ppg_bandpass` (0.5-8 Hz, 2nd-order zero-phase
Butterworth), and both detect on the cubic-resampled uniform grid.

**Active: `"terma"` -> `analysis.detect_ppg_peaks_terma`** (Elgendi 2013, the
TERMA two-event-related-moving-averages detector).
- Reference: Elgendi M, et al. "Systolic Peak Detection in Acceleration
  Photoplethysmograms...", PLoS ONE 2013;8(10):e76585.
- 0.5-8 Hz band-pass -> clip negatives to 0 -> square (emphasises the systolic
  upslope) -> two moving averages: `MA_peak` over `w1 = 111 ms`, `MA_beat` over
  `w2 = 667 ms`.
- Threshold `THR1 = MA_beat + beta*mean(squared)`, `beta = 0.02`. "Blocks of
  interest" are the runs where `MA_peak > THR1`; blocks narrower than `w1` are
  rejected as noise; within each surviving block the systolic peak is the max
  of the **band-passed** signal.
- Small-RR doublet filter (`_filter_doublets`, `frac=0.6`): drop a peak closer
  than 0.6×median-RR to the last kept peak, keeping the taller.
- The threshold adapts to the local signal level, so unlike the find_peaks
  detector it needs no per-recording prominence/distance tuning.

**Alternative: `"prominence"` -> `analysis.detect_ppg_peaks_bp`** (scale-relative
`find_peaks`).
- `find_peaks(bp, distance=max(1,int(0.5*fs)), prominence=1.1*std(bp), width=0.13)`,
  then the same doublet filter.
  - `distance = 0.5*fs` ~ 500 ms (caps ~120 BPM).
  - `prominence = 1.1 * std(bandpassed)` (scale-relative, robust to ADC scale).
  - `width=0.13` is in **samples** (scipy default units), i.e. effectively no
    width constraint.

Both run on the uniform grid; the detected peak *times* come from the grid's own
evenly-spaced real-time axis (preserving the sub-sample timing the cubic resample
recovered) and are **not** snapped back to the nearest recorded sample (see
`analyze_channel`).

### Legacy/standalone PPG detector: `sqi.ccc.detect_ppg_peaks`
- Used by the CLI `python sqi/ccc.py`, not the webapp.
- `ppg_bandpass` (0.5-8 Hz) then `find_peaks(distance=int(0.4*fs),
  prominence=0.5*std)`. No doublet filter. Differs from both webapp detectors
  (D3).

---

## 5. Intervals and matching

Functions: `sqi.ccc.peaks_to_intervals`, `sqi.ccc.match_intervals`.

- **Intervals**: `diff` of peak timestamps (ms). RR from ECG, PPI from PPG.
  Computed from recorded timestamps, not sample index, so dropped samples or
  jitter do not bias intervals.
- **Matching** (`match_intervals`): nearest-neighbour in time, greedy, each PPI
  claimed once. Tolerance `tol_s = min(0.5, max(0.15, median_RR/2))`.
  Nearest-neighbour absorbs the pulse-transit-time lag (~200-300 ms) without
  assuming a fixed PTT.

---

## 6. NN-interval cleaning (HRV only)

Function: `sqi.hrv_clean.clean_intervals`.

Two passes, applied to RR (ECG) and PPI (PPG) **before HRV metrics**:
1. **Physiological range gate**: drop intervals outside [300, 2000] ms (30-200 BPM).
2. **Karlsson 1987 rule**: drop intervals differing > 20% (`karlsson_pct=0.20`)
   from a local rolling median over `window=5` neighbours. Median is computed
   only over pass-1 survivors so an ectopic cannot contaminate its own reference.
   Reference cited in source: Karlsson et al. 1987, Comput Biomed Res 20(4):333-340.

**Scope note**: cleaning feeds **SDNN and frequency-domain HRV only**. The raw
(uncleaned) PPI/RR are deliberately kept for the CCC/Bland-Altman/ICC matched-beat
agreement, so those reflect raw detector behaviour. (See `analyze_channel`
lines ~849-863.)

---

## 7. HRV / agreement metrics

### Time-domain
- **Mean HR (bpm)**: count-based, `60 * n_peaks / duration_s` (both ECG and PPG).
- **SDNN (ms)**: sample standard deviation (`ddof=1`) of the cleaned NN series.
  Matches Task Force 1996 definition. (`analyze_channel` ~862, `analyze_session` ~951-959.)

### Frequency-domain: `analysis._freq_domain_metrics`
- `scipy.signal.welch` on the cleaned NN series (ms), min 50 beats.
- Cubic-spline resamples NN onto a uniform 4 Hz grid, linear-detrends, runs
  Welch PSD (Hamming, `nperseg=300` = 75 s, `nfft>=4096`), integrates over
  Task Force 1996 bands: VLF 0.003-0.04, LF 0.04-0.15, HF 0.15-0.40 Hz.
- Previously called `pyhrv.frequency_domain.welch_psd` behind a try/except.
  pyhrv pulls in `spectrum` (C/Fortran toolchain), is unmaintained and predates
  numpy 2, so it was not installed -- every LF/HF value silently came back NaN
  and the LF/HF batch table rendered empty. scipy is a hard dependency and runs
  the identical method, so the metric now always computes.
- Outputs: `vlf_power_ms2`, `lf_power_ms2`, `hf_power_ms2`, `lf_hf_ratio`.
- NN passed in **ms** (the code explicitly notes that passing seconds, as the
  legacy `old/PPGanalysis.py` does, scales band powers by 1e-6).

### Agreement (PPG PPI vs ECG RR): `sqi.ccc.compute_ccc`
- **Lin's CCC** (Lin 1989): `rho_c = 2*cov_xy / (var_x + var_y + (mu_x-mu_y)^2)`,
  using **population** variance/covariance (`ddof=0`) — correct per Lin's formula.
- **Pearson r**: `cov_xy / sqrt(var_x*var_y)`.
- **Bland-Altman**: `bias = mean(PPG-ECG)`, `LOA = bias +/- 1.96*std(diff, ddof=1)`.
- **RMSE, MAE** on the differences.
- CCC label bins (Lin 1989): >0.99 almost perfect, 0.95-0.99 substantial,
  0.90-0.95 moderate, <0.90 poor (`ccc_label`).

### ICC: `analysis._safe_icc`
- `pingouin.intraclass_corr`, **ICC(A,1)** = two-way mixed, single rater,
  **absolute agreement** (penalizes systematic offset, consistent with CCC's
  bias term). Requires >= 4 matched beats. Returns ICC + 95% CI.

### Signal quality indices
- **SSQI** (`SSQI_algorithm.Ssqi`): skewness, `mean(((x-mu)/sigma)^3)` (population
  sigma). Reference cited in interpretation text: Krishnan et al. 2010 (SSQI>=1
  ~ clean). Computed on `clean_sig` (pre-bandpass).
- **ZSQI** (`zcr_sqi.windowed_zcr`): windowed zero-crossing rate of the
  mean-subtracted signal, `window=5 s`, `step=1 s`. Reports mean/std/max.
- **KSQI** (`KSQI_algorithm.Ksqi`): kurtosis, `mean(((x-mu)/sigma)^4)` (population
  sigma), **Pearson / non-excess** form -- no `-3`, so Gaussian = 3.0. Elgendi
  2016: clean finger PPG ~ 2.06 +/- 0.16. Computed on `clean_sig`
  (pre-bandpass). Reported only -- excluded from the verdict roll-up and from
  any acceptance gate (Elgendi ranked it last of eight PPG SQIs for class
  discrimination).

---

## 8. Windowed within-session agreement (optional)

`_windowed_hr_agreement` (30 s window, 5 s step, >=3 peaks/side) and
`_windowed_sdnn_agreement` (60 s window, 10 s step, >=10 intervals/side) pair
ECG vs PPG per window and run the same CCC/ICC/Bland-Altman on the per-window
vectors.

---

## 9. Cross-session aggregation

`analyze_all_sessions` runs `analyze_session` over every `MDPIdata/session_*/`,
then aggregates:
- `_aggregate_per_site`: groups channels by labelled body site, reports
  mean +/- std (`ddof=1`) for SSQI, ZSQI, KSQI, CCC, ICC, Pearson, bias, LOA span,
  RMSE, MAE. Site-level CCC/ICC/etc. only include channels with >= 2 matched beats.
- `_hr_agreement_per_channel`, `_sdnn_agreement_per_channel`,
  `_lfhf_agreement_per_channel`: one paired point per session per channel,
  then CCC/ICC/Bland-Altman across sessions.
- Fitzpatrick (FST) stratification is supported in schema but flagged
  `fst_unavailable` whenever no session carries an FST grade.

---

## Validation checklist for the reviewing agent

Standard methods that **do** match the cited literature as implemented:
- Lin's CCC formula and population-variance denominator (Lin 1989). OK.
- Bland-Altman bias +/- 1.96 SD limits of agreement. OK.
- ICC(A,1) absolute-agreement choice for method comparison. OK.
- Task Force 1996 SDNN definition and VLF/LF/HF band edges (scipy Welch). OK.
- Karlsson 1987 20% local-median ectopic rule, [300,2000] ms gate. OK.
- SSQI skewness as a PPG quality index (Krishnan 2010, Elgendi). OK.
- KSQI kurtosis in the Pearson (non-excess) form, matching Elgendi 2016 eq. and
  `vital_sqi`. OK. Band edges in `_grade_ksqi_text` are dashboard reading aids
  anchored to Elgendi's per-class values plus the sinusoid (1.5) / Gaussian
  (3.0) reference points -- Elgendi publishes no KSQI cut-off, so do not cite
  them as literature thresholds.

Discrepancies / items to resolve before citing methods:

- **D1 - PPG bandpass band**: code default in `sqi/ccc.py:ppg_bandpass` is
  **0.5-8.0 Hz, order 2**, and the module docstring in `analysis.py` says the
  same. But the most recent commit message reads "Set canonical PPG bandpass to
  0.6-3.3 Hz, order 2", and `sqi/ccc.py` currently has uncommitted edits.
  Confirm which band is intended. 0.5-8 Hz is wider than the typical 0.5-4 Hz
  HRV-PPG band; an 8 Hz LP passes more harmonics/noise. Pick one and make the
  docstrings, commit history, and code agree.

- **D2 - ECG R-peak detection [RESOLVED]**: the former detector ran scipy
  `find_peaks` on the raw ECG with a **hardcoded absolute** `height=45000` ADC
  threshold (and a docstring that did not match the code). On the study data
  that threshold sat above every R-peak in one of six sessions, scoring it 0
  beats and zeroing every PPG channel's agreement there. `detect_r_peaks` was
  rewritten as the scale/rate-relative prominence detector described in §4
  (polarity from robust percentiles, `prominence = 0.35*(p99-p40)`, 0.28 s
  refractory). It recovers the previously-dead session, places markers on the
  true peak tips, and matches a reference Pan-Tompkins implementation on beat
  count across the study sessions with equal-or-lower RR-interval variability.
  A full Pan-Tompkins detector was prototyped but not adopted: on these clean
  seated recordings its filtered-domain fiducial refinement pulled markers off
  sharp R-peak tips for no detection benefit. Revisit if the protocol later
  adds motion/exercise segments.

- **D3 - three different PPG detectors**: the webapp default is the Elgendi
  2013 TERMA detector (`detect_ppg_peaks_terma`, selected by
  `PPG_PEAK_DETECTOR="terma"`); the webapp alternative is `detect_ppg_peaks_bp`
  (`distance=0.5*fs`, `prominence=1.1*std`, `width=0.13`); and the standalone
  CLI still uses `detect_ppg_peaks` (`distance=0.4*fs`, `prominence=0.5*std`, no
  width). Results from `python sqi/ccc.py` will not equal the dashboard's. The
  `width=0.13` argument in `detect_ppg_peaks_bp` is in **samples** (scipy
  default units), i.e. effectively no width constraint; confirm that is
  intended. State the active detector (TERMA) explicitly in the manuscript.

- **D4 - CCC vs cleaned intervals**: matched-beat CCC/ICC use **uncleaned**
  PPI/RR while SDNN and frequency metrics use **cleaned** NN. This is a
  deliberate, documented choice but should be stated explicitly in any
  manuscript, since most agreement studies report CCC on artifact-rejected
  intervals.

- **D5 - PPG source channel**: stored sample is the MAX30102 **Red** LED, not
  IR. Confirm against the prior research being matched.
