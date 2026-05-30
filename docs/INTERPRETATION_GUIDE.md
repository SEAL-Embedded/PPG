# Interpretation guide — what the metrics mean

This document explains every signal-quality metric the dashboard reports, the literature behind it, and the exact thresholds the dashboard uses to colour-code its plain-English verdict cards. If you just want to use the dashboard, [`DASHBOARD_GUIDE.md`](DASHBOARD_GUIDE.md) is shorter. If you want to inspect the raw schema, [`DATA_FORMATS.md`](DATA_FORMATS.md). If you want to call the API directly, [`API_REFERENCE.md`](API_REFERENCE.md).

## Table of contents

1. [Why intervals, not raw waveforms](#why-intervals-not-raw-waveforms)
2. [SSQI — skewness](#ssqi--skewness)
3. [ZSQI — zero-crossing rate](#zsqi--zero-crossing-rate)
4. [CCC — Lin's concordance correlation coefficient](#ccc--lins-concordance-correlation-coefficient)
5. [ICC — intraclass correlation, ICC(A,1)](#icc--intraclass-correlation-icca1)
6. [Bland-Altman — bias, LOA, RMSE, MAE](#bland-altman--bias-loa-rmse-mae)
7. [Channel verdict roll-up](#channel-verdict-roll-up)
8. [Site verdict roll-up](#site-verdict-roll-up)
9. [Known code-vs-manuscript gaps](#known-code-vs-manuscript-gaps)
10. [Per-HRV-feature CCC (Σ page)](#per-hrv-feature-ccc-σ-page)
11. [Ectopic-beat / NN-interval cleaning](#ectopic-beat--nn-interval-cleaning)
12. [ticks_us rollover unwrap](#ticks_us-rollover-unwrap)

---

## Why intervals, not raw waveforms

ECG and PPG have completely different waveform shapes — a sharp QRS spike on ECG vs a smooth perfusion pulse on PPG. Sample-by-sample comparison is meaningless. The shared physiological quantity is the **timing between beats**: RR (R-peak to R-peak on ECG) and PPI (systolic peak to systolic peak on PPG). CCC, ICC, Bland-Altman, RMSE, and MAE in the dashboard are all computed on **matched intervals**, not on raw samples.

`sqi/bland_altman.py` does sample-by-sample raw-waveform Bland-Altman; it is intentionally not wired into the dashboard. The matched-interval Bland-Altman is inside `sqi/ccc.py:compute_ccc` and is what every Bland-Altman card in the UI renders.

---

## SSQI — skewness

**What it is.** The third statistical moment of the raw PPG signal, computed in `sqi/SSQI_algorithm.Ssqi`:

$$\mathrm{SSQI} = \mathbb{E}\!\left[\left(\frac{x-\mu}{\sigma}\right)^{3}\right]$$

**Why it works for PPG.** A clean, well-perfused PPG has a sharp systolic upstroke followed by a slower diastolic decline — a right-leaning shape that produces *positive* skew. Krishnan et al. (2010) showed SSQI ≥ ~1 corresponds to clean cardiac PPG; values near zero or negative mean the waveform's shape is washed out by noise, motion, or the optical lead is inverted.

**Dashboard thresholds (from `_grade_ssqi_text`):**

| SSQI value     | Grade label    | Meaning                                          |
|----------------|----------------|--------------------------------------------------|
| > 1.5          | **very good**  | Strong positive skew, classic well-shaped PPG    |
| > 0.5          | **good**       | Positive skew, pulse shape is recognisable       |
| -0.5 → 0.5     | **borderline** | Near-zero skew, waveform shape is weak or noisy  |
| < -0.5         | **bad**        | Negative skew — signal likely inverted, saturated, or noise-dominated |
| not finite     | **undefined**  | Signal too short or constant                     |

---

## ZSQI — zero-crossing rate

**What it is.** The fraction of consecutive samples whose sign changes after mean-subtraction, computed per window in `sqi/zcr_sqi.compute_zcr` and `windowed_zcr`. The dashboard reports the mean (μ) and standard deviation (σ) across 5-s sliding windows.

$$\mathrm{ZSQI}_{\text{win}} = \frac{|\{i : \operatorname{sign}(x_i - \bar{x}) \ne \operatorname{sign}(x_{i-1} - \bar{x})\}|}{N - 1}$$

**Why it works for PPG.** A clean cardiac-band PPG crosses zero roughly twice per beat — for a typical 60–90 bpm resting heart rate, that's ZSQI ≈ 0.02–0.05. Much higher means high-frequency noise is dominating the signal; high *variance* across windows means contact is intermittent (jostling, motion artefact).

**Dashboard thresholds (from `_grade_zsqi_text`):**

| ZSQI μ              | σ          | Grade label    | Meaning |
|---------------------|------------|----------------|---------|
| < 0.06              | < 0.02     | **very good**  | Low, stable rate — sensor contact looks consistent |
| < 0.10              | any        | **good**       | Within the typical clean-PPG range |
| < 0.20              | any        | **borderline** | Elevated — noisy or loose contact |
| ≥ 0.20              | any        | **bad**        | Very high — dominated by noise / motion artifact |

---

## CCC — Lin's concordance correlation coefficient

**What it is.** Lin (1989):

$$\rho_c = \frac{2 \, \mathrm{cov}(x,y)}{\sigma_x^2 + \sigma_y^2 + (\mu_x - \mu_y)^2}$$

with $x$ = PPG-derived PPI and $y$ = ECG-derived RR (gold standard). Computed in `sqi/ccc.py:compute_ccc`.

**Why it's better than Pearson r for validation.** Pearson is invariant to additive offset and scale, so two devices that disagree by a constant bias can still score r = 1. CCC penalises systematic offset (via the $(\mu_x - \mu_y)^2$ term in the denominator) — exactly what we need when validating PPG against ECG, because PPG-derived intervals tend to be biased by the pulse transit time.

**Dashboard thresholds** (`sqi/ccc.py:ccc_label`, mirrored in `_grade_ccc_text`):

| CCC value      | Label              | Source       |
|----------------|--------------------|--------------|
| > 0.99         | Almost perfect     | Lin 1989     |
| 0.95 – 0.99    | Substantial        | Lin 1989     |
| 0.90 – 0.95    | Moderate           | Lin 1989     |
| 0.50 – 0.90    | Poor               | Lin 1989: <0.90 |
| ≤ 0.50         | Very poor          | interval timing does not track ECG |

In the SQI table CCC cells are colour-coded: green ≥ 0.95, amber 0.90–0.95, red below.

---

## ICC — intraclass correlation, ICC(A,1)

**What it is.** Intraclass correlation, computed via `pingouin.intraclass_corr` in `webapp/analysis.py:_safe_icc`. The dashboard reports the **ICC(A,1)** form — two-way mixed, single rater, **absolute agreement** (Shrout & Fleiss 1979 notation; equivalent to ICC3 in the older McGraw–Wong table when restricted to absolute agreement).

**Why ICC(A,1) for PPG vs ECG.** The two raters (ECG, PPG) are fixed instruments, not random draws (mixed-effects). Single rater because each beat is rated once by each device. Absolute agreement (not consistency) because PTT bias is real and we want it captured — same reason `compute_ccc` includes the $(\mu_x - \mu_y)^2$ term.

For paired interval data ICC and CCC are mathematically nearly identical — the dashboard reports both because they're separately published in different validation literatures, and the small numerical gap can be diagnostic when one statistic suggests a subtle issue (e.g. variance heterogeneity) that the other doesn't surface.

**Dashboard thresholds.** Reused from CCC for visual consistency; pingouin's own CI is preserved in `analysis.json` (`stats.icc_ci_low`, `stats.icc_ci_high`) for reporting.

**Minimum sample size.** `_safe_icc` requires ≥ 4 matched beats and returns `None` otherwise (the column shows `—`). Without `pingouin` installed every ICC cell is `—`.

---

## Bland-Altman — bias, LOA, RMSE, MAE

**What it is.** For each matched (RR, PPI) pair, compute the difference $d_i = \mathrm{PPI}_i - \mathrm{RR}_i$ and the mean $m_i = (\mathrm{PPI}_i + \mathrm{RR}_i)/2$. Report:

- **Bias** = $\mathbb{E}[d]$ — mean difference. PPG peaks lag ECG R-peaks by the pulse transit time, so a small positive bias is expected (~50–300 ms is normal for forehead / finger).
- **LOA (limits of agreement)** = $\mathrm{bias} \pm 1.96 \cdot \mathrm{SD}(d)$ — 95% of the differences fall within this band if they're approximately normal.
- **RMSE** = $\sqrt{\mathbb{E}[d^2]}$
- **MAE** = $\mathbb{E}[|d|]$

Each PPG channel gets its own Bland-Altman card showing $d$ vs $m$ with the bias and LOA± lines overlaid.

**Bias interpretation (from `_grade_bias_text`):**

| \|bias\| (ms) | Label        | Meaning |
|---------------|--------------|---------|
| < 20          | very small   | Tracks ECG with no meaningful offset |
| 20 – 100      | small        | Within typical PTT range |
| 100 – 500     | moderate     | Bigger than PTT alone; check for missed/double beats |
| ≥ 500         | huge         | Orders of magnitude larger than PTT — peak detector likely matched the wrong feature (respiration baseline, harmonic) |

A huge bias is the most common failure mode in the SEAL dataset: the PPG peak detector's default thresholds don't always reject low-frequency baseline wander, and the resulting "intervals" are several seconds long.

---

## Channel verdict roll-up

`webapp/analysis.py:interpret_channel` combines the metrics into a single grade per channel:

| Condition (in order) | Grade   | Verdict head |
|----------------------|---------|--------------|
| `matched == 0`       | **bad** | "no usable beats" |
| `ccc > 0.95` AND SSQI good/very-good | **good** | "substantial agreement, ECG-grade signal" |
| `ccc > 0.90`         | **ok**  | "moderate agreement, usable with caveats" |
| `ccc > 0.50`         | **warn**| "poor agreement, inspect before using" |
| anything else        | **bad** | "no agreement" |

A second pass downgrades any good/ok channel with `matched < 30` to **warn** ("Small matched-beat count — re-run on a longer window before quoting numbers"). This catches the case where the pipeline produced a nominally high CCC from a handful of cherry-picked beats.

---

## Site verdict roll-up

`webapp/analysis.py:_site_summary` produces the per-site verdict shown in the batch view. The mean CCC across every channel from every session assigned to that site drives the grade:

| mean CCC across site | Grade |
|----------------------|-------|
| > 0.95               | **good**  |
| > 0.90               | **ok**    |
| > 0.50               | **warn**  |
| ≤ 0.50               | **bad**   |

Additional flags surfaced in the verdict text:
- SSQI mean across the site (good shape / borderline / poor-or-inverted)
- A "large mean bias" annotation when the mean of the per-channel biases exceeds 500 ms

---

## Known code-vs-manuscript gaps

Surface these whenever you quote numbers from the dashboard — the methods section of the manuscript-in-preparation specifies things the current code does not yet do. Tracked in `memory/project_code_paper_gaps.md`.

1. **Sample rate.** Paper claims 750 Hz. Firmware (`fullpipico.py`, `pipico_code/ppgcode/main.py`) calls `set_sample_rate(3200)` with `set_fifo_average(8)`, giving ≈ 400 Hz per channel — further reduced by round-robin servicing across active lanes. The dashboard reports the *actually inferred* fs (typically ~130 Hz per active lane) in the SQI table and in `_grade_ccc_text` notes when fs < 250 Hz.
2. **ECG R-peak detector.** Paper specifies Pan–Tompkins. `sqi/ccc.py:detect_r_peaks` uses scipy `find_peaks` on a 0.5–40 Hz Butterworth with a 90th-percentile height threshold and auto-flip — *not* Pan–Tompkins. Numbers differ on noisy beats.
3. **HRV frequency domain.** Paper specifies Lomb–Scargle for LF/HF. `webapp/analysis.py` now computes LF/HF per channel via `pyhrv.welch_psd` (Welch on cubic-spline-resampled NN) so the per-session HR/SDNN/LF-HF agreement tables match the canonical `old/PPGanalysis.py` pipeline; the legacy `signal_visualization/ppgvis.py` has been deprecated and replaced with an import-raising stub.
4. **PPG bandpass.** Paper specifies 0.5–4.0 Hz fourth-order zero-phase Butterworth. `sqi/ccc.py:ppg_bandpass(low=0.5, high=4.0, order=4)` is the canonical paper-spec filter and is used by the peak-detection path (`detect_ppg_peaks_bp` inside `webapp/analysis.py`); the dashboard's display-overlay `webapp/analysis.py:ppg_bandpass` delegates to it so the same waveform feeds detection and display.
5. **Per-site Bland–Altman across the cohort.** Now wired by the dashboard's batch view (per-site aggregate table), but the FST × site cross-tab the manuscript promises is blocked until participant.json carries a Fitzpatrick grade for every session. The batch view's meta-grid shows `FST strata: unavailable` until at least one session has FST saved.

---

## Ectopic-beat / NN-interval cleaning

All HRV features (SDNN, RMSSD, LF/HF, SampEn, Poincaré) are now computed
on cleaned NN intervals, not raw RR/PPI intervals. Cleaning happens in
`sqi/hrv_clean.py:clean_intervals` via a two-pass filter:

1. **Physiological range gate** — drop intervals outside [300, 2000] ms
   (200 BPM .. 30 BPM).
2. **Karlsson 1987 local-median rule** — drop intervals differing by
   more than 20% from a 5-beat rolling median.

Reference: Karlsson, M. et al. (1987). *Computers and Biomedical Research*
20(4), 333-340.

Without this step, one missed beat or one ectopic complex inflates SDNN
by a factor of 5-10 and contaminates every downstream HRV feature. The
legacy `signal_visualization/ppgvis.py` had ad-hoc median-ratio filters
that were dropped in the dashboard rewrite — this module restores that
defensive layer with a clean, tested implementation.

---

## ticks_us rollover unwrap

MicroPython on the RP2040 uses a 30-bit `ticks_us()` counter that wraps
at 2^30 µs (~17.89 minutes). The receiver writes raw values to CSV; any
continuous recording longer than the wrap period would have shown a
~1.07-billion-µs backward step at the boundary, poisoning every
downstream metric.

`webapp/analysis.py:_unwrap_ticks_us` now detects backward steps larger
than 2^29 µs and adds the wrap period cumulatively, restoring monotone
timestamps. Old CSVs that crossed the wrap before this fix landed will
be unwrapped automatically on next load.
