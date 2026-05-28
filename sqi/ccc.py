"""
ccc.py  --  PPG vs ECG Interval Comparison  |  SEAL Lab
================================================================
Run with no arguments:
    python ccc.py

PIPELINE (all built in — no external scripts needed):
    1. Load raw ECG CSV  (col0 = time ms, col1 = signal)
    2. Load raw PPG CSV  (col0 = time ms, col1 = signal)
    3. Detect R-peaks in ECG  -> compute RR intervals (ms)
    4. Detect systolic peaks in PPG  -> compute PPI intervals (ms)
    5. Match RR and PPI by nearest-neighbour in time
    6. Compute CCC, Pearson r, Bland-Altman, RMSE, MAE
    7. Plot identity + Bland-Altman (optional)

WHY INTERVALS, NOT RAW WAVEFORMS:
    ECG and PPG have completely different waveform shapes (sharp QRS
    spike vs smooth hump), so sample-by-sample comparison is meaningless.
    The shared physiological quantity is the *timing* between beats,
    expressed as RR (from ECG) and PPI (from PPG). CCC on these intervals
    is the standard validation method in the literature.

CCC FORMULA (Lin, 1989):
    rho_c = (2 * cov_xy) / (var_x + var_y + (mu_x - mu_y)^2)

    Unlike Pearson r, CCC penalises systematic offset (bias) between
    PPG and ECG — critical because PPG-derived intervals tend to
    underestimate ECG-derived ones.

    Interpretation:
      > 0.99  -> Almost perfect
      0.95-0.99 -> Substantial
      0.90-0.95 -> Moderate
      < 0.90  -> Poor
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt


# ── Filtering ────────────────────────────────────────────────────────────────

def bandpass(signal, fs, low=0.5, high=40.0, order=2):
    """Butterworth bandpass to remove baseline wander and high-freq noise."""
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)


def lowpass(signal, fs, cutoff=8.0, order=2):
    """Butterworth lowpass for PPG (removes motion noise above 8 Hz)."""
    nyq = fs / 2.0
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, signal)


# ── Peak detection ────────────────────────────────────────────────────────────

def detect_r_peaks(ecg, fs):
    """
    Detect R-peaks in a bandpass-filtered ECG signal.

    Strategy:
      - Bandpass filter (0.5-40 Hz) to remove baseline wander
      - Auto-flip if the lead is inverted (|min| > max after centering)
      - find_peaks with minimum distance = 0.4s (caps at ~150 BPM)
        and height threshold derived from the 90th-percentile amplitude

    Parameters
    ----------
    ecg : np.ndarray  -- raw ECG signal
    fs  : float       -- sampling frequency (Hz)

    Returns
    -------
    peaks : np.ndarray  -- sample indices of R-peaks
    """
    #filtered = bandpass(ecg, fs, low=0.5, high=40.0)
    filtered = ecg
    # Auto-detect polarity: if the negative excursion dominates, the lead is inverted.
    centered = filtered - np.mean(filtered)
    if np.abs(centered.min()) > centered.max():
        filtered = -filtered
    min_distance  = int(0.22 * fs)         # minimum 300 ms between beats (~200 BPM max)
    height_thresh = np.percentile(filtered, 90) * 1.25   # 50% of 90th percentile
    peaks, _ = find_peaks(filtered, distance=min_distance, height=height_thresh)
    return peaks


def detect_ppg_peaks(ppg, fs):
    """
    Detect systolic peaks in a lowpass-filtered PPG signal.

    Strategy:
      - Lowpass filter (8 Hz) to smooth motion noise
      - find_peaks with minimum distance = 0.4s
        and prominence threshold to reject small oscillations

    Parameters
    ----------
    ppg : np.ndarray  -- raw PPG signal
    fs  : float       -- sampling frequency (Hz)

    Returns
    -------
    peaks : np.ndarray  -- sample indices of systolic peaks
    """
    filtered = lowpass(ppg, fs, cutoff=8.0)
    min_distance = int(0.4 * fs)
    prominence_thresh = np.std(filtered) * 0.5
    peaks, _ = find_peaks(filtered, distance=min_distance, prominence=prominence_thresh)
    return peaks


def peaks_to_intervals(peak_indices, time_ms):
    """
    Convert peak indices to inter-peak intervals in ms, using the recorded
    timestamps (not an inferred sample rate). This stays accurate even if
    the device drops samples or has timing jitter.

    Parameters
    ----------
    peak_indices : np.ndarray  -- sample positions of detected peaks
    time_ms      : np.ndarray  -- per-sample timestamps in milliseconds

    Returns
    -------
    intervals_ms : np.ndarray  -- intervals in ms (length = n_peaks - 1)
    peak_times_s : np.ndarray  -- timestamp of the later peak in each pair (s)
    """
    peak_times_ms = time_ms[peak_indices]
    intervals_ms  = np.diff(peak_times_ms)
    peak_times_s  = peak_times_ms[1:] / 1000.0   # timestamp = end of each interval
    return intervals_ms, peak_times_s


# ── Interval matching ─────────────────────────────────────────────────────────

def match_intervals(rr_ms, rr_times, ppi_ms, ppi_times):
    """
    Match RR (ECG) and PPI (PPG) intervals one-to-one by nearest timestamp.

    Because of Pulse Transit Time (~200-300 ms), PPG peaks are delayed
    relative to ECG R-peaks. Nearest-neighbour matching handles this
    without requiring a fixed PTT assumption.

    The acceptance window is set to half the median RR (capped at 500 ms)
    so that at high heart rates we can't accidentally pair across two
    beats. Each PPI may only be claimed once.

    Parameters
    ----------
    rr_ms    : RR intervals in ms
    rr_times : timestamps of RR intervals (seconds)
    ppi_ms   : PPI intervals in ms
    ppi_times: timestamps of PPI intervals (seconds)

    Returns
    -------
    matched_rr, matched_ppi : np.ndarrays of matched pairs
    """
    if len(rr_times) == 0 or len(ppi_times) == 0:
        return np.array([]), np.array([])

    median_rr_s   = float(np.median(rr_ms)) / 1000.0
    tol_s         = min(0.5, max(0.15, median_rr_s / 2.0))
    used          = np.zeros(len(ppi_times), dtype=bool)
    matched_rr, matched_ppi = [], []
    for i, t in enumerate(rr_times):
        # Among unused PPI timestamps, pick the closest within the tolerance.
        candidates = np.where(~used)[0]
        if candidates.size == 0:
            break
        j = candidates[np.argmin(np.abs(ppi_times[candidates] - t))]
        if np.abs(ppi_times[j] - t) <= tol_s:
            matched_rr.append(rr_ms[i])
            matched_ppi.append(ppi_ms[j])
            used[j] = True
    return np.array(matched_rr), np.array(matched_ppi)


# ── CCC and statistics ────────────────────────────────────────────────────────

def compute_ccc(x, y):
    """
    Compute Lin's CCC and companion statistics.

    x = PPG-derived PPI (ms)
    y = ECG-derived RR  (ms)  [gold standard]
    """
    if len(x) != len(y):
        raise ValueError(f"Matched arrays differ in length ({len(x)} vs {len(y)}).")
    if len(x) < 2:
        raise ValueError("Need at least 2 matched pairs.")

    mu_x, mu_y    = np.mean(x), np.mean(y)
    var_x, var_y  = np.var(x, ddof=0), np.var(y, ddof=0)
    cov_xy        = np.mean((x - mu_x) * (y - mu_y))

    denom = var_x + var_y + (mu_x - mu_y) ** 2
    ccc   = (2.0 * cov_xy) / denom if denom != 0 else np.nan

    denom_r   = np.sqrt(var_x * var_y)
    pearson_r = cov_xy / denom_r if denom_r != 0 else np.nan

    diff      = x - y                     # PPG - ECG
    bias      = np.mean(diff)
    std_diff  = np.std(diff, ddof=1)
    loa_upper = bias + 1.96 * std_diff
    loa_lower = bias - 1.96 * std_diff
    rmse      = np.sqrt(np.mean(diff ** 2))
    mae       = np.mean(np.abs(diff))

    return dict(n=len(x), ccc=ccc, pearson_r=pearson_r,
                mean_ppg=mu_x, mean_ecg=mu_y,
                bias=bias, std_diff=std_diff,
                loa_upper=loa_upper, loa_lower=loa_lower,
                rmse=rmse, mae=mae)


def ccc_label(val):
    if np.isnan(val): return "Undefined"
    if val > 0.99:    return "Almost perfect"
    if val > 0.95:    return "Substantial"
    if val > 0.90:    return "Moderate"
    return "Poor"


# ── I/O helpers ───────────────────────────────────────────────────────────────

def prompt(msg, default=None):
    suffix = f"  [{default}]" if default is not None else ""
    raw = input(f"{msg}{suffix}: ").strip()
    return raw if raw != "" else (default if default is not None else "")


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_raw_csv(csv_path):
    """
    Load a headerless CSV with col0=time(ms), col1=signal.
    Returns time_ms (np.ndarray) and signal (np.ndarray).
    """
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(_SCRIPT_DIR, csv_path)
    if not os.path.isfile(csv_path):
        sys.exit(f"[ERROR] File not found: {csv_path}\n        (script dir: {_SCRIPT_DIR})")
    df = pd.read_csv(csv_path, header=None)
    print(f"  -> Loaded '{csv_path}'  |  {df.shape[0]} rows, {df.shape[1]} cols")
    time_ms = pd.to_numeric(df.iloc[:, 0], errors='coerce').to_numpy(dtype=float)
    signal  = pd.to_numeric(df.iloc[:, 1], errors='coerce').to_numpy(dtype=float)
    # Drop rows where either column failed to parse (would otherwise poison fs / find_peaks)
    valid = ~(np.isnan(time_ms) | np.isnan(signal))
    n_dropped = int((~valid).sum())
    if n_dropped:
        print(f"  -> Dropped {n_dropped} non-numeric rows")
    time_ms, signal = time_ms[valid], signal[valid]
    if len(time_ms) < 2:
        sys.exit(f"[ERROR] {csv_path} has fewer than 2 valid rows after parsing.")
    # Infer fs from median sample interval (used only as a fallback for filter design)
    dt_ms = np.median(np.diff(time_ms))
    if dt_ms <= 0:
        sys.exit(f"[ERROR] Non-monotonic timestamps in {csv_path} (median dt = {dt_ms} ms).")
    fs = 1000.0 / dt_ms
    print(f"  -> Inferred fs: {fs:.1f} Hz  |  Signal length: {len(signal)} samples\n")
    return time_ms, signal, fs


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_results(ecg_sig, ecg_t_ms, ecg_peaks,
                 ppg_sig, ppg_t_ms, ppg_peaks,
                 matched_rr, matched_ppi, stats):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib not installed -- skipping plot.")
        return

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("PPG vs ECG Comparison  |  SEAL Lab", fontsize=13)

    # ── Row 1: raw signals with detected peaks ────────────────────────────────
    ax_ecg = fig.add_subplot(3, 2, 1)
    ax_ecg.plot(ecg_t_ms / 1000, ecg_sig, color='steelblue', lw=0.6)
    ax_ecg.scatter(ecg_t_ms[ecg_peaks] / 1000, ecg_sig[ecg_peaks],
                   color='red', s=20, zorder=5, label=f'R-peaks ({len(ecg_peaks)})')
    ax_ecg.set_title("ECG Signal — R-peak Detection")
    ax_ecg.set_xlabel("Time (s)")
    ax_ecg.set_ylabel("Amplitude")
    ax_ecg.legend(fontsize=8)
    ax_ecg.grid(True, alpha=0.3)

    ax_ppg = fig.add_subplot(3, 2, 2)
    ax_ppg.plot(ppg_t_ms / 1000, ppg_sig, color='darkorange', lw=0.6)
    ax_ppg.scatter(ppg_t_ms[ppg_peaks] / 1000, ppg_sig[ppg_peaks],
                   color='red', s=20, zorder=5, label=f'Systolic peaks ({len(ppg_peaks)})')
    ax_ppg.set_title("PPG Signal — Systolic Peak Detection")
    ax_ppg.set_xlabel("Time (s)")
    ax_ppg.set_ylabel("Amplitude")
    ax_ppg.legend(fontsize=8)
    ax_ppg.grid(True, alpha=0.3)

    # ── Row 2: interval time series ───────────────────────────────────────────
    ax_iv = fig.add_subplot(3, 1, 2)
    ax_iv.plot(matched_rr,  color='steelblue',  lw=1.2, marker='o', ms=3, label='RR (ECG)')
    ax_iv.plot(matched_ppi, color='darkorange', lw=1.2, marker='s', ms=3, label='PPI (PPG)')
    ax_iv.set_title("Matched RR vs PPI Intervals per Beat")
    ax_iv.set_xlabel("Beat index")
    ax_iv.set_ylabel("Interval (ms)")
    ax_iv.legend(fontsize=9)
    ax_iv.grid(True, alpha=0.3)

    # ── Row 3: identity plot + Bland-Altman ───────────────────────────────────
    ax_id = fig.add_subplot(3, 2, 5)
    all_v = np.concatenate([matched_ppi, matched_rr])
    lim   = [np.min(all_v) * 0.97, np.max(all_v) * 1.03]
    ax_id.scatter(matched_rr, matched_ppi, color='steelblue', alpha=0.65,
                  edgecolors='white', lw=0.4, s=45)
    ax_id.plot(lim, lim, 'k--', lw=1.2, label='y = x (perfect agreement)')
    ax_id.set_xlim(lim); ax_id.set_ylim(lim)
    ax_id.set_xlabel("ECG RR interval (ms)")
    ax_id.set_ylabel("PPG PPI interval (ms)")
    ax_id.set_title(
        f"Identity Plot\nCCC = {stats['ccc']:.4f}  ({ccc_label(stats['ccc'])})  "
        f"|  r = {stats['pearson_r']:.4f}")
    ax_id.legend(fontsize=8)
    ax_id.grid(True, alpha=0.3)
    ax_id.set_aspect('equal', adjustable='box')

    ax_ba = fig.add_subplot(3, 2, 6)
    means = (matched_ppi + matched_rr) / 2.0
    diffs = matched_ppi - matched_rr
    ax_ba.scatter(means, diffs, color='darkorange', alpha=0.65,
                  edgecolors='white', lw=0.4, s=45)
    ax_ba.axhline(stats['bias'],      color='gray', ls='--', lw=1.3,
                  label=f"Bias = {stats['bias']:+.2f} ms")
    ax_ba.axhline(stats['loa_upper'], color='red',  ls=':',  lw=1.3,
                  label=f"Upper LOA = {stats['loa_upper']:+.2f} ms")
    ax_ba.axhline(stats['loa_lower'], color='red',  ls=':',  lw=1.3,
                  label=f"Lower LOA = {stats['loa_lower']:+.2f} ms")
    ax_ba.fill_between([np.min(means), np.max(means)],
                       stats['loa_lower'], stats['loa_upper'],
                       alpha=0.08, color='red')
    ax_ba.set_xlabel("Mean of PPI and RR (ms)")
    ax_ba.set_ylabel("Difference  PPG - ECG (ms)")
    ax_ba.set_title("Bland-Altman Plot")
    ax_ba.legend(fontsize=8)
    ax_ba.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  CCC Compare  --  PPG vs ECG  |  SEAL Lab")
    print("=" * 62)
    print()
    print("Input: raw signal CSVs with no headers.")
    print("  col 0 = time (ms)")
    print("  col 1 = signal (ECG in mV / PPG in ADC or mV)")
    print()
    print("The script will:")
    print("  1. Detect R-peaks in ECG  -> RR intervals")
    print("  2. Detect systolic peaks in PPG  -> PPI intervals")
    print("  3. Match intervals by time")
    print("  4. Compute CCC, Bland-Altman, RMSE, MAE")
    print()

    # ECG
    print("-- ECG File (Gold Standard) ----------------------------------")
    ecg_path = prompt("ECG CSV path", default="sample_ecg.csv")
    ecg_t, ecg_sig, ecg_fs = load_raw_csv(ecg_path)

    # PPG
    print("-- PPG File --------------------------------------------------")
    ppg_path = prompt("PPG CSV path", default="sample_ppg.csv")
    ppg_t, ppg_sig, ppg_fs = load_raw_csv(ppg_path)

    # Trim both signals to overlapping time window
    t_start = max(ecg_t[0],  ppg_t[0])
    t_end   = min(ecg_t[-1], ppg_t[-1])
    ecg_mask = (ecg_t >= t_start) & (ecg_t <= t_end)
    ppg_mask = (ppg_t >= t_start) & (ppg_t <= t_end)
    ecg_t, ecg_sig = ecg_t[ecg_mask], ecg_sig[ecg_mask]
    ppg_t, ppg_sig = ppg_t[ppg_mask], ppg_sig[ppg_mask]
    print(f"  Overlapping window     : {t_start:.0f} – {t_end:.0f} ms  "
          f"({(t_end - t_start) / 1000:.1f} s)")
    print(f"  ECG samples in window  : {len(ecg_sig)}")
    print(f"  PPG samples in window  : {len(ppg_sig)}")
    print()

    # Options
    print("-- Options ---------------------------------------------------")
    do_plot = prompt("Show plots? (y/n)", default="y").lower().startswith("y")
    out_raw = prompt("Save results CSV? Enter path or leave blank", default="").strip()
    print()

    # Peak detection
    print("-- Peak Detection --------------------------------------------")
    ecg_peaks = detect_r_peaks(ecg_sig, ecg_fs)
    ppg_peaks = detect_ppg_peaks(ppg_sig, ppg_fs)
    print(f"  ECG R-peaks detected   : {len(ecg_peaks)}")
    print(f"  PPG systolic peaks     : {len(ppg_peaks)}")

    # Intervals (use recorded timestamps so jitter / dropped samples don't bias us)
    rr_ms,  rr_times  = peaks_to_intervals(ecg_peaks, ecg_t)
    ppi_ms, ppi_times = peaks_to_intervals(ppg_peaks, ppg_t)
    print(f"  RR intervals           : {len(rr_ms)}  "
          f"(mean {rr_ms.mean():.1f} ms, std {rr_ms.std():.1f} ms)")
    print(f"  PPI intervals          : {len(ppi_ms)}  "
          f"(mean {ppi_ms.mean():.1f} ms, std {ppi_ms.std():.1f} ms)")

    # Matching
    matched_rr, matched_ppi = match_intervals(rr_ms, rr_times, ppi_ms, ppi_times)
    print(f"  Matched pairs          : {len(matched_rr)}")
    print()

    if len(matched_rr) < 2:
        sys.exit("[ERROR] Too few matched pairs. Check peak detection or signal quality.")

    # CCC
    stats = compute_ccc(matched_ppi, matched_rr)

    # Results
    print("=" * 62)
    print("  Results")
    print("=" * 62)
    print(f"  Matched beats            : {stats['n']}")
    print(f"  Mean PPG PPI             : {stats['mean_ppg']:.2f} ms")
    print(f"  Mean ECG RR  (ref)       : {stats['mean_ecg']:.2f} ms")
    print()
    print(f"  -- Agreement -----------------------------------------")
    print(f"  Lin's CCC (rho_c)        : {stats['ccc']:.4f}  [{ccc_label(stats['ccc'])}]")
    print(f"  Pearson r                : {stats['pearson_r']:.4f}")
    print()
    print(f"  -- Bland-Altman --------------------------------------")
    print(f"  Mean bias (PPG - ECG)    : {stats['bias']:+.2f} ms")
    print(f"  Std of differences       : {stats['std_diff']:.2f} ms")
    print(f"  Upper LOA (+1.96 std)    : {stats['loa_upper']:+.2f} ms")
    print(f"  Lower LOA (-1.96 std)    : {stats['loa_lower']:+.2f} ms")
    print()
    print(f"  -- Error Metrics -------------------------------------")
    print(f"  RMSE                     : {stats['rmse']:.2f} ms")
    print(f"  MAE                      : {stats['mae']:.2f} ms")
    print("=" * 62)
    print()

    # Save  (resolve relative paths against the script dir, like load_raw_csv does)
    if out_raw:
        if not os.path.isabs(out_raw):
            out_raw = os.path.join(_SCRIPT_DIR, out_raw)
        pd.DataFrame({
            "ecg_rr_ms":    matched_rr,
            "ppg_ppi_ms":   matched_ppi,
            "difference_ms": matched_ppi - matched_rr,
            "mean_ms":       (matched_ppi + matched_rr) / 2.0,
        }).to_csv(out_raw, index=False)
        print(f"[INFO] Results saved to: {out_raw}")

    # Plot
    if do_plot:
        plot_results(ecg_sig, ecg_t, ecg_peaks,
                     ppg_sig, ppg_t, ppg_peaks,
                     matched_rr, matched_ppi, stats)


if __name__ == "__main__":
    main()