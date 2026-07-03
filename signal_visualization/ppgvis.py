# =============================================================================
# PPG Signal Processing & Visualization
# Loads raw PPG data, corrects timestamps, filters the signal, detects peaks,
# computes HRV metrics (time & frequency domain), SNR, and plots results.
# =============================================================================


# =============================================================================
# === Imports ===
# =============================================================================
import matplotlib.pyplot as plt
import csv
import os
import numpy as np
from scipy.signal import hilbert, savgol_filter, find_peaks, welch, detrend, butter, filtfilt
import pandas as pd
from scipy.integrate import simpson
from scipy.interpolate import interp1d
import pyhrv
import pyhrv.frequency_domain as fd


# =============================================================================
# === Configuration ===
# =============================================================================

# Path to the raw PPG CSV file (column 0 = timestamp µs, column 1 = IR value)
#file_path = os.path.join("PPG", "day1test2530.csv")
file_path = "MDPIdata\\session_20260530_191052\\ppg_data_ch0.csv"

# Nominal sampling interval and rate (used for timestamp correction and resampling)
time_interval = 1/400       # seconds per sample (400 Hz)
time_interval_us = (int)(time_interval * 1e6)   # converted to microseconds
actual_sampling_rate = 1/time_interval           # Hz


# =============================================================================
# === Data Loading ===
# Reads raw timestamps (µs) and IR PPG values from CSV.
# =============================================================================

with open(file_path, 'r') as file:
    reader = csv.reader(file)

    rawtimestamps = []
    ir_data = []
    for row in reader:
        # Skip rows with missing IR data
        if len(row) >= 2 and row[1].strip() != '':
            rawtimestamps.append(float(row[0]))    # Column 0 = timestamp (µs)
            ir_data.append(float(row[1]))          # Column 1 = PPG IR signal

# Build an evenly-spaced time axis based on nominal sample rate (not corrected timestamps)
x_values = [i * time_interval for i in range(len(ir_data))]


# =============================================================================
# === Timestamp Correction ===
# Raw hardware timestamps can have large gaps due to dropped samples or clock
# jitter. This pass detects jumps > 10ms and shifts all subsequent timestamps
# back by the excess, producing a continuous timeline at the nominal rate.
# =============================================================================

corrected_timestamps = np.array(rawtimestamps, dtype=np.int64)
print(len(corrected_timestamps))

for i in range(1, len(corrected_timestamps)):
    time_diff = corrected_timestamps[i] - corrected_timestamps[i-1]

    if abs(time_diff) > 10000:  # Threshold: 10 ms gap indicates a timing anomaly
        print(f"\n--- Detected jump at index {i} ---")
        print(f"  Previous timestamp: {corrected_timestamps[i-1]} us")
        print(f"  Observed difference: {time_diff} us")

        # The excess time is how much longer this gap was than expected.
        # Subtracting it from all subsequent timestamps realigns the timeline.
        excess_time_us = time_diff - time_interval_us

        print(f"  Expected period: {time_interval_us} us")
        print(f"  Excess time to correct: {excess_time_us} us")

        corrected_timestamps[i:] -= excess_time_us
        print(f"  Timestamp at index {i} corrected to: {corrected_timestamps[i]} us")
        print(f"  New difference from previous: {corrected_timestamps[i] - corrected_timestamps[i-1]} us")
        print(f"  Remaining timestamps adjusted by {excess_time_us} us.")
        print("---------------------------------")


# =============================================================================
# === Resampling / Interpolation ===
# Converts corrected µs timestamps to seconds, zero-offsets them, then
# resamples the signal onto a uniform time grid using cubic interpolation.
# =============================================================================

timestamps_seconds = np.array(corrected_timestamps) / 1e6  # µs → seconds
first_time = timestamps_seconds[0]
timestamps = timestamps_seconds - first_time    # Shift so recording starts at t=0
first_time = timestamps[0]

print("useless")
print("start time:", first_time)

end_time = timestamps[-1]
num_samples = int(np.ceil((end_time - first_time) * actual_sampling_rate))
uniform_timestamps = np.linspace(first_time, end_time, num_samples)

# Cubic interpolation to fill gaps and regularize the sample grid
interpolator = interp1d(timestamps, ir_data, kind='cubic', fill_value='extrapolate')
resampled_signal = interpolator(uniform_timestamps)

print("new time")
print(len(uniform_timestamps))


# =============================================================================
# === Time Range Configuration ===
# Loads per-file start/end times from a metadata CSV so only the relevant
# portion of each recording is analyzed. Falls back to a 'default' entry.
# =============================================================================

# Default fallback range (overridden by metadata CSV below)
start_time = 0    # seconds
end_time = 900    # seconds

# Load per-file time ranges from metadata
custom_time_ranges = {}
with open("old\\Metadata\\PPG_time_ranges.csv", mode='r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        time_ranges_filename = row['filename']
        start = float(row['start'])
        end = float(row['end'])
        custom_time_ranges[time_ranges_filename] = (start, end)

# Match the current file to its metadata entry (strip extension from filename)
time_ranges_filename = os.path.splitext(os.path.basename(file_path))[0]
print(time_ranges_filename)
start_time, end_time = custom_time_ranges.get(time_ranges_filename, custom_time_ranges['default'])


# =============================================================================
# === Filter Data to Time Window ===
# Slices the resampled signal to only include samples within [start_time, end_time].
# =============================================================================

filtered_indices = [i for i, t in enumerate(uniform_timestamps) if start_time <= t <= end_time]
x_filtered = [uniform_timestamps[i] for i in filtered_indices]
ir_filtered = [resampled_signal[i] for i in filtered_indices]


# =============================================================================
# === Outlier Removal ===
# Identifies samples that deviate significantly from a rolling median and
# replaces them with linearly interpolated values from neighboring good samples.
# =============================================================================

y = np.array(ir_filtered)

# Rolling median as a smooth reference baseline
y_median = pd.Series(y).rolling(window=15, center=True, min_periods=1).median()
gap = abs(y_median - y)

# Flag samples that are more than 4.25 std deviations from the rolling median
threshold = 4.25 * np.std(gap.dropna())
outliers = gap > threshold

x_filtered = np.array(x_filtered)
bad_x = x_filtered[outliers]   # Time positions of outlier samples (for plotting)
bad_y = y[outliers]             # Outlier amplitude values (for plotting)

# Replace outliers with linearly interpolated values from surrounding good samples
y_fixed = np.copy(y)
y_fixed[outliers.to_numpy()] = np.interp(
    x_filtered[outliers.to_numpy()],    # x positions of bad points
    x_filtered[~outliers.to_numpy()],   # x positions of good points
    y[~outliers.to_numpy()]             # y values of good points
)


# =============================================================================
# === Smoothing ===
# Simple centered moving average. window_size=0 means no smoothing (pass-through).
# =============================================================================

window_size = 0
smooth = lambda data: [np.mean(data[max(0, i-window_size): min(len(data), i+window_size + 1)]) for i in range(len(data))]

ir_smooth = smooth(y_fixed)
x_smoothed = uniform_timestamps[:len(ir_smooth)]


# =============================================================================
# === Bandpass Filtering ===
# Butterworth bandpass filter isolates the cardiac frequency range (0.6–3.3 Hz),
# removing slow baseline drift and high-frequency noise.
# =============================================================================

def bandpass(data, lowcut=0.5, highcut=10, fs=50, order=2):
    """Apply a zero-phase Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Bandpass limits chosen to span resting heart rate range (36–198 BPM)
hpfiltered = bandpass(ir_smooth, lowcut=0.6, highcut=3.3, fs=actual_sampling_rate)


# =============================================================================
# === Peak Detection ===
# Finds systolic peaks in the filtered PPG signal using prominence and distance
# constraints tuned to plausible heart rate ranges.
# =============================================================================

# distance = 0.4 * fs enforces a minimum of ~150 BPM upper bound between peaks
peaks, props = find_peaks(hpfiltered, prominence=1.1, width=0.13, distance=0.4 * actual_sampling_rate)


# =============================================================================
# === FFT ===
# Single-sided FFT of the bandpass-filtered signal for frequency inspection.
# =============================================================================

y = hpfiltered
fs = actual_sampling_rate

N = len(y)
yf = np.fft.fft(y)
xf = np.fft.fftfreq(N, 1/fs)

# Keep only positive frequencies (single-sided spectrum)
idx = xf >= 0
xf = xf[idx]
yf = np.abs(yf[idx])   # magnitude spectrum


# =============================================================================
# === Heart Rate & HRV Calculation (Initial Pass) ===
# Computes RR intervals from detected peaks, then applies a median-based filter
# to remove implausibly short intervals (likely double-detections).
# =============================================================================

# Convert peak indices to times using the nominal time interval
peak_times = np.array(peaks * time_interval) + start_time
rr_intervals = np.diff(peak_times)
median_rr = np.median(rr_intervals)

# First-pass filter: remove peaks whose RR interval is less than half the median
# (keeps only peaks that are at least half as far apart as typical)
filtered_peaks = [peaks[0]]
last_idx = 0   # index (into peaks/peak_times) of the last kept peak
for i in range(1, len(peak_times)):
    rr = peak_times[i] - peak_times[last_idx]
    if 0.6 * median_rr < rr:
        # Well separated from the last kept peak: a new beat.
        filtered_peaks.append(peaks[i])
        last_idx = i
    elif hpfiltered[peaks[i]] > hpfiltered[peaks[last_idx]]:
        # Too close to be a separate beat — it's a doublet. Keep the
        # taller of the two (the true systolic peak), drop the other.
        filtered_peaks[-1] = peaks[i]
        last_idx = i
    # else: peaks[i] is the shorter of the doublet — discard it.


# =============================================================================
# === Metadata: Remove Bad Ranges ===
# Reads time segments marked as artifactual from a CSV and removes any peaks
# that fall within those segments.
# =============================================================================

df = pd.read_csv("old\\Metadata\\RemoveRanges.csv")
# Get only the removal ranges relevant to this file
ranges = df[df['filename'] == time_ranges_filename][['start_idx', 'end_idx']].values

print(ranges)

filtered_peaks = np.array(filtered_peaks)
filtered_peak_times = np.array(filtered_peaks) * time_interval + start_time
print(filtered_peak_times)

# Build a boolean mask: True = keep the peak (not in any bad range)
keep_mask = np.ones_like(filtered_peak_times, dtype=bool)
for start, end in ranges:
    keep_mask &= ~((filtered_peak_times >= start) & (filtered_peak_times <= end))

# Apply mask to remove peaks inside artifact ranges
ranges_removed_peak_times = filtered_peak_times[keep_mask]
ranges_removed_peaks = ((ranges_removed_peak_times - start_time) / time_interval).astype(int)


# =============================================================================
# === Metadata: Manual Peak Additions ===
# For segments where automatic detection missed peaks, manually specified time
# windows are used to find the local maximum and add it as a peak.
# =============================================================================

final_peaks = []
custom_peaks = []

df = pd.read_csv("old\\Metadata\\Manual_peaks.csv")
file_ranges = df[df['filename'].str.strip() == time_ranges_filename]

for _, row in file_ranges.iterrows():
    start_idx = float(row['startidx'])
    end_idx = float(row['endidx'])
    print("goon")

    # Convert time bounds to signal indices, clamped to valid range
    start_idx = max(0, int(((start_idx - start_time) / time_interval)))
    end_idx = min(len(hpfiltered) - 1, int(((end_idx - start_time) / time_interval)))

    print(start_idx, end_idx)

    if start_idx < end_idx:
        # Find the index of the local maximum within the specified window
        local_max_idx = start_idx + np.argmax(hpfiltered[start_idx:end_idx+1])

        # Only add if not already detected
        if local_max_idx not in ranges_removed_peaks:
            custom_peaks.append(local_max_idx)

print(custom_peaks)
print(final_peaks)


# =============================================================================
# === Final Peak Consolidation & RR Intervals ===
# Merges auto-detected (range-removed) peaks with manually added peaks,
# then applies a final outlier filter on RR intervals before computing
# heart rate and HRV.
# =============================================================================

# Merge and sort all accepted peak indices
final_peaks = np.sort(np.concatenate([ranges_removed_peaks, np.array(custom_peaks, dtype=int)]))
final_peaks = np.array(final_peaks, dtype=int)

print(final_peaks)

# Convert peak indices back to times
final_peak_times = final_peaks * time_interval + start_time
custom_peak_times = np.array(custom_peaks) * time_interval + start_time

# RR intervals from the consolidated peak set
filtered_rr_intervals = np.diff(final_peak_times)
median_rr = np.median(filtered_rr_intervals)

# Second-pass filter: remove intervals longer than 2.5x the median
# (catches missed beats or large artifacts that slipped through range removal)
re_filtered_intervals = [filtered_rr_intervals[0]]
for i in range(1, len(filtered_rr_intervals)):
    if 2.5 * median_rr > filtered_rr_intervals[i]:
        re_filtered_intervals.append(filtered_rr_intervals[i])

# Final biometric summary statistics
heart_rate = 60 / np.mean(re_filtered_intervals)   # BPM
bps = heart_rate / 60                               # beats per second (for SNR bands)
hrv = np.std(re_filtered_intervals)                 # SDNN (standard deviation of NN intervals)

print(re_filtered_intervals)
re_filtered_intervals = np.array(re_filtered_intervals)


# =============================================================================
# === LF/HF Frequency Domain HRV Analysis ===
# Resamples the RR interval series to a uniform 4 Hz grid, then estimates the
# power spectral density via Welch's method. Integrates power in VLF, LF, and
# HF bands to compute the LF/HF ratio — a marker of autonomic balance.
# =============================================================================

sampling_rate_hz = 4        # Resample RR series to 4 Hz for PSD estimation
segment_length_s = 570      # Target analysis window length (seconds)
nfft_points = 1024          # FFT length for Welch PSD

# Build a cumulative time axis from the RR intervals (seconds since first beat)
r_peak_times_s = np.cumsum(re_filtered_intervals)
r_peak_times_s = r_peak_times_s - r_peak_times_s[0]    # Zero-offset

# Warn if the recording is shorter than the desired segment
if r_peak_times_s[-1] < segment_length_s:
    print(f"Warning: Recording is only {r_peak_times_s[-1]:.2f} seconds long, "
          f"which is shorter than the desired segment length of {segment_length_s} seconds.")
    segment_length_s = r_peak_times_s[-1]

# Uniform time grid at the target HRV sampling rate
time_interp = np.arange(0, segment_length_s, 1 / sampling_rate_hz)

# Deduplicate RR timestamps (required for interp1d) and interpolate
unique_r_peak_times, unique_rr_intervals = np.unique(r_peak_times_s, return_index=True)
f_interp = interp1d(unique_r_peak_times, re_filtered_intervals[unique_rr_intervals],
                    kind='linear', fill_value="extrapolate")
interpolated_rr_s = f_interp(time_interp)

# Detrend to remove slow drift before PSD estimation
detrended_signal = detrend(interpolated_rr_s)

# Welch PSD — window size clamped to nearest lower power of 2
nperseg_actual = min(nfft_points, len(detrended_signal))
nperseg_actual = int(2**np.floor(np.log2(nperseg_actual)))
noverlap_actual = 0     # No overlap (single-segment analysis)

fafrequencies, psd = welch(detrended_signal,
                           fs=sampling_rate_hz,
                           nperseg=nperseg_actual,
                           noverlap=noverlap_actual,
                           nfft=nfft_points,
                           window='hann',
                           scaling='spectrum')

# Convert PSD from s²/Hz to ms²/Hz
psd_ms2_per_hz = psd * (1000**2)

# Standard HRV frequency band definitions
VLF_band = (0.003, 0.04)    # Very Low Frequency (not typically interpreted for short recordings)
LF_band  = (0.04, 0.15)     # Low Frequency (sympathetic + parasympathetic)
HF_band  = (0.15, 0.4)      # High Frequency (parasympathetic / respiratory)

vlf_indices = np.where((fafrequencies >= VLF_band[0]) & (fafrequencies < VLF_band[1]))[0]
lf_indices  = np.where((fafrequencies >= LF_band[0])  & (fafrequencies < LF_band[1]))[0]
hf_indices  = np.where((fafrequencies >= HF_band[0])  & (fafrequencies < HF_band[1]))[0]

# Frequency resolution (Hz per bin) used as integration step
df = fafrequencies[1] - fafrequencies[0]

# Band power (rectangular integration of PSD)
total_power = np.sum(psd_ms2_per_hz) * df
vlf_power   = np.sum(psd_ms2_per_hz[vlf_indices]) * df
lf_power    = np.sum(psd_ms2_per_hz[lf_indices]) * df
hf_power    = np.sum(psd_ms2_per_hz[hf_indices]) * df

# Normalized units (LF+HF as denominator, VLF excluded per standard convention)
lf_norm = (lf_power / (lf_power + hf_power)) * 100 if (lf_power + hf_power) > 0 else 0
hf_norm = (hf_power / (lf_power + hf_power)) * 100 if (lf_power + hf_power) > 0 else 0

# LF/HF ratio — sympathovagal balance indicator
lf_hf_ratio = lf_power / hf_power if hf_power > 0 else np.inf

print("\n--- HRV Frequency Domain Analysis Results ---")
print(f"Total Power: {total_power:.2f} ms^2")
print(f"VLF Power: {vlf_power:.2f} ms^2 (Band: {VLF_band[0]}-{VLF_band[1]} Hz)")
print(f"LF Power: {lf_power:.2f} ms^2 (Band: {LF_band[0]}-{LF_band[1]} Hz)")
print(f"HF Power: {hf_power:.2f} ms^2 (Band: {HF_band[0]}-{HF_band[1]} Hz)")
print(f"Normalized LF (LFnu): {lf_norm:.2f}")
print(f"Normalized HF (HFnu): {hf_norm:.2f}")
print(f"LF/HF Ratio: {lf_hf_ratio:.2f}")


# =============================================================================
# === SNR Calculation ===
# Estimates signal quality by comparing the spectral power at the fundamental
# heart rate frequency and its harmonics against total broadband power.
# =============================================================================

fs = actual_sampling_rate

# High-resolution PSD of the outlier-corrected (pre-filter) signal
frequencies, psd = welch(y_fixed,
                         fs=fs,
                         nperseg=8192,
                         noverlap=4096,
                         nfft=65536,
                         scaling='density')

# Define narrow bands around the fundamental HR and first two harmonics
plusminusrange = 0.2    # ±0.2 Hz window around each harmonic
print("BPS")
print(bps)
heart_rate_range  = (bps - plusminusrange, bps + plusminusrange)
first_harmonic    = (bps*2 - plusminusrange, bps*2 + plusminusrange)
second_harmonic   = (bps*3 - plusminusrange, bps*3 + plusminusrange)

# Integrate power within each harmonic band
zero_mask = (frequencies >= heart_rate_range[0]) & (frequencies <= heart_rate_range[1])
first_mask = (frequencies >= first_harmonic[0])   & (frequencies <= first_harmonic[1])
second_mask = (frequencies >= second_harmonic[0]) & (frequencies <= second_harmonic[1])

zero_h_area   = simpson(psd[zero_mask],   frequencies[zero_mask])
first_h_area  = simpson(psd[first_mask],  frequencies[first_mask])
second_h_area = simpson(psd[second_mask], frequencies[second_mask])
heart_area = zero_h_area + first_h_area + second_h_area

# Broadband analysis window (0.5–15 Hz)
mask_all = (frequencies >= 0.5) & (frequencies <= 15)

# Dominant frequency in the broadband window (cross-check against peak-based HR)
dominant_freq = frequencies[mask_all][np.argmax(psd[mask_all])]
heart_rate_bpm = dominant_freq * 60    # Hz → BPM

all_area   = simpson(psd[mask_all], frequencies[mask_all])
noise_area = simpson(psd[mask_all], frequencies[mask_all]) - heart_area

# SNR: ratio of cardiac harmonic power to residual noise power
snr = heart_area / noise_area
print(snr)

# HTTR (Heart-to-Total Ratio): fraction of broadband power attributable to cardiac signal
print("httr")
httr = heart_area / all_area
print(httr)


# =============================================================================
# === APA (Average Peak Amplitude) ===
# Mean amplitude of the bandpass-filtered signal at detected peak locations.
# Provides a rough proxy for pulse strength.
# =============================================================================

peak_values = hpfiltered[filtered_peaks]
average_peak_value = np.mean(peak_values)
print(average_peak_value)


# =============================================================================
# === TERMA Peak Detection (Elgendi 2013) ===
# Alternative systolic-peak detector based on two event-related moving
# averages plus an offset threshold (no fixed prominence). Clip+square the
# bandpassed signal, then mark "blocks of interest" where a short moving
# average (systolic-peak width) exceeds a long one (one heartbeat) plus an
# offset; the bandpassed max within each wide-enough block is the peak.
# Reference: Elgendi M, et al. PLoS ONE 2013;8(10):e76585.
# =============================================================================

def detect_peaks_terma(sig, fs, w1_ms=111.0, w2_ms=667.0, beta=0.02):
    sig = np.asarray(sig, dtype=float)
    clipped = np.clip(sig, 0.0, None)
    squared = clipped * clipped
    w1 = max(1, int(round(w1_ms / 1000.0 * fs)))
    w2 = max(1, int(round(w2_ms / 1000.0 * fs)))
    if len(squared) <= w2:
        return np.array([], dtype=int)
    ma_peak = np.convolve(squared, np.ones(w1) / w1, mode="same")
    ma_beat = np.convolve(squared, np.ones(w2) / w2, mode="same")
    alpha = beta * float(np.mean(squared))
    blocks = ma_peak > (ma_beat + alpha)
    peaks = []
    n = len(blocks)
    i = 0
    while i < n:
        if not blocks[i]:
            i += 1
            continue
        j = i
        while j < n and blocks[j]:
            j += 1
        if (j - i) >= w1:
            peaks.append(i + int(np.argmax(sig[i:j])))
        i = j
    return np.asarray(peaks, dtype=int)

# Match the dashboard: run TERMA on a 0.5-8 Hz bandpass (the app's canonical
# band), not the 0.6-3.3 Hz hpfiltered that Plot 2 uses.
terma_bandpassed = np.asarray(
    bandpass(ir_smooth, lowcut=0.5, highcut=8, fs=actual_sampling_rate))
terma_peaks = detect_peaks_terma(terma_bandpassed, actual_sampling_rate)

# Same small-RR doublet filter as the non-TERMA path: walk the peaks and
# reject any whose interval from the last kept peak is below 0.6*median;
# when two are too close, keep the taller one (amplitude on the TERMA
# detection signal).
if len(terma_peaks) > 2:
    _tt = terma_peaks * time_interval + start_time
    _med = np.median(np.diff(_tt))
    _kept = [0]
    for i in range(1, len(terma_peaks)):
        rr = _tt[i] - _tt[_kept[-1]]
        if 0.6 * _med < rr:
            _kept.append(i)
        elif terma_bandpassed[terma_peaks[i]] > terma_bandpassed[terma_peaks[_kept[-1]]]:
            _kept[-1] = i
    terma_peaks = terma_peaks[np.asarray(_kept, dtype=int)]

terma_peak_times = terma_peaks * time_interval + start_time
if len(terma_peaks) > 1:
    terma_rr = np.diff(terma_peak_times)
    terma_hr = 60.0 / np.mean(terma_rr)
    terma_hrv = np.std(terma_rr)
else:
    terma_hr = float("nan")
    terma_hrv = float("nan")
print(f"TERMA peaks: {len(terma_peaks)}  HR: {terma_hr:.1f} BPM")


# =============================================================================
# === Plotting ===
# =============================================================================

# --- Plot 1: Raw Signal with Outlier Highlights ---
plt.figure(1, figsize=(12, 6))
plt.plot(x_filtered, ir_filtered, label='IR Smoothed')
plt.plot(x_filtered, ir_filtered, 'o', label='IR Raw', markersize=3)
plt.plot(bad_x, bad_y, 'o', label='IR Raw', markersize=4, color='black')

plt.title(f"PPG Signal — Smoothed & Raw — {start_time}s to {end_time}s")
plt.xlabel("Time (seconds)")
plt.ylabel("Sensor Value")
plt.legend()
plt.xlim(start_time, max(x_filtered)+5)
plt.grid(True)
plt.tight_layout()


# --- Plot 2: Bandpass-Filtered Signal with All Peak Layers ---
plt.figure(2, figsize=(12, 6))
plt.plot(x_filtered, hpfiltered, label='High Pass Detrended', color='purple')
plt.plot(filtered_peak_times, hpfiltered[filtered_peaks], 'ro', label="Peaks", markersize=4)
plt.plot(peak_times, hpfiltered[peaks], 'ro', label="unfiltered peaks", markersize=2, color='green')
plt.plot(ranges_removed_peak_times, hpfiltered[ranges_removed_peaks], 'ro', label="removed ranges peaks", markersize=4, color='black')
plt.plot(custom_peak_times, hpfiltered[custom_peaks], 'ro', label="added peaks", markersize=4, color='red')

# Overlay computed HR and HRV as a text box
stats_text = f"Heart Rate: {heart_rate:.1f} BPM\nHRV (std RR): {hrv:.3f} s"
plt.text(0.83, 0.6, stats_text, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, boxstyle="round"))

plt.title(f"PPG Signal (HP Detrended) — {start_time}s to {end_time}s")
plt.xlabel("Time (seconds)")
plt.ylabel("HP Detrended Value")
plt.legend()
plt.xlim(start_time, max(x_filtered)+5)
plt.grid(True)
plt.tight_layout()


# --- Plots 3 & 4: FFT + PSD spectra (hidden) ---
# Set SHOW_FFT_PSD = True to bring these two pop-ups back. The spectra and
# SNR are still computed above; only the figures are suppressed.
SHOW_FFT_PSD = False
if SHOW_FFT_PSD:
    # --- Plot 3: FFT Magnitude Spectrum ---
    plt.figure(3, figsize=(12, 6))
    plt.plot(xf, yf, label='High Pass Detrended', color='purple')
    plt.title("FFT of the Signal")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.xlim(0, 9)
    plt.grid(True)


    # --- Plot 4: Power Spectral Density with HR Bands ---
    plt.figure(4, figsize=(12, 6))
    plt.plot(frequencies, psd, label='PSD')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (V²/Hz)')
    plt.title(f'PPG Power Spectral Density\nDominant HR: {heart_rate_bpm:.1f} BPM | SNR: {snr:.2f}')

    # Shade the fundamental and first harmonic heart rate bands
    plt.axvspan(heart_rate_range[0], heart_rate_range[1], color='green', alpha=0.1, label='Heart Rate Band')
    plt.axvspan(first_harmonic[0],   first_harmonic[1],   color='green', alpha=0.1, label='First Harmonic Heart Rate Band')

    # Mark dominant frequency and its harmonics
    plt.axvline(dominant_freq,   color='red',    linestyle='--', label=f'Dominant Frequency ({dominant_freq:.2f} Hz)')
    plt.axvline(dominant_freq*2, color='purple', linestyle='--', label=f'Dominant Frequency ({dominant_freq*2:.2f} Hz)')
    plt.axvline(dominant_freq*3, color='purple', linestyle='--', label=f'Dominant Frequency ({dominant_freq*3:.2f} Hz)')
    plt.axvline(dominant_freq*4, color='purple', linestyle='--', label=f'Dominant Frequency ({dominant_freq*4:.2f} Hz)')

    plt.xlim(0, 15)
    plt.ylim(0, np.max(psd) * 1.1)

print("hiiii")

# --- pyhrv Frequency Domain Cross-check ---
# Uses the pyhrv library as an independent verification of the LF/HF calculation.
# pyhrv.welch_psd builds a PSD figure even with show=False, so close any figure
# it opens (keeping our Plot 1/2/5) to suppress the extra Welch PSD pop-up.
nni = np.array(re_filtered_intervals)
_figs_before = set(plt.get_fignums())
result = fd.welch_psd(nni=nni, show=False)
for _fnum in set(plt.get_fignums()) - _figs_before:
    plt.close(_fnum)

print(result['fft_ratio'])
freqbands = result['fft_abs']
print(freqbands)
print("hi")


# --- Plot 5: 0.5-8 Hz Bandpass (app band) with TERMA (moving-average) Peaks ---
# Mirrors the dashboard: 0.5-8 Hz bandpass + Elgendi 2013 two-moving-average
# adaptive-threshold detector, instead of Plot 2's 0.6-3.3 Hz + find_peaks.
plt.figure(5, figsize=(12, 6))
plt.plot(x_filtered, terma_bandpassed, label='Bandpass 0.5-8 Hz', color='purple')
plt.plot(terma_peak_times, terma_bandpassed[terma_peaks], 'o', label="TERMA peaks",
         markersize=4, color='red')

terma_stats = f"TERMA peaks: {len(terma_peaks)}\nHeart Rate: {terma_hr:.1f} BPM\nHRV (std RR): {terma_hrv:.3f} s"
plt.text(0.83, 0.6, terma_stats, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, boxstyle="round"))

plt.title(f"PPG Signal (0.5-8 Hz) — TERMA Peaks — {start_time}s to {end_time}s")
plt.xlabel("Time (seconds)")
plt.ylabel("Bandpassed Value (0.5-8 Hz)")
plt.legend()
plt.xlim(start_time, max(x_filtered)+5)
plt.grid(True)
plt.tight_layout()

plt.show()