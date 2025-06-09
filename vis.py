import matplotlib.pyplot as plt
import csv
import os
import numpy as np
from scipy.signal import hilbert, savgol_filter, find_peaks, welch, detrend
import pandas as pd
from scipy.integrate import simpson
from scipy.interpolate import interp1d

# === File paths ===
baser_path = os.path.join("PPG-Sleepiness-Detection", "data")
base_path = os.path.join(baser_path, "firstTests")
ir_path = os.path.join(base_path, "belle.csv")

# === Load IR data ===
with open("data\\firstTests\\ppg_data.csv", 'r') as file:
    reader = csv.reader(file)
    
    # Read all rows once and extract timestamps and IR data
    rawtimestamps = []
    ir_data = []
    for row in reader:
        if len(row) >= 2 and row[1].strip() != '':  # Ensure row has at least 2 columns and column 1 is non-empty
            rawtimestamps.append(float(row[0]))  # Column 0 = timestamp
            ir_data.append(float(row[1]))        # Column 1 = PPG signal


time_interval = 1/400  # 1/200
time_interval_us = (int)(time_interval * 1e6)  # Convert to microseconds
x_values = [i * time_interval for i in range(len(ir_data))]
actual_sampling_rate = 1/time_interval

corrected_timestamps = np.array(rawtimestamps, dtype=np.int64)  # Convert to numpy array for easier manipulation
print(len(corrected_timestamps))
for i in range(1, len(corrected_timestamps)):
        time_diff = corrected_timestamps[i] - corrected_timestamps[i-1]

        if abs(time_diff) > 10000:
            print(f"\n--- Detected jump at index {i} ---")
            print(f"  Previous timestamp: {corrected_timestamps[i-1]} us")
            print(f"  Observed difference: {time_diff} us")

            # Calculate the actual "missing" time or excessive delay
            # This is the amount we need to subtract from subsequent timestamps.
            # We assume the two points involved in the jump *should* have been
            # exactly one expected_period_us apart.
            # So, the "excess" time is (observed_diff - expected_period).
            # This excess is what causes everything after it to be shifted.
            excess_time_us = time_diff - time_interval_us

            print(f"  Expected period: {time_interval_us} us")
            print(f"  Excess time to correct: {excess_time_us} us")

            # Subtract this excess time from all *subsequent* timestamps
            corrected_timestamps[i:] -= excess_time_us
            print(f"  Timestamp at index {i} corrected to: {corrected_timestamps[i]} us")
            print(f"  New difference from previous: {corrected_timestamps[i] - corrected_timestamps[i-1]} us")
            print(f"  Remaining timestamps adjusted by {excess_time_us} us.")
            print("---------------------------------")


timestamps_seconds = np.array(corrected_timestamps) / 1e6  # µs → seconds
first_time = timestamps_seconds[0]
timestamps = timestamps_seconds - first_time  # Now starts at 0
first_time = timestamps[0]
print("useless")
print("start time:", first_time)
end_time = timestamps[-1]
num_samples = int(np.ceil((end_time - first_time) * actual_sampling_rate))
uniform_timestamps = np.linspace(first_time, end_time, num_samples)
interpolator = interp1d(timestamps, ir_data, kind='cubic', fill_value='extrapolate')
resampled_signal = interpolator(uniform_timestamps)

print("new time")
print(len(uniform_timestamps))
# === Time axis (assuming 200 Hz sampling) ===


# === CONTROL: Time range to plot (in seconds) ===
start_time = 0   # set this to your desired start
end_time = 300     # set this to your desired end

# === Filter data based on time range ===
filtered_indices = [i for i, t in enumerate(uniform_timestamps) if start_time <= t <= end_time]

x_filtered = [uniform_timestamps[i] for i in filtered_indices]

ir_filtered = [resampled_signal[i] for i in filtered_indices]

# === Raw Points ===


# === Remove Outliers ===
y = np.array(ir_filtered)
y_median = pd.Series(y).rolling(window=15, center=True, min_periods=1).median()
gap = abs(y_median - y)
threshold = 4.25 * np.std(gap.dropna())
outliers = gap > threshold
x_filtered = np.array(x_filtered)
bad_x = x_filtered[outliers]
bad_y = y[outliers]


y_fixed = np.copy(y)

y_fixed[outliers.to_numpy()] = np.interp(
    x_filtered[outliers.to_numpy()],  # x positions of bad points
    x_filtered[~outliers.to_numpy()], # x positions of good points
    y[~outliers.to_numpy()]           # y values of good points
)

# === Smoothing (moving average) ===
window_size = 0
smooth = lambda data: [np.mean(data[max(0, i-window_size): min(len(data), i+window_size + 1)]) for i in range(len(data))]

ir_smooth = smooth(y_fixed)
x_smoothed = uniform_timestamps[:len(ir_smooth)]

# === High Pass Detrend ===
from scipy.signal import butter, filtfilt
def bandpass(data, lowcut=0.5, highcut=10, fs=50, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

hpfiltered = bandpass(ir_smooth, lowcut=0.6, highcut=3.3, fs=actual_sampling_rate)

# === Peak Finding ===
peaks, props = find_peaks(hpfiltered, prominence=1.7, width = 0.18, distance = 0.5*actual_sampling_rate)

'''
# === FFT ===
y = hpfiltered  # your filtered signal
fs = actual_sampling_rate  # your sampling rate, e.g., 50 Hz

N = len(y)
yf = np.fft.fft(y)
xf = np.fft.fftfreq(N, 1/fs)

idx = xf >= 0
xf = xf[idx]
yf = np.abs(yf[idx])  # magnitude

'''
# === Biometric Calculation ===
peak_times = np.array(peaks * time_interval) + start_time
rr_intervals = np.diff(peak_times)
median_rr = np.median(rr_intervals)

filtered_peaks = [peaks[0]]
for i in range(1, len(peak_times)):
    rr = peak_times[i] - peak_times[i - 1]
    if 0.5 * median_rr < rr:
        filtered_peaks.append(peaks[i])

# Recalculate using filtered peaks
filtered_peak_times = np.array(filtered_peaks) * time_interval + start_time
filtered_rr_intervals = np.diff(filtered_peak_times)
print(filtered_peak_times)

with open('output_column.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for item in filtered_rr_intervals:
        writer.writerow([item])  # Wrap each item in a list to make it a row


heart_rate = 60 / np.mean(filtered_rr_intervals)
bps = heart_rate / 60
hrv = np.std(filtered_rr_intervals)



'''
# === LF/HF Ratio ===
sampling_rate_hz=4
segment_length_s=570
nfft_points=1024

r_peak_times_s = np.cumsum(filtered_rr_intervals)
# Adjust to start time from 0 if it's not already
r_peak_times_s = r_peak_times_s - r_peak_times_s[0]

# Check if the recording is long enough
if r_peak_times_s[-1] < segment_length_s:
    print(f"Warning: Recording is only {r_peak_times_s[-1]:.2f} seconds long, "
            f"which is shorter than the desired segment length of {segment_length_s} seconds.")
    segment_length_s = r_peak_times_s[-1] # Adjust segment length to actual duration

# --- 2. Interpolation/Resampling ---
# Create a regularly spaced time axis for interpolation
# Ensure the end time for interpolation matches the segment_length_s
time_interp = np.arange(0, segment_length_s, 1 / sampling_rate_hz)

# Use linear interpolation (as a common choice for this data)
# The original RR intervals are treated as values occurring at their respective R-peak times.
# To get instantaneous heart rate or instantaneous RR, we often interpolate the RR interval series itself.
# However, sometimes it's more appropriate to interpolate the instantaneous heart rate (60/RR) or log(RR).
# For simplicity and directness here, we interpolate the RR intervals.

# Handle cases where interpolation range might be outside available data
# scipy.interpolate.interp1d requires sorted unique x values.
unique_r_peak_times, unique_rr_intervals = np.unique(r_peak_times_s, return_index=True)
f_interp = interp1d(unique_r_peak_times, filtered_rr_intervals[unique_rr_intervals], kind='linear', fill_value="extrapolate")

# Get the interpolated RR interval series (in seconds)
interpolated_rr_s = f_interp(time_interp)


# --- 3. Detrending (Optional but Recommended) ---
# Remove linear trend from the interpolated signal to improve PSD estimation
detrended_signal = detrend(interpolated_rr_s)

# --- 4. Power Spectral Density (PSD) Estimation using Welch's Method ---
# Welch's method is a robust way to estimate PSD using FFT.
# It automatically handles windowing and averaging of segments.
# nperseg: Length of each segment used for FFT (similar to the 'N_FFT_points' concept).
#          Should be a power of 2 and typically less than or equal to nfft_points.
# noverlap: Number of points to overlap between segments.
# fs: Sampling frequency of the interpolated signal.
# nfft: Number of FFT points (pads signal with zeros if nperseg < nfft).

# For a 5-minute recording, we often just want one large segment for FFT (like the document implies N_FFT_points).
# So we set nperseg to be the nfft_points or the length of the signal if it's smaller.

# Ensure nperseg is not greater than the signal length
nperseg_actual = min(nfft_points, len(detrended_signal))

# Ensure nperseg is a power of 2
nperseg_actual = int(2**np.floor(np.log2(nperseg_actual)))

# No overlap usually for a single full segment FFT like the document describes for short term
# For longer recordings, overlapping segments (e.g., 50% overlap) are common.
noverlap_actual = 0 # No overlap for single segment approach

fafrequencies, psd = welch(detrended_signal,
                            fs=sampling_rate_hz,
                            nperseg=nperseg_actual,
                            noverlap=noverlap_actual,
                            nfft=nfft_points, # Use this to pad with zeros if nperseg < nfft_points
                            window='hann', # Hanning window, as suggested in the document
                            scaling='spectrum') # 'spectrum' returns power spectral density

# Convert PSD units to ms^2/Hz (if original was seconds, then psd is s^2/Hz)
psd_ms2_per_hz = psd * (1000**2)

# --- 5. Define Frequency Bands and Calculate Power ---
# Frequencies in Hz
VLF_band = (0.003, 0.04) # Note: For short-term, VLF is often not interpreted or included in normalization base
LF_band = (0.04, 0.15)
HF_band = (0.15, 0.4)

# Find indices corresponding to each band
vlf_indices = np.where((fafrequencies >= VLF_band[0]) & (fafrequencies < VLF_band[1]))[0]
lf_indices = np.where((fafrequencies >= LF_band[0]) & (fafrequencies < LF_band[1]))[0]
hf_indices = np.where((fafrequencies >= HF_band[0]) & (fafrequencies < HF_band[1]))[0]

# Calculate power within each band by summing the PSD values in the band
# multiplied by the frequency resolution (df)
df = fafrequencies[1] - fafrequencies[0] # Frequency resolution

total_power = np.sum(psd_ms2_per_hz) * df # Total power across all calculated frequencies

vlf_power = np.sum(psd_ms2_per_hz[vlf_indices]) * df
lf_power = np.sum(psd_ms2_per_hz[lf_indices]) * df
hf_power = np.sum(psd_ms2_per_hz[hf_indices]) * df

# Calculate normalized units (excluding VLF from the denominator, as per document)
lf_norm = (lf_power / (lf_power + hf_power)) * 100 if (lf_power + hf_power) > 0 else 0
hf_norm = (hf_power / (lf_power + hf_power)) * 100 if (lf_power + hf_power) > 0 else 0

# Calculate LF/HF Ratio
lf_hf_ratio = lf_power / hf_power if hf_power > 0 else np.inf # Handle division by zero


print("\n--- HRV Frequency Domain Analysis Results ---")
print(f"Total Power: {total_power:.2f} ms^2")
print(f"VLF Power: {vlf_power:.2f} ms^2 (Band: {VLF_band[0]}-{VLF_band[1]} Hz)")
print(f"LF Power: {lf_power:.2f} ms^2 (Band: {LF_band[0]}-{LF_band[1]} Hz)")
print(f"HF Power: {hf_power:.2f} ms^2 (Band: {HF_band[0]}-{HF_band[1]} Hz)")
print(f"Normalized LF (LFnu): {lf_norm:.2f}")
print(f"Normalized HF (HFnu): {hf_norm:.2f}")
print(f"LF/HF Ratio: {lf_hf_ratio:.2f}")



'''
# === SNR ===
fs = actual_sampling_rate  # Sampling frequency
frequencies, psd = welch(y_fixed, 
                        fs=fs, 
                        nperseg=8192,  # Window size
                        noverlap=4096,  # Overlap between segments
                        nfft=65536,
                        scaling='density')



# Calculate SNR (Heart band vs Noise band)
plusminusrange = 0.2
print("BPS")
print(bps)
heart_rate_range = (bps-plusminusrange, bps+plusminusrange)   # Very low frequency noise
first_harmonic = (bps*2-plusminusrange, bps*2+plusminusrange)
second_harmonic = (bps*3-plusminusrange, bps*3+plusminusrange)


zero_mask = (frequencies >= heart_rate_range[0]) & (frequencies <= heart_rate_range[1])
first_mask = (frequencies >= first_harmonic[0]) & (frequencies <= first_harmonic[1])
second_mask = (frequencies >= second_harmonic[0]) & (frequencies <= second_harmonic[1])
zero_h_area = simpson(psd[zero_mask], frequencies[zero_mask])
first_h_area = simpson(psd[first_mask], frequencies[first_mask])
second_h_area = simpson(psd[second_mask], frequencies[second_mask])
heart_area = zero_h_area + first_h_area + second_h_area

mask_all = ((frequencies >= 0.5) & (frequencies <= 15))


dominant_freq = frequencies[mask_all][np.argmax(psd[mask_all])]
heart_rate_bpm = dominant_freq * 60  # Convert Hz to BPM



all_area = simpson(psd[mask_all], frequencies[mask_all])
noise_area = simpson(psd[mask_all], frequencies[mask_all]) - heart_area

snr = heart_area / noise_area
print(snr)

print("httr")
httr = heart_area / all_area
print(httr)


# === calculating APA ===
peak_values = hpfiltered[filtered_peaks]

# 2. Calculate the average of these peak values
average_peak_value = np.mean(peak_values)

# 3. Print the result
print(average_peak_value)

# === Plot 1: Smoothed and Raw in Window 1 ===

plt.figure(1, figsize=(12, 6))
plt.plot(x_filtered, ir_filtered, label='IR Smoothed')
plt.plot(x_filtered, ir_filtered, 'o', label='IR Raw', markersize=3)
plt.plot(bad_x, bad_y, 'o', label='IR Raw', markersize=4, color = 'black')

plt.title(f"PPG Signal — Smoothed & Raw — {start_time}s to {end_time}s")
plt.xlabel("Time (seconds)")
plt.ylabel("Sensor Value")
plt.legend()
plt.xlim(start_time, max(x_filtered)+5)

plt.grid(True)
plt.tight_layout()


# === Plot 2: High Pass Detrend in Window 2 ===
plt.figure(2, figsize=(12, 6))
plt.plot(x_filtered, hpfiltered, label='High Pass Detrended', color='purple')
plt.plot(filtered_peak_times, hpfiltered[filtered_peaks], 'ro', label="Peaks", markersize = 3)
plt.plot(peak_times, hpfiltered[peaks], 'ro', label="unfiltered peaks", markersize = 2, color = 'green')

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



'''
# === Plot 3: FFT ===
plt.figure(3, figsize=(12, 6))
plt.plot(xf, yf, label='High Pass Detrended', color='purple')
plt.title("FFT of the Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, 9)  # Limit x-axis to 10 Hz for better visibility
plt.grid(True)
'''
# === Plot 4: PSD ===
plt.figure(4, figsize=(12, 6))
plt.plot(frequencies, psd, label='PSD')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (V²/Hz)')
plt.title(f'PPG Power Spectral Density\nDominant HR: {heart_rate_bpm:.1f} BPM | SNR: {snr:.2f}')

# Add heart rate band shading
plt.axvspan(heart_rate_range[0], heart_rate_range[1], color='green', alpha=0.1, label='Heart Rate Band')
plt.axvspan(first_harmonic[0], first_harmonic[1], color='green', alpha=0.1, label='First Harmonic Heart Rate Band')

plt.axvline(dominant_freq, color='red', linestyle='--', label=f'Dominant Frequency ({dominant_freq:.2f} Hz)')
plt.axvline(dominant_freq*2, color='purple', linestyle='--', label=f'Dominant Frequency ({dominant_freq*2:.2f} Hz)')
plt.axvline(dominant_freq*3, color='purple', linestyle='--', label=f'Dominant Frequency ({dominant_freq*3:.2f} Hz)')
plt.axvline(dominant_freq*4, color='purple', linestyle='--', label=f'Dominant Frequency ({dominant_freq*4:.2f} Hz)')
plt.xlim(0.5, 10)
plt.ylim(0, np.max(psd) * 1.1)  # Set y-axis limit to 110% of max PSD value


'''
# === Plot 5: RR-Interval PSD ===
plt.figure(5, figsize=(10, 6))
plt.plot(fafrequencies, psd_ms2_per_hz, color='blue', label='PSD')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density ($ms^2/Hz$)')
plt.title('Heart Rate Variability Power Spectral Density (PSD)')
#plt.grid(True, linestyle='--', alpha=0.7)

# Highlight frequency bands
plt.axvspan(LF_band[0], LF_band[1], color='red', alpha=0.2, label='LF Band')
plt.axvspan(HF_band[0], HF_band[1], color='green', alpha=0.2, label='HF Band')
plt.xlim(0, 1.5)  # Show up to 15 Hz (no need to show Nyquist at 200 Hz)

'''

# Formatting



# === Show both windows ===

plt.show() 