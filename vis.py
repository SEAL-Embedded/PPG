import matplotlib.pyplot as plt
import csv
import os
import numpy as np
from scipy.signal import hilbert, savgol_filter, find_peaks, welch
import pandas as pd
from scipy.integrate import simpson
from scipy.interpolate import interp1d



# === File paths ===
baser_path = os.path.join("PPG-Sleepiness-Detection", "data")
base_path = os.path.join(baser_path, "firstTests")
ir_path = os.path.join(base_path, "400hzlongsample.csv")
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




print(len(rawtimestamps))
print(len(ir_data))


timestamps_seconds = np.array(rawtimestamps) / 1e6  # µs → seconds
start_time = timestamps_seconds[0]
timestamps = timestamps_seconds - start_time  # Now starts at 0

start_time = timestamps[0]
end_time = timestamps[-1]
num_samples = int(np.ceil((end_time - start_time) * 400))
uniform_timestamps = np.linspace(start_time, end_time, num_samples)
interpolator = interp1d(timestamps, ir_data, kind='cubic', fill_value='extrapolate')
resampled_signal = interpolator(uniform_timestamps)


# === Time axis (assuming 200 Hz sampling) ===
time_interval = 1/395  # 1/200
x_values = [i * time_interval for i in range(len(ir_data))]
actual_sampling_rate = 1/time_interval

# === CONTROL: Time range to plot (in seconds) ===
start_time = 50   # set this to your desired start
end_time = 500     # set this to your desired end

# === Filter data based on time range ===
filtered_indices = [i for i, t in enumerate(x_values) if start_time <= t <= end_time]
x_filtered = [x_values[i] for i in filtered_indices]
ir_filtered = [resampled_signal[i] for i in filtered_indices]


# === Raw Points ===
x_raw_filtered = [x_values[i] for i in range(len(x_values)) if start_time <= x_values[i] <= end_time]
ir_raw_filtered = [resampled_signal[i] for i in range(len(x_values)) if start_time <= x_values[i] <= end_time]

# === Remove Outliers ===
y = np.array(ir_raw_filtered)
y_median = pd.Series(y).rolling(window=15, center=True, min_periods=1).median()
gap = y_median - y
threshold = 6 * np.std(gap.dropna())
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
x_smoothed = x_values[:len(ir_smooth)]



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
peaks, props = find_peaks(hpfiltered, prominence=8, width = 0.2, distance = 0.48*actual_sampling_rate)


# === FFT ===
y = hpfiltered  # your filtered signal
fs = actual_sampling_rate  # your sampling rate, e.g., 50 Hz

N = len(y)
yf = np.fft.fft(y)
xf = np.fft.fftfreq(N, 1/fs)

idx = xf >= 0
xf = xf[idx]
yf = np.abs(yf[idx])  # magnitude


# === Biometric Calculation ===
peak_times = np.array(peaks * time_interval) + start_time
print(peak_times)
rr_intervals = np.diff(peak_times)
print(rr_intervals)
median_rr = np.median(rr_intervals)
print(median_rr)

filtered_peaks = [peaks[0]]
for i in range(1, len(peak_times)):
    rr = peak_times[i] - peak_times[i - 1]
    if 0.5 * median_rr < rr < 1.7 * median_rr:
        filtered_peaks.append(peaks[i])

print(filtered_peaks)
# Recalculate using filtered peaks
filtered_peak_times = np.array(filtered_peaks) * time_interval + start_time
filtered_rr_intervals = np.diff(filtered_peak_times)

heart_rate = 60 / np.mean(filtered_rr_intervals)
bps = heart_rate / 60
hrv = np.std(filtered_rr_intervals)

# === LF/HF Ratio ===
fs = 400
frequencies, psd = welch(resampled_signal, 
                        fs=fs, 
                        nperseg=8192,  # Window size
                        noverlap=4096,  # Overlap between segments
                        scaling='density')

# Find dominant heart rate frequency
heart_band = (0.5, 4.0)  # 0.5-4 Hz (30-240 BPM)
mask_heart = (frequencies >= heart_band[0]) & (frequencies <= heart_band[1])
dominant_freq = frequencies[mask_heart][np.argmax(psd[mask_heart])]
heart_rate_bpm = dominant_freq * 60  # Convert Hz to BPM

# Calculate SNR (Heart band vs Noise band)
noise_band_low = (bps-0.2, bps+0.2)   # Very low frequency noise
first_harmonic = (bps*2-0.2, bps*2+0.2)
noise_band_high = (4.0, 10.0) # Higher frequency noise
mask_noise = ((frequencies >= noise_band_low[0]) & (frequencies <= noise_band_low[1])) | \
             ((frequencies >= noise_band_high[0]) & (frequencies <= noise_band_high[1]))

snr = np.trapz(psd[mask_heart], frequencies[mask_heart]) / \
      np.trapz(psd[mask_noise], frequencies[mask_noise])

# === Plot 1: Smoothed and Raw in Window 1 ===
plt.figure(1, figsize=(12, 6))
plt.plot(x_filtered, ir_smooth, label='IR Smoothed')
plt.plot(x_raw_filtered, ir_raw_filtered, 'o', label='IR Raw', markersize=3)
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

# === Plot 3: FFT ===
plt.figure(3, figsize=(12, 6))
plt.plot(xf, yf, label='High Pass Detrended', color='purple')
plt.title("FFT of the Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)

# === Plot 4: RR-Interval PSD ===
plt.figure(4, figsize=(12, 6))
plt.semilogy(frequencies, psd, label='PSD')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (V²/Hz)')
plt.title(f'PPG Power Spectral Density\nDominant HR: {heart_rate_bpm:.1f} BPM | SNR: {snr:.2f}')

# Add heart rate band shading
plt.axvspan(noise_band_low[0], noise_band_low[1], color='green', alpha=0.1, label='Heart Rate Band')
plt.axvspan(first_harmonic[0], first_harmonic[1], color='green', alpha=0.1, label='First Harmonic Heart Rate Band')

plt.axvline(dominant_freq, color='red', linestyle='--', label=f'Dominant Frequency ({dominant_freq:.2f} Hz)')
plt.axvline(dominant_freq*2, color='purple', linestyle='--', label=f'Dominant Frequency ({dominant_freq*2:.2f} Hz)')
plt.axvline(dominant_freq*3, color='purple', linestyle='--', label=f'Dominant Frequency ({dominant_freq*3:.2f} Hz)')
plt.axvline(dominant_freq*4, color='purple', linestyle='--', label=f'Dominant Frequency ({dominant_freq*4:.2f} Hz)')

# Formatting
plt.xlim(0, 15)  # Show up to 15 Hz (no need to show Nyquist at 200 Hz)
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()


# === Show both windows ===

plt.show() 