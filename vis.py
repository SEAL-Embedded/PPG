import matplotlib.pyplot as plt
import csv
import os
import numpy as np
from scipy.signal import hilbert, savgol_filter, find_peaks, welch
import pandas as pd

# === File paths ===
baser_path = os.path.join("PPG-Sleepiness-Detection", "data")
base_path = os.path.join(baser_path, "firstTests")
ir_path = os.path.join(base_path, "400hzlongsample.csv")
# === Load IR data ===
with open("data\\firstTests\\400hzlongsample.csv", 'r') as file:

    reader = csv.reader(file)
    ir_data = [float(value) for row in reader for value in row if value.strip() != '']


# === Time axis (assuming 200 Hz sampling) ===
time_interval = 1/400  # 1/200
x_values = [i * time_interval for i in range(len(ir_data))]
actual_sampling_rate = 1/time_interval

# === CONTROL: Time range to plot (in seconds) ===
start_time = 5   # set this to your desired start
end_time = 200     # set this to your desired end

# === Filter data based on time range ===
filtered_indices = [i for i, t in enumerate(x_values) if start_time <= t <= end_time]
x_filtered = [x_values[i] for i in filtered_indices]
ir_filtered = [ir_data[i] for i in filtered_indices]


# === Raw Points ===
x_raw_filtered = [x_values[i] for i in range(len(x_values)) if start_time <= x_values[i] <= end_time]
ir_raw_filtered = [ir_data[i] for i in range(len(x_values)) if start_time <= x_values[i] <= end_time]

# === Remove Outliers ===
y = np.array(ir_raw_filtered)

# Rolling median (captures the general curve shape)
y_median = pd.Series(y).rolling(window=15, center=True, min_periods=1).median()

# Calculate the gap BELOW the expected median
gap = y_median - y

# Set a threshold (you can tune this value!)
threshold = 6 * np.std(gap.dropna())

# Find points that are much lower than expected
outliers = gap > threshold
x_filtered = np.array(x_filtered)
bad_x = x_filtered[outliers]
bad_y = y[outliers]

y_fixed = np.copy(y)

# Interpolate bad points
y_fixed[outliers.to_numpy()] = np.interp(
    x_filtered[outliers.to_numpy()],  # x positions of bad points
    x_filtered[~outliers.to_numpy()], # x positions of good points
    y[~outliers.to_numpy()]           # y values of good points
)

# === Smoothing (moving average) ===
window_size = 1
smooth = lambda data: [np.mean(data[max(0, i-window_size): min(len(data), i+window_size + 1)]) for i in range(len(data))]

ir_smooth = smooth(y_fixed)
x_smoothed = x_values[:len(ir_smooth)]



# === High Pass Detrend ===
from scipy.signal import butter, filtfilt
def bandpass(data, lowcut=0.5, highcut=10, fs=50, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

hpfiltered = bandpass(ir_smooth, lowcut=0.5, highcut=30, fs=actual_sampling_rate)

# === Peak Finding ===
peaks, props = find_peaks(hpfiltered, prominence=35, width = 0.2)


# === FFT ===
# Your data
y = hpfiltered  # your filtered signal
fs = actual_sampling_rate  # your sampling rate, e.g., 50 Hz

# 1. Compute FFT
N = len(y)
yf = np.fft.fft(y)
xf = np.fft.fftfreq(N, 1/fs)

# 2. Keep only the positive half
idx = xf >= 0
xf = xf[idx]
yf = np.abs(yf[idx])  # magnitude

frequencies, power = welch(y, fs=fs, nperseg=16384)



# === Biometric Calculation ===
peak_times = (peaks * time_interval) + start_time
print(peak_times)
rr_intervals = np.diff(peak_times)
print(rr_intervals)
# Step 3: Compute median RR
median_rr = np.median(rr_intervals)
print(median_rr)

# Step 4: Filter peaks based on RR deviation
filtered_peaks = [peaks[0]]
for i in range(1, len(peak_times)):
    rr = peak_times[i] - peak_times[i - 1]
    if 0.55 * median_rr < rr < 1.65 * median_rr:
        filtered_peaks.append(peaks[i])

print(filtered_peaks)
# Recalculate using filtered peaks
filtered_peak_times = np.array(filtered_peaks) * time_interval + start_time
filtered_rr_intervals = np.diff(filtered_peak_times)

heart_rate = 60 / np.mean(filtered_rr_intervals)
hrv = np.std(filtered_rr_intervals)

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
plt.plot(filtered_peak_times, hpfiltered[filtered_peaks], 'ro', label="Peaks")

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
plt.semilogy(frequencies, power)
plt.title("FFT of the Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)


# === Show both windows ===

plt.show() 