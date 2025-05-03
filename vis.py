import matplotlib.pyplot as plt
import csv
import os
import numpy as np
from scipy.signal import hilbert, savgol_filter, find_peaks, welch
import pandas as pd
from scipy.integrate import simpson


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

# === LF/HF Ratio ===
fs = 4  # resample rate (Hz)
t = np.arange(0, 150, 1/fs)

cumulative_time = np.cumsum(filtered_rr_intervals)
uniform_time = np.linspace(0, cumulative_time[-1], len(t))
NNI_interp = np.interp(uniform_time, cumulative_time, filtered_rr_intervals[:len(cumulative_time)])

# bandpass
def bandpass(data, lowcut, highcut, fs):
    nyq = 0.5 * fs
    b, a = butter(4, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, data)

LF = bandpass(NNI_interp, 0.04, 0.15, fs)
HF = bandpass(NNI_interp, 0.15, 0.4, fs)

# hilbert
LFiA = np.abs(hilbert(LF))
HFiA = np.abs(hilbert(HF))

# average
window_len = 150 * fs  # 300s * fs
step_size = 5 * fs    # 10s * fs

LFiA_mean = []
HFiA_mean = []

def trimmed_mean(x, trim=0.2):
    x_sorted = np.sort(x)
    n = len(x_sorted)
    x_trimmed = x_sorted[int(n*trim):int(n*(1-trim))]
    return np.mean(x_trimmed)

for start in range(0, len(LFiA) - window_len + 1, step_size):
    lw = LFiA[start:start + window_len]
    hw = HFiA[start:start + window_len]
    LFiA_mean.append(trimmed_mean(lw))
    HFiA_mean.append(trimmed_mean(hw))

# final
LF_HF_ratio = np.array(LFiA_mean) / np.array(HFiA_mean)
print("LF HF RATIO:")
print(LF_HF_ratio)


# psd
freqs, psd = welch(filtered_rr_intervals, fs=fs, nperseg=256)

# Define LF and HF bands
lf_band = (0.04, 0.15)
hf_band = (0.15, 0.4)

# Find indices of frequencies in each band
lf_idx = np.logical_and(freqs >= lf_band[0], freqs <= lf_band[1])
hf_idx = np.logical_and(freqs >= hf_band[0], freqs <= hf_band[1])

# Integrate the PSD over the frequency bands using Simpson’s rule
lf_power = simps(psd[lf_idx], freqs[lf_idx])
hf_power = simps(psd[hf_idx], freqs[hf_idx])

lf_hf_ratio = lf_power / hf_power if hf_power > 0 else np.nan

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
plt.plot(filtered_peak_times, hpfiltered[filtered_peaks], 'ro', label="Peaks", markersize = 2)

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
plt.figure(4, figsize=(8, 6))
plt.scatter(HFiA_mean, LFiA_mean, c='blue', alpha=0.6)
plt.xlabel('HFiA (Instantaneous Amplitude HF)')
plt.ylabel('LFiA (Instantaneous Amplitude LF)')
plt.title('2D LF-HF Scatter Plot (Stress Analysis)')
plt.grid(True)


# === Show both windows ===

plt.show() 