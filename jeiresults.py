import matplotlib.pyplot as plt
import csv
import os
import numpy as np
from scipy.signal import hilbert, savgol_filter, find_peaks, welch, detrend, butter, filtfilt
import pandas as pd
from scipy.integrate import simpson
from scipy.interpolate import interp1d


# === File paths ===
data_folder = os.path.join("data", "firstTests")

def process_ppg_file(file_path):
    """
    Processes a single PPG signal file to calculate SNR and APH.

    Args:
        file_path (str): The full path to the PPG data file (CSV).

    Returns:
        tuple: A tuple containing (file_name, SNR, APH).
               Returns (file_name, None, None) if processing fails.
    """
    print(f"\n--- Processing file: {os.path.basename(file_path)} ---")

    # --- Load IR data ---
    try:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            rawtimestamps = []
            ir_data = []
            for row in reader:
                if len(row) >= 2 and row[1].strip() != '':
                    rawtimestamps.append(float(row[0]))
                    ir_data.append(float(row[1]))
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return os.path.basename(file_path), None, None

    if not ir_data:
        print(f"No valid data found in {file_path}.")
        return os.path.basename(file_path), None, None

    time_interval = 1 / 400
    time_interval_us = int(time_interval * 1e6)
    actual_sampling_rate = 1 / time_interval
    x_values = [i * time_interval for i in range(len(ir_data))]
    corrected_timestamps = np.array(rawtimestamps, dtype=np.int64)
    for i in range(1, len(corrected_timestamps)):
        time_diff = corrected_timestamps[i] - corrected_timestamps[i - 1]
        if abs(time_diff) > 50000:
            excess_time_us = time_diff - time_interval_us
            corrected_timestamps[i:] -= excess_time_us

    timestamps_seconds = np.array(corrected_timestamps) / 1e6
    start_time_raw = timestamps_seconds[0]
    timestamps = timestamps_seconds - start_time_raw
    # start_time_processed = timestamps[0] # This isn't needed as timestamps now start from 0
    end_time_processed = timestamps[-1]
    num_samples = int(np.ceil((end_time_processed - 0) * actual_sampling_rate)) # Assuming start is 0
    uniform_timestamps = np.linspace(0, end_time_processed, num_samples)
    print("new time")
    print(len(uniform_timestamps))

    # Handle cases where timestamps might not be strictly increasing due to correction or input data
    unique_timestamps, unique_indices = np.unique(timestamps, return_index=True)


    interpolator = interp1d(unique_timestamps, np.array(ir_data)[unique_indices], kind='cubic', fill_value='extrapolate')
    resampled_signal = interpolator(uniform_timestamps)

    # --- Filter data based on time range ---
    start_time_segment = 20  # Renamed to avoid conflict with `start_time` for plotting if it were used
    end_time_segment = min(320, end_time_processed) # Renamed to avoid conflict

    print(end_time_segment)
    # === Filter data based on time range ===
    # Important: Convert uniform_timestamps to a NumPy array for boolean indexing
    uniform_timestamps_np = np.array(uniform_timestamps)

    # Apply the time range filter
    time_mask = (uniform_timestamps_np >= start_time_segment) & \
                (uniform_timestamps_np <= end_time_segment)

    x_filtered_calc = uniform_timestamps_np[time_mask]
    ir_filtered_calc = resampled_signal[time_mask] # Also apply mask to resampled_signal

    x_filtered_calc = [x_values[i] for i in range(len(x_values)) if start_time_segment <= x_values[i] <= end_time_segment]
    ir_filtered_calc = [resampled_signal[i] for i in range(len(x_values)) if start_time_segment <= x_values[i] <= end_time_segment]


    # --- Remove Outliers ---
    y = np.array(ir_filtered_calc)
    y_median = pd.Series(y).rolling(window=15, center=True, min_periods=1).median()
    gap = y_median - y
    threshold = 6 * np.std(gap.dropna())
    outliers = gap > threshold
    x_filtered = np.array(x_filtered_calc)
    bad_x = x_filtered[outliers]
    bad_y = y[outliers]

    y_fixed = np.copy(y)

    y_fixed[outliers.to_numpy()] = np.interp(
        x_filtered[outliers.to_numpy()],  # x positions of bad points
        x_filtered[~outliers.to_numpy()], # x positions of good points
        y[~outliers.to_numpy()]           # y values of good points
    )


    # --- Smoothing (moving average) ---
    window_size = 0 # No smoothing if 0. Set to a positive integer for actual smoothing.
    if window_size > 0:
        ir_smooth = [np.mean(y_fixed[max(0, i-window_size): min(len(y_fixed), i+window_size + 1)]) for i in range(len(y_fixed))]
    else:
        ir_smooth = y_fixed # No smoothing applied if window_size is 0

    # --- High Pass Detrend ---
    def bandpass(data, lowcut=0.5, highcut=10, fs=50, order=2):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    # Ensure ir_smooth is not empty
    if len(ir_smooth) == 0:
        print(f"Smoothed signal is empty in {file_path}. Cannot perform bandpass filtering.")
        return os.path.basename(file_path), None, None

    hpfiltered = bandpass(ir_smooth, lowcut=0.6, highcut=3.3, fs=actual_sampling_rate)

    # --- Peak Finding ---
    try:
        peaks, props = find_peaks(hpfiltered, prominence=8, width=0.2, distance=0.55 * actual_sampling_rate)
    except ValueError as e:
        print(f"Error during peak finding in {file_path}: {e}")
        return os.path.basename(file_path), None, None

    # --- Biometric Calculation (for SNR) ---
    peak_times = np.array(peaks * time_interval) + start_time_segment
    rr_intervals = np.diff(peak_times)


    median_rr = np.median(rr_intervals)

    filtered_peaks = [peaks[0]]
    for i in range(1, len(peak_times)):
        rr = peak_times[i] - peak_times[i - 1]
        if 0.5 * median_rr < rr < 1.7 * median_rr:
            filtered_peaks.append(peaks[i])


    # Recalculate filtered_peak_times based on the filtered_peaks indices
    filtered_peak_times = np.array(filtered_peaks) * time_interval + start_time_segment
    filtered_rr_intervals = np.diff(filtered_peak_times)

    if os.path.basename(file_path) == "21_middle.csv":
        print(filtered_peak_times)

    heart_rate = 60 / np.mean(filtered_rr_intervals)
    bps = heart_rate / 60

    # --- SNR Calculation ---
    fs_snr = actual_sampling_rate
    nperseg_snr = min(8192, len(ir_filtered_calc))
    #nperseg_snr = int(2**np.floor(np.log2(nperseg_snr)))
    noverlap_snr = nperseg_snr // 2

    if len(ir_filtered_calc) < nperseg_snr:
        print(f"Signal length ({len(ir_filtered_calc)}) is too short for nperseg_snr ({nperseg_snr}) in {file_path}. Adjusting nperseg_snr.")
        nperseg_snr = len(ir_filtered_calc)
        noverlap_snr = 0 # No overlap for a single segment

    if nperseg_snr == 0:
        print(f"Insufficient data to perform Welch's method in {file_path}.")
        return os.path.basename(file_path), None, None

    frequencies, psd = welch(ir_filtered_calc,
                             fs=fs_snr,
                             nperseg=8192,
                             noverlap=4096,
                             scaling='density')

    plusminusrange = 0.1
    print(bps)
    heart_rate_range = (bps - plusminusrange, bps + plusminusrange)
    first_harmonic = (bps * 2 - plusminusrange, bps * 2 + plusminusrange)
    second_harmonic = (bps * 3 - plusminusrange, bps * 3 + plusminusrange)

    # Define masks carefully to avoid empty selections
    zero_mask = (frequencies >= heart_rate_range[0]) & (frequencies <= heart_rate_range[1])
    first_mask = (frequencies >= first_harmonic[0]) & (frequencies <= first_harmonic[1])
    second_mask = (frequencies >= second_harmonic[0]) & (frequencies <= second_harmonic[1])

    heart_area = 0
    if np.any(zero_mask):
        heart_area += simpson(psd[zero_mask], frequencies[zero_mask])
    if np.any(first_mask):
        heart_area += simpson(psd[first_mask], frequencies[first_mask])
    if np.any(second_mask):
        heart_area += simpson(psd[second_mask], frequencies[second_mask])

    # Define the "all" frequency band for noise calculation
    mask_all = (frequencies >= 0.5) & (frequencies <= 15) # Frequencies typically relevant for PPG

    all_area = 0
    if np.any(mask_all):
        all_area = simpson(psd[mask_all], frequencies[mask_all])

    noise_area = all_area - heart_area

    if noise_area <= 0:
        snr = float('inf') if heart_area > 0 else 0
    else:
        snr = heart_area / noise_area

    # --- Calculating APA (Average Peak Height) ---
    # Ensure filtered_peaks contains valid indices within hpfiltered
    valid_filtered_peaks = [p for p in filtered_peaks if p < len(hpfiltered)]
    peak_values = hpfiltered[valid_filtered_peaks]
    average_peak_value = np.mean(peak_values) if len(peak_values) > 0 else 0

    # --- Store plotting data if this is the target file ---
    plotting_data = None
    if os.path.basename(file_path) == "21_middle.csv":
        # Calculate dominant frequency for plotting title
        # This requires finding the max PSD in the heart rate band
        mask_heart_combined = ((frequencies >= heart_rate_range[0]) & (frequencies <= heart_rate_range[1])) | \
                              ((frequencies >= first_harmonic[0]) & (frequencies <= first_harmonic[1])) | \
                              ((frequencies >= second_harmonic[0]) & (frequencies <= second_harmonic[1]))
        
        dominant_freq = 0
        heart_rate_bpm = 0
        if np.any(mask_heart_combined):
            dominant_freq_idx = np.argmax(psd[mask_heart_combined])
            dominant_freq = frequencies[mask_heart_combined][dominant_freq_idx]
            heart_rate_bpm = dominant_freq * 60

        plotting_data = {
            'frequencies': frequencies,
            'psd': psd,
            'heart_rate_bpm': heart_rate_bpm,
            'snr': snr,
            'heart_rate_range': heart_rate_range,
            'first_harmonic': first_harmonic,
            'second_harmonic': second_harmonic, # You might want to add this to the plot as well
            'dominant_freq': dominant_freq,

            'x_values_raw_plot': x_filtered_calc, # The actual x_values for the smoothed line (same as uniform_timestamps)
            'ir_data_smoothed_plot': ir_filtered_calc, # The resampled_signal for the smoothed line

            'filterpeaktimes': filtered_peak_times,

            'peakss': filtered_peaks,
            'hpfilter': hpfiltered,
            'xfilter': x_filtered_calc
        }
        

    return os.path.basename(file_path), snr, average_peak_value, plotting_data


# --- Main script to loop through files ---
if __name__ == "__main__":

    results = []
    plot_for_21_middle = None # Variable to store plotting data for '21_middle.csv'

    # Get all CSV files in the specified data_folder
    for filename in os.listdir(data_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(data_folder, filename)
            file_name, snr_value, aph_value, plotting_data = process_ppg_file(file_path)
            results.append({"File": file_name, "SNR": snr_value, "APH": aph_value})
            if file_name == "21_middle.csv" and plotting_data is not None:
                plot_for_21_middle = plotting_data


    ## Processing Summary
    print("\n" + "="*50)
    print("                 Processing Summary")
    print("="*50)
    for result in results:
        snr_str = f"{result['SNR']:.4f}" if result['SNR'] is not None else "N/A"
        aph_str = f"{result['APH']:.4f}" if result['APH'] is not None else "N/A"
        print(f"File: {result['File']:<25} | SNR: {snr_str:<8} | APH: {aph_str:<8}")
    print("="*50)

    if results: # Only save if there are results
        with open('output.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Determine the number of columns needed per group (File + APH + SNR)
            cols_per_file = 1 # File Name, APH, SNR

            # Create a dynamic header
            header_row = []
            num_groups = (len(results) + 4) // 5 # Ceiling division
            for i in range(num_groups):
                header_row.extend([f'File_{i+1}'])
            writer.writerow(header_row)

            # Prepare rows for output
            output_rows = [[] for _ in range(len(results) * 4)]  # 4 rows per pair: SNR, APH, blank, blank

            for col_start in range(0, len(results), 5):  # New column every 5 pairs
                for i in range(5):
                    index = col_start + i
                    if index >= len(results):
                        break
                    result = results[index]
                    snr_value = f"{result['SNR']:.4f}" if result['SNR'] is not None else ""
                    aph_value = f"{result['APH']:.4f}" if result['APH'] is not None else ""

                    row_index = i * 4  # 4 lines per result
                    output_rows[row_index].append(snr_value)        # SNR
                    output_rows[row_index + 1].append(aph_value)    # APH
                    output_rows[row_index + 2].append("")           # blank
                    output_rows[row_index + 3].append("")           # blank

            # Write to CSV
            writer.writerows(output_rows)

        print(f"\nGrouped results saved to file")
    else:
        print("\nNo results to save to CSV.")

    if plot_for_21_middle is not None:
        print("\n--- Generating PSD Plot for 21_middle.csv ---")
        frequencies = plot_for_21_middle['frequencies']
        psd = plot_for_21_middle['psd']
        heart_rate_bpm = plot_for_21_middle['heart_rate_bpm']
        snr = plot_for_21_middle['snr']
        heart_rate_range = plot_for_21_middle['heart_rate_range']
        first_harmonic = plot_for_21_middle['first_harmonic']
        # second_harmonic = plot_for_21_middle['second_harmonic'] # Not used in plot, but available
        dominant_freq = plot_for_21_middle['dominant_freq']


        plt.figure(1, figsize=(12, 6))
        plt.plot(plot_for_21_middle['x_values_raw_plot'], plot_for_21_middle['ir_data_smoothed_plot'], label='IR Smoothed')
        plt.plot(plot_for_21_middle['x_values_raw_plot'], plot_for_21_middle['ir_data_smoothed_plot'], 'o', label='IR Raw', markersize=3)
        plt.title(f"PPG Signal — Smoothed & Raw s")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Sensor Value")
        plt.legend()
        plt.xlim(5, 325)
        plt.grid(True)
        plt.tight_layout()

        plt.figure(2, figsize=(12, 6))
        plt.plot(plot_for_21_middle['xfilter'], plot_for_21_middle['hpfilter'], label='High Pass Detrended', color='purple')
        plt.plot(plot_for_21_middle['filterpeaktimes'], plot_for_21_middle['hpfilter'][plot_for_21_middle['peakss']], 'ro', label="Peaks", markersize = 3)

        plt.title(f"PPG Signal (HP Detrended)")
        plt.xlabel("Time (seconds)")
        plt.ylabel("HP Detrended Value")
        plt.legend()
        plt.xlim(5, 325)
        plt.grid(True)
        plt.tight_layout()


        plt.figure(4, figsize=(12, 6))
        plt.plot(frequencies, psd, label='PSD')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density (V²/Hz)')
        plt.title(f'PPG Power Spectral Density\nDominant HR: {heart_rate_bpm:.1f} BPM | SNR: {snr:.2f}')

        # Add heart rate band shading
        plt.axvspan(heart_rate_range[0], heart_rate_range[1], color='green', alpha=0.1, label='Heart Rate Band')
        plt.axvspan(first_harmonic[0], first_harmonic[1], color='green', alpha=0.1, label='First Harmonic Heart Rate Band')
        # You can add the second harmonic shading here if desired:
        # plt.axvspan(second_harmonic[0], second_harmonic[1], color='green', alpha=0.1, label='Second Harmonic Heart Rate Band')

        if dominant_freq > 0: # Only plot vertical lines if a dominant frequency was found
            plt.axvline(dominant_freq, color='red', linestyle='--', label=f'Dominant Frequency ({dominant_freq:.2f} Hz)')
            if dominant_freq * 2 < 10: # Only plot if within xlim
                plt.axvline(dominant_freq*2, color='purple', linestyle='--', label=f'2nd Harmonic ({dominant_freq*2:.2f} Hz)')
            if dominant_freq * 3 < 10:
                plt.axvline(dominant_freq*3, color='purple', linestyle='--', label=f'3rd Harmonic ({dominant_freq*3:.2f} Hz)')
            if dominant_freq * 4 < 10:
                plt.axvline(dominant_freq*4, color='purple', linestyle='--', label=f'4th Harmonic ({dominant_freq*4:.2f} Hz)')

        plt.xlim(0.5, 10)
        plt.ylim(0, np.max(psd) * 1.1)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show() # Display the plot
