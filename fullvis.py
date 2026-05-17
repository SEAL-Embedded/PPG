"""
Combined quick-look plot for one acquisition session.

Reads every `ppg_data_chN.csv` and the `ecg_data.csv` (if present)
sitting next to this script, and plots them on a single figure with a
shared x-axis. The shared axis is the real point: because both signals
were captured on the same Pico's `ticks_us()` clock, you can visually
correlate ECG R-peaks with the PPG pulse that follows a few hundred ms
later (pulse transit time).

ECG goes on top (med-display convention), PPG channels stack below in
channel order. Leads-off ECG samples are NaN-masked and the spans are
shaded red. This is a sanity check, not the full HRV/PTT pipeline.
"""

import csv
import glob
import math
import os
import sys

import matplotlib.pyplot as plt
SCRIPT_DIR = "session_20260516_174059"
ECG_FILENAME = "ecg_data.csv"


def load_ppg(path):
    times_us = []
    samples = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                times_us.append(int(row[0]))
                samples.append(int(row[1]))
            except ValueError:
                # Skip rows where either column failed to parse — e.g. a
                # partially-flushed final row from a hard kill.
                continue
    return times_us, samples


def load_ecg(path):
    times_us = []
    samples = []      # NaN wherever leads were off, so the line has a gap
    leads_off = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                t = int(row[0])
                s = int(row[1])
                lo = int(row[2])
            except ValueError:
                continue
            times_us.append(t)
            leads_off.append(lo)
            samples.append(float("nan") if lo else float(s))
    return times_us, samples, leads_off


def leads_off_spans(times_s, leads_off):
    """Collapse the per-sample leads_off flag into (start_s, end_s) spans
    so we can shade them with one axvspan per disconnect instead of one
    per sample."""
    spans = []
    start = None
    for t, lo in zip(times_s, leads_off):
        if lo and start is None:
            start = t
        elif not lo and start is not None:
            spans.append((start, t))
            start = None
    if start is not None:
        spans.append((start, times_s[-1]))
    return spans


def effective_fs(times_s, n_samples):
    if len(times_s) < 2:
        return float("nan")
    duration_s = times_s[-1] - times_s[0]
    return n_samples / duration_s if duration_s > 0 else float("nan")


def main():
    ppg_paths = sorted(glob.glob(os.path.join(SCRIPT_DIR, "ppg_data_ch*.csv")))
    ecg_path = os.path.join(SCRIPT_DIR, ECG_FILENAME)

    # Load PPG, drop any empty CSVs so they don't get a blank subplot.
    ppg_data = []
    for path in ppg_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        times_us, samples = load_ppg(path)
        if times_us:
            ppg_data.append((name, times_us, samples))

    ecg = None
    if os.path.exists(ecg_path):
        times_us, samples, leads_off = load_ecg(ecg_path)
        if times_us:
            ecg = (times_us, samples, leads_off)

    if not ppg_data and ecg is None:
        print("No ppg_data_ch*.csv or ecg_data.csv with data found next to this script.")
        return 1

    # Common t0 across every signal so the shared x-axis lines ECG and
    # PPG up correctly. Both files were timestamped from the same Pico
    # ticks_us() clock, so this is meaningful.
    first_timestamps = [t[0] for (_, t, _) in ppg_data]
    if ecg is not None:
        first_timestamps.append(ecg[0][0])
    t0 = min(first_timestamps)

    n_rows = len(ppg_data) + (1 if ecg is not None else 0)
    fig, axes = plt.subplots(
        n_rows, 1, sharex=True, figsize=(12, 2.3 * n_rows)
    )
    if n_rows == 1:
        axes = [axes]

    row = 0

    # ECG on top, following the conventional med-display layout.
    if ecg is not None:
        times_us, samples, leads_off = ecg
        ax = axes[row]
        row += 1
        times_s = [(t - t0) / 1e6 for t in times_us]
        fs = effective_fs(times_s, len(samples))
        valid = sum(1 for s in samples if not math.isnan(s))
        off = len(samples) - valid

        ax.plot(times_s, samples, linewidth=0.7, color="C3")
        for start_s, end_s in leads_off_spans(times_s, leads_off):
            ax.axvspan(start_s, end_s, color="red", alpha=0.15)
        ax.set_title(
            f"ECG — {len(samples)} samples "
            f"({valid} valid, {off} leads-off) @ {fs:.1f} Hz"
        )
        ax.set_ylabel("ECG (ADC)")
        ax.grid(True, alpha=0.3)

    # PPG channels below, in channel order.
    for (name, times_us, samples) in ppg_data:
        ax = axes[row]
        row += 1
        times_s = [(t - t0) / 1e6 for t in times_us]
        fs = effective_fs(times_s, len(samples))
        ax.plot(times_s, samples, linewidth=0.7)
        ax.set_title(f"{name} — {len(samples)} samples @ {fs:.1f} Hz")
        ax.set_ylabel("PPG (red)")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s, relative to earliest sample across all signals)")
    fig.suptitle("Combined PPG + ECG quick-look")
    fig.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())