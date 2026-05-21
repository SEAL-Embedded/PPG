"""
Quick-look plot for the multi-PPG recording.

Reads every `ppg_data_chN.csv` sitting next to this script and plots each
channel on its own subplot so you can eyeball that all sensors recorded
something sensible. This is a sanity check, not the full HRV pipeline —
that lives in signal_visualization/ppgvis.py.
"""

import csv
import glob
import os
import sys

import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_channel(path):
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


def main():
    paths = sorted(glob.glob(os.path.join(SCRIPT_DIR, "ppg_data_ch*.csv")))
    if not paths:
        print("No ppg_data_ch*.csv files found.")
        return 1

    fig, axes = plt.subplots(len(paths), 1, sharex=True, figsize=(12, 2.5 * len(paths)))
    if len(paths) == 1:
        axes = [axes]

    for ax, path in zip(axes, paths):
        ch_name = os.path.splitext(os.path.basename(path))[0]
        times_us, samples = load_channel(path)
        if not samples:
            ax.set_title(f"{ch_name} (no data)")
            continue

        t0 = times_us[0]
        times_s = [(t - t0) / 1e6 for t in times_us]
        ax.plot(times_s, samples, linewidth=0.7)
        ax.set_title(f"{ch_name} — {len(samples)} samples")
        ax.set_ylabel("Sample")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s, relative to first sample)")
    fig.suptitle("Multi-PPG quick-look")
    fig.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
