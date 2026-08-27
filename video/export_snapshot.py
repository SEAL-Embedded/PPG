"""Render a one-page metrics snapshot for a session, styled like the video.

Reads the cached analysis.json (HR, SDNN, LF/HF, ZSQI are already there) and
adds the two metrics the pipeline does not persist:

  RMSSD  — sqrt(mean(diff(NN)^2)) on the same cleaned NN series SDNN uses.
  KSQI   — Elgendi 2016 kurtosis SQI, mean(((x-mu)/sigma)^4), computed on the
           outlier-removed *unfiltered* grid, same signal SSQI runs on.

    python video/export_snapshot.py [session_name] [--selfcheck]
"""
import json, os, sys, textwrap

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

from webapp import analysis, sessions
from sqi.hrv_clean import clean_intervals

NAME = "session_20260715_164725"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "snapshot.png")

# video/src/theme.ts
PAGE, SURFACE, INK, INK2, MUTED = "#0d0d0d", "#151517", "#ffffff", "#c3c2b7", "#898781"
BORDER, ECG_COLOR = (1, 1, 1, 0.10), "#0ca30c"
SITE_COLORS = ["#3987e5", "#d95926", "#9085e9", "#d55181", "#c98500"]


def hrv_time_domain(peak_times_s):
    """SDNN and RMSSD (ms) from peak times, on the cleaned NN series."""
    t = np.asarray(peak_times_s, dtype=float)
    if len(t) < 3:
        return float("nan"), float("nan")
    intervals_ms = np.diff(t) * 1000.0
    nn, _, _ = clean_intervals(intervals_ms, t[1:])
    if len(nn) < 2:
        return float("nan"), float("nan")
    return float(np.std(nn, ddof=1)), float(np.sqrt(np.mean(np.diff(nn) ** 2)))


def ksqi(sig):
    """Kurtosis SQI on the same preprocessed signal SSQI uses."""
    std = float(np.std(sig, ddof=0))
    if len(sig) < 2 or std == 0.0:
        return float("nan")
    return float(np.mean(((sig - sig.mean()) / std) ** 4))


def channel_ksqi(sdir, ch):
    ts_ms, sig = analysis.load_ppg(os.path.join(sdir, f"ppg_data_ch{ch}.csv"))
    ts_ms, sig = analysis._drop_non_monotonic(ts_ms, sig)
    _, grid_sig = analysis._resample_uniform(ts_ms, sig, analysis.infer_fs(ts_ms))
    return ksqi(analysis._remove_outliers(grid_sig))


def collect(name):
    sdir = sessions.session_path(name)
    with open(os.path.join(sdir, "analysis.json")) as f:
        a = json.load(f)

    ecg = a["ecg"]
    sdnn, rmssd = hrv_time_domain(ecg["peak_times_s"])
    rows = [{
        "label": "ECG", "sub": "reference · lead II", "color": ECG_COLOR,
        "hr": ecg["mean_hr_bpm"], "sdnn": sdnn, "rmssd": rmssd,
        "lf": ecg["lf_power_ms2"], "hf": ecg["hf_power_ms2"],
        "ratio": ecg["lf_hf_ratio"], "zsqi": None, "ksqi": None,
    }]
    for r in a["results"]:
        ch = r["channel"]
        sdnn, rmssd = hrv_time_domain(r["ppg_peak_times_s"])
        rows.append({
            "label": r["site"], "sub": f"PPG ch{ch} · MAX30102",
            "color": SITE_COLORS[ch % len(SITE_COLORS)],
            "hr": r["mean_hr_bpm"], "sdnn": sdnn, "rmssd": rmssd,
            "lf": r["lf_power_ms2"], "hf": r["hf_power_ms2"],
            "ratio": r["lf_hf_ratio"], "zsqi": r["zsqi_mean"],
            "ksqi": channel_ksqi(sdir, ch),
            "json_sdnn": r["sdnn_ms"],
        })
    a["session"] = name
    return a, rows


COLUMNS = [
    ("HR", "BPM", "hr", "{:.1f}"),
    ("SDNN", "ms", "sdnn", "{:.1f}"),
    ("RMSSD", "ms", "rmssd", "{:.1f}"),
    ("LF POWER", "ms²", "lf", "{:,.0f}"),
    ("HF POWER", "ms²", "hf", "{:,.0f}"),
    ("LF/HF", "ratio", "ratio", "{:.2f}"),
    ("ZSQI", "zero-cross", "zsqi", "{:.3f}"),
    ("KSQI", "kurtosis", "ksqi", "{:.2f}"),
]


def render(a, rows, out):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=PAGE)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    def card(x, y, w, h):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=SURFACE, edgecolor=BORDER,
                               linewidth=1.2, zorder=1))

    ax.text(0.026, 0.945, "SEAL PPG", color=INK, fontsize=27, fontweight="bold")
    ax.text(0.158, 0.947, "Session metrics snapshot", color=MUTED, fontsize=15)
    # Participant identity is masked — this frame is shareable as-is.
    ax.text(0.974, 0.947, "*" * len(a["participant"]["participant_id"]),
            color=INK, fontsize=19, fontweight="bold", ha="right")

    # Column geometry: label block on the left, metrics evenly across the rest.
    # Headers and values share one centre per column so they read as a stack.
    x0, x1 = 0.026, 0.974
    label_w = 0.175
    col_x = np.linspace(x0 + label_w, x1, len(COLUMNS) + 1)[:-1]
    col_w = col_x[1] - col_x[0]
    centres = col_x + col_w * 0.5

    top, row_h, gap = 0.868, 0.110, 0.011
    for i, (title, unit, _, _) in enumerate(COLUMNS):
        ax.text(centres[i], top + 0.021, title.upper(), color=MUTED, fontsize=11.5,
                ha="center", fontweight="bold")
        ax.text(centres[i], top + 0.001, unit, color="#5f5e59", fontsize=10, ha="center")

    y = top - row_h
    for row in rows:
        card(x0, y, x1 - x0, row_h)
        ax.add_patch(Rectangle((x0, y), 0.0045, row_h, facecolor=row["color"],
                               edgecolor="none", zorder=2))
        ax.plot(x0 + 0.020, y + row_h * 0.62, "o", markersize=9,
                color=row["color"], zorder=2)
        ax.text(x0 + 0.031, y + row_h * 0.555, row["label"], color=INK,
                fontsize=20, fontweight="bold", zorder=2)
        ax.text(x0 + 0.031, y + row_h * 0.245, row["sub"], color=MUTED,
                fontsize=11.5, zorder=2)

        for i, (_, _, key, fmt) in enumerate(COLUMNS):
            v = row[key]
            txt = "—" if v is None or not np.isfinite(v) else fmt.format(v)
            ax.text(centres[i], y + row_h * 0.36, txt, color=INK2 if v is None else INK,
                    fontsize=23, ha="center", zorder=2)
        y -= row_h + gap

    # Wrapped to the table width so the note block reads as one full-bleed
    # paragraph rather than a narrow column under the first few metrics.
    note = (
        "HR is beat-count over the recording span. SDNN and RMSSD run on NN intervals after the [300, 2000] ms range gate and the "
        "Karlsson ±20% local-median rule; LF (0.04–0.15 Hz) and HF (0.15–0.4 Hz) are Welch PSD on that same NN series. ZSQI (mean "
        "windowed zero-crossing rate) and KSQI (kurtosis) are computed on the outlier-removed, unfiltered pulse — not the 0.6–3.3 Hz "
        "band the systolic detector runs on. ECG carries no PPG quality index."
    )
    ny = y + row_h + gap - 0.048
    for line in textwrap.wrap(note, width=205):
        ax.text(x0, ny, line, color=MUTED, fontsize=12)
        ny -= 0.028

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, facecolor=PAGE)
    plt.close(fig)


def selfcheck(rows):
    """Recomputed SDNN must match the pipeline's, i.e. same NN series feeds RMSSD."""
    for row in rows[1:]:
        assert abs(row["sdnn"] - row["json_sdnn"]) < 0.5, (
            f"{row['label']}: SDNN {row['sdnn']:.2f} != analysis.json {row['json_sdnn']:.2f}")
    assert all(np.isfinite(r["ksqi"]) and r["ksqi"] > 0 for r in rows[1:])
    print("selfcheck ok")


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    a, rows = collect(args[0] if args else NAME)
    if "--selfcheck" in sys.argv:
        selfcheck(rows)
    render(a, rows, OUT)
    print(f"wrote {OUT}")
    for r in rows:
        print(f"  {r['label']:<10} HR={r['hr']:.1f}  SDNN={r['sdnn']:.1f}  RMSSD={r['rmssd']:.1f}  "
              f"LF={r['lf']:.0f}  HF={r['hf']:.0f}  LF/HF={r['ratio']:.2f}")
