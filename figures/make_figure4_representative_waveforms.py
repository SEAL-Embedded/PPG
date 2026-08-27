"""Figure 4 — representative simultaneous ECG + band-pass filtered PPG.

Recreates the committed figure4_representative_waveforms.{pdf,png} from the
live pipeline so the layout can be edited. The session and 10-s window were
identified by matching the original figure: SESSION is the unique batch
session whose per-site SSQIs equal the original's annotation boxes
(Fitzpatrick I participant), and START_S reproduces its 15-R-peak pattern.

Run from the repo root:  .venv/bin/python figures/make_figure4_representative_waveforms.py
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from webapp.analysis import analyze_session, ecg_detail, load_session_signals
from figures._style import print_provenance

SESSION = "session_20260714_142339"
# The batch (use_saved_windows=True) crops this session to its saved best
# window; the SSQI annotations only match the manuscript when the same crop
# is applied here.
CROP = {"start_s": 50.0, "end_s": 350.0}
START_S = 253.02  # window start (s since session t0); axis shows 0-10 s
DUR_S = 10.0

ECG_COLOR = "#1b3a5c"
PPG_COLOR = "#c92a2a"
# (site key in results, display label) in manuscript row order.
ROWS = [("finger", "Finger"), ("wrist", "Wrist"), ("earlobe", "Earlobe"),
        ("forehead", "Temple"), ("shoulder", "Shoulder")]
_MM = 1 / 25.4


def _slice(t, y, lo, hi):
    t, y = np.asarray(t, float), np.asarray(y, float)
    sel = (t >= lo) & (t <= hi)
    return t[sel] - lo, y[sel]


def main():
    res = analyze_session(SESSION, **CROP)
    sig = load_session_signals(SESSION, max_points=10**9, **CROP)
    ecg = ecg_detail(SESSION, **CROP)
    by_site = {r["site"]: r for r in res["results"]}
    by_ch = {c["channel"]: c for c in sig["channels"]}
    lo, hi = START_S, START_S + DUR_S

    plt.rcParams.update({
        "font.size": 8, "axes.linewidth": 0.8,
        "xtick.labelsize": 8, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(len(ROWS) + 1, 1, sharex=True,
                             figsize=(170 * _MM, 150 * _MM), dpi=300)

    # ECG panel with R-peak markers.
    ax = axes[0]
    t, y = _slice(ecg["time_s"], ecg["signal"], lo, hi)
    ax.plot(t, y, color=ECG_COLOR, lw=0.7)
    pk_t, pk_v = _slice(ecg["peak_times_s"], ecg["peak_values"], lo, hi)
    # zorder above the count box so no triangle hides behind its white patch
    ax.plot(pk_t, pk_v + 0.06 * np.ptp(y), "v", color="#d62728", ms=4, zorder=5)
    # Headroom pushes the triangles below the count box's text line.
    ax.set_ylim(top=y.max() + 0.42 * np.ptp(y))
    ax.set_ylabel("ECG", rotation=0, ha="right", va="center",
                  fontweight="bold", fontsize=9, labelpad=12)
    ax.text(0.985, 0.93, f"{len(pk_t)} R-peaks", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, zorder=4,
            bbox=dict(boxstyle="round", fc="white", ec="0.6", lw=0.6))

    # One panel per PPG site: bandpassed trace + detected peaks + SSQI box.
    for ax, (site, label) in zip(axes[1:], ROWS):
        row = by_site[site]
        ch = by_ch[row["channel"]]
        t, y = _slice(ch["time_bp_s"], ch["signal_bp"], lo, hi)
        ax.plot(t, y, color=PPG_COLOR, lw=0.9)
        pk = np.asarray(row["ppg_peak_times_s"], float)
        pk = pk[(pk >= lo) & (pk <= hi)]
        ax.plot(pk - lo, np.interp(pk - lo, t, y), "o", color=ECG_COLOR, ms=3)
        ax.set_ylabel(label, rotation=0, ha="right", va="center",
                      fontweight="bold", fontsize=9, labelpad=12)
        ax.text(0.985, 0.93, f"SSQI = {row['ssqi']:+.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="0.6", lw=0.6))

    for ax in axes:
        ax.set_yticks([])
        ax.set_xlim(0, DUR_S)
        ax.grid(axis="x", color="0.92", lw=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Time (s)")

    fst = (res.get("participant") or {}).get("fitzpatrick")
    fig.suptitle("Simultaneous ECG and band-pass filtered multi-site PPG\n"
                 f"representative participant (Fitzpatrick {'I' * int(fst)}), "
                 f"{DUR_S:.0f} s", fontsize=10, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.962))

    out = ROOT / "figures"
    for ext in ("pdf", "png"):
        fig.savefig(out / f"figure4_representative_waveforms.{ext}")
        print(f"wrote {out / f'figure4_representative_waveforms.{ext}'}")
    print(f"session {SESSION}, window {lo:.1f}-{hi:.1f} s, "
          f"{len(pk_t)} R-peaks in window")
    print_provenance(1)


if __name__ == "__main__":
    main()
