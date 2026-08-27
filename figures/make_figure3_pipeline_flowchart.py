"""Figure 3 — ECG and PPG beat-detection pipeline flowchart.

Recreates the committed figure3_pipeline_flowchart.{pdf,png} (whose generator
was never committed) so the layout can be edited. Pure drawing, no data.

Run from the repo root:  .venv/bin/python figures/make_figure3_pipeline_flowchart.py
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from figures._style import print_provenance

NAVY, RED, GREY = "#1b3a5c", "#c92a2a", "#555555"
NAVY_FILL, RED_FILL, GREY_FILL = "#e9eff5", "#fdecec", "#efefef"
_MM = 1 / 25.4

W, H = 170.0, 163.0          # page size, mm
MX, COL_W = 6.0, 76.0        # side margin / branch column width
LX, RX = MX, W - MX - COL_W  # branch column left edges
LCX, RCX = LX + COL_W / 2, RX + COL_W / 2
CX = W / 2

# (title, [sub lines]) per branch box, top to bottom.
ECG_BOXES = [
    ("Polarity orientation",
     ["flip if lead inverted", "(robust P1/P99; ≥1.2 asymmetry guard)"], 15.0),
    ("Relative prominence threshold",
     ["peak stands out by", "0.35 × (P99 − P40) of oriented signal"], 15.0),
    ("Refractory constraint",
     ["0.28 s minimum RR", "(≈214 bpm cap; rejects T-wave)"], 15.0),
]
PPG_BOXES = [
    ("Band-pass filter",
     ["0.5–8.0 Hz zero-phase Butterworth (2nd order)",
      "(after uniform resample + outlier scrub)"], 15.0),
    ("TERMA moving averages",
     ["MA$_{peak}$ = 111 ms   •   MA$_{beat}$ = 667 ms"], 11.5),
    ("Block-of-interest thresholding",
     ["keep runs where MA$_{peak}$ > MA$_{beat}$ + β·mean;",
      "reject blocks narrower than 111 ms"], 15.0),
    ("Doublet filter",
     ["drop peaks < 0.6 × median PPI,",
      "keep the taller of each close pair"], 15.0),
]


def draw_box(ax, x, y, w, h, title, lines, fc, ec, title_fs=9, lw=1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, fc=fc, ec=ec, lw=lw,
                                boxstyle="round,pad=0,rounding_size=1.8"))
    if lines:
        ax.text(x + w / 2, y + 4.2, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color="#222")
        for i, ln in enumerate(lines):
            ax.text(x + w / 2, y + 8.6 + 3.4 * i, ln, ha="center",
                    va="center", fontsize=7, color="#222")
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")


def arrow(ax, x, y0, y1, color):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3,
                                mutation_scale=11, shrinkA=0, shrinkB=0))


def main():
    plt.rcParams.update({"font.family": "sans-serif", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(W * _MM, H * _MM), dpi=300)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.invert_yaxis()
    ax.axis("off")

    ax.text(CX, 6.2, "ECG and PPG beat-detection pipeline", ha="center",
            va="center", fontsize=13, fontweight="bold", color=NAVY)
    ax.text(CX, 11.6, "parallel branches converging on shared window "
            "screening and feature extraction", ha="center", va="center",
            fontsize=8, color="#777")

    # Raw acquisition (full width)
    y = 15.0
    draw_box(ax, MX, y, W - 2 * MX, 13.0, "Raw acquisition",
             ["AD8232 3-lead ECG   +   5 × MAX30102 PPG   "
              "(finger, wrist, earlobe, temple, shoulder)"],
             GREY_FILL, GREY, title_fs=9.5)
    y += 13.0

    # Split connector into the two branches, with ECG / PPG chips.
    ax.plot([CX, CX], [y, y + 3], color=GREY, lw=1.3)
    ax.plot([LCX, RCX], [y + 3, y + 3], color=GREY, lw=1.3)
    arrow(ax, LCX, y + 3, y + 8, NAVY)
    arrow(ax, RCX, y + 3, y + 8, RED)
    for cx, lbl, c in ((LCX, "ECG", NAVY), (RCX, "PPG", RED)):
        ax.add_patch(FancyBboxPatch((cx - 7, y + 2.2), 14, 4.6, fc=c, ec=c,
                                    boxstyle="round,pad=0,rounding_size=1.2"))
        ax.text(cx, y + 4.5, lbl, ha="center", va="center", fontsize=8,
                fontweight="bold", color="white")
    y += 8.0

    # Branch columns
    def column(x, cx, boxes, fill, edge, y0):
        yy = y0
        for i, (title, lines, h) in enumerate(boxes):
            if i:
                arrow(ax, cx, yy, yy + 4, edge)
                yy += 4
            draw_box(ax, x, yy, COL_W, h, title, lines, fill, edge)
            yy += h
        return yy

    l_end = column(LX, LCX, ECG_BOXES, NAVY_FILL, NAVY, y)
    r_end = column(RX, RCX, PPG_BOXES, RED_FILL, RED, y)

    # Interval-series boxes, aligned below the taller branch.
    y = max(l_end, r_end) + 4.0
    arrow(ax, LCX, l_end, y, NAVY)
    arrow(ax, RCX, r_end, y, RED)
    draw_box(ax, LX, y, COL_W, 10.0, "RR interval series", None, NAVY, NAVY)
    draw_box(ax, RX, y, COL_W, 10.0, "PP interval series", None, RED, RED)
    y += 10.0

    # Window screening
    arrow(ax, LCX, y, y + 4, NAVY)
    arrow(ax, RCX, y, y + 4, RED)
    y += 4.0
    draw_box(ax, MX, y, W - 2 * MX, 15.5,
             "Window screening — physiological plausibility "
             "(Orphanidou 2015)",
             ["non-overlapping 10 s windows, ≥4 intervals each; "
              "all three must hold:",
              "①  40 ≤ HR ≤ 180 bpm       ②  no inter-beat "
              "gap > 3 s       ③  max/min interval ratio < 2.2"],
             GREY_FILL, GREY, title_fs=9.5)
    y += 15.5

    # Feature extraction
    arrow(ax, CX, y, y + 4, GREY)
    y += 4.0
    draw_box(ax, MX, y, W - 2 * MX, 15.5, "Feature extraction",
             ["NN cleaning (300–2000 ms gate + Karlsson ±20%)  "
              "→  RR–PPI nearest-neighbour matching",
              "agreement: CCC, ICC(A,1), Bland–Altman (bias, LoA), "
              "RMSE, MAE       HRV: SDNN, LF/HF"],
             GREY_FILL, GREY, title_fs=9.5)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    out = ROOT / "figures"
    for ext in ("pdf", "png"):
        fig.savefig(out / f"figure3_pipeline_flowchart.{ext}")
        print(f"wrote {out / f'figure3_pipeline_flowchart.{ext}'}")
    print_provenance(0)


if __name__ == "__main__":
    main()
