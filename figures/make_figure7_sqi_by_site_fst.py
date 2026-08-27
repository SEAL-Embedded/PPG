"""Figure 7 — SQI by body site, stratified by Fitzpatrick skin-tone band.

Regenerates figures/figure7_sqi_by_site_fst.{pdf,png} from the live pipeline
(webapp.analysis.analyze_all_sessions with each session's saved best window).
Boxes span the IQR with the median marked; individual recordings overlaid as
points. Participant counts are printed to stdout, not drawn in the figure.

Run from the repo root:  .venv/bin/python figures/make_figure7_sqi_by_site_fst.py
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from webapp.analysis import _skin_group_of, analyze_all_sessions
from figures._style import (GROUP_COLOR, GROUP_LABEL, GROUP_MARKER,
                            GROUP_ORDER, print_provenance)

# (band name from _skin_group_of, legend label, colour, marker) — all from
# the shared manuscript-wide scheme in figures/_style.py.
GROUPS = [(g, GROUP_LABEL[g], GROUP_COLOR[g], GROUP_MARKER[g])
          for g in GROUP_ORDER]
SITES = [  # (site label in the data, display label)
    ("finger", "Finger"), ("wrist", "Wrist"), ("earlobe", "Earlobe"),
    ("forehead", "Temple"), ("shoulder", "Shoulder"),
]
# (row key, panel letter, panel title, subtitle or None, y-axis label).
# KSQI carries no better/worse annotation: in this dataset KSQI orders the
# sites inversely to SSQI, so a "higher = better" tag would assert the
# opposite of the result.
METRICS = [
    ("ssqi",      "(a)", "Skewness SQI (SSQI)",       "higher = better",
     "SSQI (dimensionless)"),
    ("ksqi",      "(b)", "Kurtosis SQI (KSQI)",       None,
     "KSQI (dimensionless)"),
    ("zsqi_mean", "(c)", "Zero-crossing rate (ZCR)",  None,
     "ZCR (crossings per sample)"),
]
# Axis clipping, evaluated per panel and per tail. A candidate bound sits
# K_IQR pooled IQRs beyond the most extreme group quartile, so the boxes
# (the bulk of the distribution) occupy most of the vertical space; the clip
# is applied only if it shrinks the axis range by at least CLIP_MIN_SHRINK
# vs. the unclipped range — otherwise that tail renders unclipped with no
# boundary markers or annotation. Clipped points are clamped to the limit
# and drawn as hollow triangles, not dropped.
K_IQR = 1.5
CLIP_MIN_SHRINK = 0.25


def collect():
    batch = analyze_all_sessions(use_saved_windows=True)
    counts = {g: 0 for g, _, _, _ in GROUPS}
    ungraded = 0
    # data[metric][site][group] -> list of values
    data = {m: {s: {g: [] for g, _, _, _ in GROUPS} for s, _ in SITES}
            for m, *_ in METRICS}
    for sess in batch["sessions"]:
        grp = _skin_group_of((sess.get("participant") or {}).get("fitzpatrick"))
        if grp is None:
            ungraded += 1
            continue
        counts[grp] += 1
        for row in sess.get("results", []):
            site = row.get("site")
            if site not in data["ssqi"]:
                continue
            for m, *_ in METRICS:
                v = row.get(m)
                if v is not None and np.isfinite(v):
                    data[m][site][grp].append(float(v))
    return data, counts, ungraded, len(batch["sessions"])


def draw(data):
    plt.rcParams.update({
        "font.size": 8, "axes.linewidth": 0.6,
        "xtick.labelsize": 8, "ytick.labelsize": 7.5,
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(170 / 25.4, 2.9), dpi=300)
    rng = np.random.default_rng(7)
    offsets = [-0.26, 0.0, 0.26]
    box_w = 0.20

    offscale = []  # (metric, site, group, value) for the stdout summary
    for ax, (metric, letter, title, subtitle, ylabel) in zip(axes, METRICS):
        pooled_list = [np.asarray(data[metric][s][g], float)
                       for s, _ in SITES for g, _, _, _ in GROUPS
                       if data[metric][s][g]]
        pooled = np.concatenate(pooled_list) if pooled_list else np.array([0.0])
        group_q1, group_q3 = [], []
        for s, _ in SITES:
            for g, _, _, _ in GROUPS:
                if data[metric][s][g]:
                    q1, q3 = np.percentile(data[metric][s][g], [25, 75])
                    group_q1.append(q1)
                    group_q3.append(q3)
        iqr = float(np.subtract(*np.percentile(pooled, [75, 25]))) or 1.0
        full_lo, full_hi = float(pooled.min()), float(pooled.max())
        full_span = (full_hi - full_lo) or 1.0
        cand_lo = (min(group_q1) - K_IQR * iqr) if group_q1 else full_lo
        cand_hi = (max(group_q3) + K_IQR * iqr) if group_q3 else full_hi
        lo_b = cand_lo if (cand_lo - full_lo) / full_span >= CLIP_MIN_SHRINK \
            else full_lo
        hi_b = cand_hi if (full_hi - cand_hi) / full_span >= CLIP_MIN_SHRINK \
            else full_hi
        n_below = n_above = 0
        for i, (site, _) in enumerate(SITES):
            for (grp, _, color, marker), off in zip(GROUPS, offsets):
                vals = np.asarray(data[metric][site][grp])
                if vals.size == 0:
                    continue
                x = i + off
                # Boxes always use the full, unclamped values.
                q1, med, q3 = np.percentile(vals, [25, 50, 75])
                ax.add_patch(Rectangle((x - box_w / 2, q1), box_w, q3 - q1,
                                       facecolor=color, alpha=0.25,
                                       edgecolor=color, linewidth=0.9, zorder=2))
                ax.hlines(med, x - box_w / 2, x + box_w / 2,
                          color=color, linewidth=1.4, zorder=3)
                jit = rng.uniform(-0.055, 0.055, size=vals.size)
                below, above = vals < lo_b, vals > hi_b
                in_range = ~(below | above)
                ax.scatter(x + jit[in_range], vals[in_range], s=11,
                           marker=marker, color=color, edgecolors="white",
                           linewidths=0.4, alpha=0.9, zorder=4)
                # Off-scale points: clamped to the axis limit as hollow
                # triangles pointing off the axis, keeping x and colour.
                for sel, bound, mk in ((below, lo_b, "v"), (above, hi_b, "^")):
                    if sel.any():
                        ax.scatter(x + jit[sel], np.full(sel.sum(), bound),
                                   s=16, marker=mk, facecolors="none",
                                   edgecolors=color, linewidths=0.8, zorder=5)
                        offscale.extend((metric, site, grp, float(v))
                                        for v in vals[sel])
                n_below += int(below.sum())
                n_above += int(above.sum())
        # Wider pad on annotated sides: the count text lives in the margin
        # band beyond the clamp line, where no point can sit, so it can
        # never collide with data.
        span = float(hi_b - lo_b) or 1.0
        ax.set_ylim(lo_b - (0.12 if n_below else 0.05) * span,
                    hi_b + (0.12 if n_above else 0.05) * span)
        if n_above:
            ax.text(0.02, 0.97, f"{n_above} off-scale (>{hi_b:.2g})",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=6.5, color="0.45")
        if n_below:
            ax.text(0.02, 0.03, f"{n_below} off-scale (<{lo_b:.2g})",
                    transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=6.5, color="0.45")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xticks(range(len(SITES)))
        ax.set_xticklabels([lbl for _, lbl in SITES], rotation=25, ha="right")
        ax.set_xlim(-0.6, len(SITES) - 0.4)
        ax.set_title(title, fontsize=8.5, pad=12)
        ax.text(-0.02, 1.13, letter, transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top", ha="right")
        if subtitle:
            ax.text(0.5, 1.015, subtitle, transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=7,
                    style="italic", color="0.45")
        ax.grid(axis="y", linewidth=0.4, color="0.9", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    handles = [Line2D([], [], linestyle="none", marker=m, markersize=5,
                      markerfacecolor=c, markeredgecolor=c, label=lbl)
               for _, lbl, c, m in GROUPS]
    fig.legend(handles=handles, loc="lower center", ncols=3, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig, offscale


def main():
    data, counts, ungraded, n_analyzed = collect()
    print(f"Sessions analyzed: {n_analyzed}")
    for grp, label, _, _ in GROUPS:
        print(f"  {label}: n = {counts[grp]}")
    print(f"  ungraded (no FST, excluded): n = {ungraded}")
    print(f"  total graded: n = {sum(counts.values())}")

    print("Box glyph: each box spans the interquartile range (25th-75th "
          "percentile); the horizontal line is the median; no whiskers are "
          "drawn (deliberate box-only design); every individual recording "
          "is overlaid as a jittered point.")

    fig, offscale = draw(data)
    print(f"Off-scale points ({len(offscale)}, clamped to the axis limit and "
          "drawn as hollow triangles):")
    if offscale:
        site_label = dict(SITES)
        for metric, site, grp, v in offscale:
            print(f"  {metric} / {site_label[site]} / {grp}: {v:.4g}")
    else:
        print("  none")
    out = ROOT / "figures"
    for ext in ("pdf", "png"):
        fig.savefig(out / f"figure7_sqi_by_site_fst.{ext}")
        print(f"wrote {out / f'figure7_sqi_by_site_fst.{ext}'}")
    print_provenance(sum(counts.values()))


if __name__ == "__main__":
    main()
