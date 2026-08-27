"""Figure 9 - PPG-to-ECG agreement heatmap: Pearson r per site x metric x
Fitzpatrick skin band, from the stratified batch aggregations.

Run from the repo root:  .venv/bin/python figures/make_figure9_agreement_heatmap.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from webapp.analysis import analyze_all_sessions
from figures._style import GROUP_LABEL, print_provenance

SITES = ["finger", "wrist", "earlobe", "temple", "shoulder"]
# The batch labels the head sensor "forehead"; the manuscript site name is
# "temple". Alias so the manuscript row picks up the data.
SITE_ALIAS = {"forehead": "temple"}
METRICS = [("HR", "hr_per_channel"),
           ("SDNN", "sdnn_per_channel"),
           ("LF/HF", "lfhf_per_channel")]
GROUPS = ["light", "medium", "dark"]
MIN_N = 3

MM = 1 / 25.4  # mm -> inch


def build_matrix(stratified):
    """Return (r_matrix, n_matrix), shape (site, metric*group), NaN = blank."""
    bands = {b["group"]: b for b in stratified}
    r = np.full((len(SITES), len(METRICS) * len(GROUPS)), np.nan)
    n = np.zeros_like(r, dtype=int)
    for mi, (_, key) in enumerate(METRICS):
        for gi, grp in enumerate(GROUPS):
            col = mi * len(GROUPS) + gi
            for row in bands.get(grp, {}).get(key, []):
                site = SITE_ALIAS.get(row.get("site"), row.get("site"))
                if site not in SITES:
                    continue
                si = SITES.index(site)
                n[si, col] = row.get("n_sessions", 0)
                pr = row.get("pearson_r")
                if n[si, col] >= MIN_N and pr is not None and np.isfinite(pr):
                    r[si, col] = float(pr)
    return r, n


def main():
    res = analyze_all_sessions()
    r, n = build_matrix(res["stratified_by_skin"])

    blanked = [(SITES[si], METRICS[ci // 3][0], GROUPS[ci % 3], int(n[si, ci]))
               for si in range(r.shape[0]) for ci in range(r.shape[1])
               if not np.isfinite(r[si, ci])]
    print("Blanked cells (n<%d or undefined r):" % MIN_N)
    if blanked:
        for site, metric, grp, cnt in blanked:
            print(f"  {site} / {metric} / {grp}: n={cnt}")
    else:
        print("  none")

    plt.rcParams.update({"font.size": 8, "font.family": "sans-serif"})
    fig, ax = plt.subplots(figsize=(170 * MM, 72 * MM))

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#e0e0e0")
    masked = np.ma.masked_invalid(r)
    im = ax.imshow(masked, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    # cell annotations, text colour chosen by fill luminance
    for si in range(r.shape[0]):
        for ci in range(r.shape[1]):
            if np.isfinite(r[si, ci]):
                rgba = cmap((r[si, ci] + 1) / 2)
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                ax.text(ci, si, f"{r[si, ci]:.2f}", ha="center", va="center",
                        fontsize=7.5, color="white" if lum < 0.5 else "black")

    # thin white grid between cells, heavier separators between metric groups
    ax.set_xticks(np.arange(-0.5, r.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, r.shape[0]), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    for x in (2.5, 5.5):
        ax.axvline(x, color="white", linewidth=3)
    ax.tick_params(which="both", length=0)

    # Column headers carry the per-group session count so small-n cells
    # (which can show r = 1.00 from a handful of sessions) are visibly
    # less reliable than well-supported ones.
    band_n = {b["group"]: b.get("n_sessions", 0)
              for b in res["stratified_by_skin"]}
    col_labels = [GROUP_LABEL[g].replace(" (", "\n(") + f"\nn = {band_n.get(g, 0)}"
                  for _ in METRICS for g in GROUPS]
    ax.set_xticks(range(r.shape[1]))
    ax.set_xticklabels(col_labels, fontsize=6.5)
    ax.set_yticks(range(len(SITES)))
    ax.set_yticklabels([s.capitalize() for s in SITES])

    # second-level header: metric name centred over its three columns
    for mi, (name, _) in enumerate(METRICS):
        ax.text(mi * 3 + 1, -0.85, name, ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
                        ticks=[-1, -0.5, 0, 0.5, 1])
    cbar.set_label("Pearson r", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_visible(False)

    fig.subplots_adjust(left=0.10, right=0.90, top=0.86, bottom=0.20)
    outdir = os.path.dirname(os.path.abspath(__file__))
    for ext in ("pdf", "png"):
        path = os.path.join(outdir, f"figure9_agreement_heatmap.{ext}")
        fig.savefig(path, dpi=300)
        print("Wrote", path)
    print_provenance(sum(band_n.values()))


if __name__ == "__main__":
    main()
