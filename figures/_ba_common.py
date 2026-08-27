"""Shared Bland-Altman panel-grid plotting for manuscript figures 6 and 8.

Metric-agnostic: callers supply per-site reference/test value arrays and a
y-axis label; this module only does the plotting and the bias/LoA maths.
Sign convention everywhere: difference = test - reference (PPG - ECG).
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from figures._style import GROUP_LABEL, SKIN_STYLE

_MM = 1 / 25.4  # mm -> inch


def ba_stats(ref, test):
    """Bland-Altman stats for one site. Arrays must be same length, no NaNs.

    Returns dict with n, bias, sd, loa_lower, loa_upper (bias/LoA are None
    when n < 3 — too few points for a meaningful SD)."""
    ref = np.asarray(ref, dtype=float)
    test = np.asarray(test, dtype=float)
    d = test - ref
    n = len(d)
    if n < 3:
        return {"n": n, "bias": None, "sd": None,
                "loa_lower": None, "loa_upper": None}
    bias = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    return {"n": n, "bias": bias, "sd": sd,
            "loa_lower": bias - 1.96 * sd, "loa_upper": bias + 1.96 * sd}


def plot_ba_grid(panels, ylabel, xlabel, pdf_path, png_path):
    """Draw a Bland-Altman panel grid and save PDF + PNG.

    panels: ordered list of dicts, one per site:
        {"site": str, "ref": 1-D array, "test": 1-D array,
         "group": list of skin-group names ("light"/"medium"/"dark"),
                  same length as ref/test}
    ylabel: y-axis label (e.g. "PPG - ECG HR (bpm)").
    xlabel: x-axis label (e.g. "Mean of ECG and PPG HR (bpm)").

    One shared y-range across all panels (deliberate: per-panel autoscale
    would hide the contrast between tight and scattered sites). Panels with
    n < 3 get points but no bias/LoA lines, with a warning to stdout.

    Returns list of per-site stats dicts (site, n, bias, sd, loa_lower,
    loa_upper) in panel order."""
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
    })

    n_panels = len(panels)
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True,
                             figsize=(170 * _MM, 55 * _MM * nrows))
    axes = np.atleast_1d(axes).ravel()

    # Shared y-range over all differences and LoA lines.
    all_stats = [dict(ba_stats(p["ref"], p["test"]), site=p["site"]) for p in panels]
    all_d = np.concatenate([np.asarray(p["test"], float) - np.asarray(p["ref"], float)
                            for p in panels if len(p["ref"])] or [np.array([0.0])])
    y_ext = [all_d.min(), all_d.max()]
    for st in all_stats:
        if st["bias"] is not None:
            y_ext += [st["loa_lower"], st["loa_upper"]]
    lo, hi = min(y_ext), max(y_ext)
    pad = 0.08 * (hi - lo or 1.0)
    ylim = (lo - pad, hi + pad)

    # Readability note if one site's spread dwarfs another's on the shared axis.
    spans = [np.ptp(np.asarray(p["test"], float) - np.asarray(p["ref"], float))
             for p in panels if len(p["ref"]) >= 2]
    if spans and min(spans) > 0 and max(spans) / min(spans) > 20:
        print("Note: shared y-axis kept by design; the tightest site's spread is "
              f"{max(spans) / min(spans):.0f}x smaller than the widest and may "
              "look flat at this scale.")

    # Panels fill cells in reading order; the legend takes the first spare
    # cell (bottom-right). A panel sitting above a non-panel cell would have
    # its x tick labels suppressed by sharex, so force labelbottom on those.
    n_cells = nrows * ncols
    panel_cells = list(range(n_panels))
    legend_cell = n_panels if n_cells > n_panels else None

    for i, cell in enumerate(panel_cells):
        p, st = panels[i], all_stats[i]
        ax = axes[cell]
        if cell + ncols < n_cells and cell + ncols not in panel_cells:
            ax.tick_params(labelbottom=True)
        ref = np.asarray(p["ref"], float)
        test = np.asarray(p["test"], float)
        mean = (ref + test) / 2.0
        diff = test - ref
        for grp, (colour, marker) in SKIN_STYLE.items():
            sel = [g == grp for g in p["group"]]
            if any(sel):
                ax.scatter(mean[sel], diff[sel], s=14, c=colour, marker=marker,
                           edgecolors="white", linewidths=0.3, zorder=3)
        if st["bias"] is None:
            print(f"Warning: site '{p['site']}' has only n={st['n']} usable "
                  "session(s); bias/LoA lines omitted.")
        else:
            ax.axhline(st["bias"], color="0.2", lw=1.0, zorder=2)
            for y in (st["loa_lower"], st["loa_upper"]):
                ax.axhline(y, color="0.2", lw=0.8, ls="--", zorder=2)
        ax.axhline(0, color="0.75", lw=0.6, zorder=1)
        ax.set_ylim(*ylim)
        ax.set_title(f"({chr(ord('a') + i)}) {p['site'].capitalize()}",
                     fontsize=8, loc="left")
        ax.spines[["top", "right"]].set_visible(False)

    # Hide non-panel cells; shared legend in the reserved top-right cell,
    # else below the grid.
    handles = [Line2D([], [], ls="", marker=m, color=c, markersize=5,
                      markeredgecolor="white", markeredgewidth=0.3,
                      label=GROUP_LABEL[g])
               for g, (c, m) in SKIN_STYLE.items()]
    for j in range(len(axes)):
        if j not in panel_cells:
            axes[j].axis("off")
    if legend_cell is not None:
        axes[legend_cell].legend(handles=handles, loc="center", frameon=False,
                                 title="Skin group")
    else:
        fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                   frameon=False)

    fig.supxlabel(xlabel, fontsize=8)
    fig.supylabel(ylabel, fontsize=8)
    fig.tight_layout()
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return all_stats
