"""Figure 8 - Bland-Altman SDNN agreement (PPG - ECG, ms) per body site,
points coloured/shaped by Fitzpatrick skin band. Run from the repo root:

    .venv/bin/python figures/make_figure8_bland_altman_sdnn.py

Note for the Discussion: in webapp/analysis.py mean HR is computed from a raw
peak count (60.0 * len(peaks) / duration_s), while SDNN is computed from the
interval series after sqi.hrv_clean.clean_intervals has run. Figures 6 and 8
therefore derive from different interval sets within the same recordings; this
is deliberate and not reconciled here.
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from webapp.analysis import analyze_all_sessions, _skin_group_of  # noqa: E402
from figures._ba_common import plot_ba_grid  # noqa: E402
from figures.make_figure6_bland_altman_hr import SITES, _SITE_ALIAS  # noqa: E402
from figures._style import print_provenance  # noqa: E402


def collect_panels():
    batch = analyze_all_sessions()
    panels = {s: {"site": s, "ref": [], "test": [], "group": []} for s in SITES}
    n_included = 0
    for sess in batch["sessions"]:
        grp = _skin_group_of((sess.get("participant") or {}).get("fitzpatrick"))
        if grp is None:
            continue
        sdnn_e = (sess.get("ecg") or {}).get("sdnn_ms")
        if sdnn_e is None or not np.isfinite(sdnn_e):
            continue
        n_included += 1
        for row in sess.get("results") or []:
            site = _SITE_ALIAS.get(row.get("site"), row.get("site"))
            sdnn_p = row.get("sdnn_ms")
            if site not in panels or sdnn_p is None or not np.isfinite(sdnn_p):
                continue
            panels[site]["ref"].append(float(sdnn_e))
            panels[site]["test"].append(float(sdnn_p))
            panels[site]["group"].append(grp)
    return [panels[s] for s in SITES], n_included


def main():
    out_dir = os.path.join(REPO_ROOT, "figures")
    panels, n_included = collect_panels()
    stats = plot_ba_grid(
        panels,
        ylabel="PPG − ECG SDNN (ms)",
        xlabel="Mean of ECG and PPG SDNN (ms)",
        pdf_path=os.path.join(out_dir, "figure8_bland_altman_sdnn.pdf"),
        png_path=os.path.join(out_dir, "figure8_bland_altman_sdnn.png"),
    )
    print(f"\n{'Site':<10}{'n':>4}{'Bias (ms)':>12}{'LoA lower':>12}{'LoA upper':>12}")
    for st in stats:
        if st["bias"] is None:
            print(f"{st['site']:<10}{st['n']:>4}{'--':>12}{'--':>12}{'--':>12}")
        else:
            print(f"{st['site']:<10}{st['n']:>4}{st['bias']:>12.2f}"
                  f"{st['loa_lower']:>12.2f}{st['loa_upper']:>12.2f}")
    print_provenance(n_included)


if __name__ == "__main__":
    main()
