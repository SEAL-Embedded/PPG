"""Figure 6 - Bland-Altman heart-rate agreement (PPG - ECG) per body site,
points coloured/shaped by Fitzpatrick skin band. Run from the repo root:

    .venv/bin/python figures/make_figure6_bland_altman_hr.py
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from webapp.analysis import analyze_all_sessions, _skin_group_of  # noqa: E402
from figures._ba_common import plot_ba_grid  # noqa: E402
from figures._style import print_provenance  # noqa: E402

SITES = ["finger", "wrist", "earlobe", "temple", "shoulder"]
# Sessions label the head-mounted sensor "forehead"; the manuscript calls the
# same position "temple" (see _SITE_ORDER comment in webapp/analysis.py).
_SITE_ALIAS = {"forehead": "temple"}


def collect_panels():
    batch = analyze_all_sessions()
    panels = {s: {"site": s, "ref": [], "test": [], "group": []} for s in SITES}
    n_included = 0
    for sess in batch["sessions"]:
        grp = _skin_group_of((sess.get("participant") or {}).get("fitzpatrick"))
        if grp is None:
            continue
        hr_e = (sess.get("ecg") or {}).get("mean_hr_bpm")
        if hr_e is None or not np.isfinite(hr_e):
            continue
        n_included += 1
        for row in sess.get("results") or []:
            site = _SITE_ALIAS.get(row.get("site"), row.get("site"))
            hr_p = row.get("mean_hr_bpm")
            if site not in panels or hr_p is None or not np.isfinite(hr_p):
                continue
            panels[site]["ref"].append(float(hr_e))
            panels[site]["test"].append(float(hr_p))
            panels[site]["group"].append(grp)
    return [panels[s] for s in SITES], n_included


def main():
    out_dir = os.path.join(REPO_ROOT, "figures")
    panels, n_included = collect_panels()
    stats = plot_ba_grid(
        panels,
        ylabel="PPG − ECG heart rate (bpm)",
        xlabel="Mean of ECG and PPG heart rate (bpm)",
        pdf_path=os.path.join(out_dir, "figure6_bland_altman_hr.pdf"),
        png_path=os.path.join(out_dir, "figure6_bland_altman_hr.png"),
    )
    print(f"\n{'Site':<10}{'n':>4}{'Bias (bpm)':>12}{'LoA lower':>12}{'LoA upper':>12}")
    for st in stats:
        if st["bias"] is None:
            print(f"{st['site']:<10}{st['n']:>4}{'--':>12}{'--':>12}{'--':>12}")
        else:
            print(f"{st['site']:<10}{st['n']:>4}{st['bias']:>12.2f}"
                  f"{st['loa_lower']:>12.2f}{st['loa_upper']:>12.2f}")
    print_provenance(n_included)


if __name__ == "__main__":
    main()
