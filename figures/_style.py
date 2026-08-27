"""Single source of truth for manuscript-wide figure styling.

Skin-group label strings, colours, and markers live here so the four figure
scripts cannot drift apart. Also provides the provenance line every figure
script prints, so a figure can be matched to the analysis run that produced
the manuscript tables.
"""

import datetime
import os
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GROUP_ORDER = ["light", "medium", "dark"]
GROUP_LABEL = {
    "light":  "FST I–II (light)",
    "medium": "FST III–IV (medium)",
    "dark":   "FST V–VI (dark)",
}
GROUP_COLOR = {"light": "#E69F00", "medium": "#009E73", "dark": "#0072B2"}
GROUP_MARKER = {"light": "o", "medium": "s", "dark": "^"}
# (colour, marker) per group, the shape the Bland-Altman helper consumes.
SKIN_STYLE = {g: (GROUP_COLOR[g], GROUP_MARKER[g]) for g in GROUP_ORDER}


def print_provenance(n_sessions):
    """Print commit hash, timestamp, and session count to stdout."""
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           cwd=REPO_ROOT, capture_output=True, text=True,
                           timeout=5)
        commit = r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        commit = "unavailable"
    stamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"Provenance: commit={commit}  generated={stamp}  "
          f"sessions_included={n_sessions}")
