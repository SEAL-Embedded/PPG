"""DEPRECATED — see webapp/sleepiness.py for the maintained HRV pipeline.

This file's Welch PSD computation has a known scaling × df double-counting
bug (off by ~250×) and is no longer wired into the dashboard. It is kept
only so external scripts that import it fail fast and loudly with a
direct redirect to the canonical pipeline.

For HRV features (SDNN, RMSSD, LF/HF, sample entropy, Poincaré): use
``webapp.sleepiness.compute_hrv_features``. For per-HRV-feature CCC
across the cohort: use ``webapp.sleepiness.analyze_sleepiness`` and read
its ``per_feature`` response key.

Background: this file pre-dated the FastAPI dashboard rewrite. Its
Welch-after-cubic-interpolation pipeline was superseded by the
Lomb-Scargle path in ``webapp/sleepiness.py`` (which is robust to the
non-uniform RR series PPG/ECG recordings produce).
"""

raise ImportError(
    "signal_visualization.ppgvis is deprecated as of 2026-05-27. "
    "Use webapp.sleepiness.compute_hrv_features for HRV features and the "
    "dashboard's Σ page (or webapp.sleepiness.analyze_sleepiness) for "
    "cohort-level HRV agreement analysis."
)
