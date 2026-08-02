"""
KSQI — kurtosis signal-quality index for PPG.

Companion to :mod:`sqi.SSQI_algorithm` (skewness) and :mod:`sqi.zcr_sqi`
(zero-crossing rate). Same contract as ``Ssqi``: one number per signal,
computed on the raw (pre-bandpass) amplitude distribution.

Definition
----------
The fourth standardised moment, in the **Pearson** (non-excess) form used by
Elgendi 2016 and by the ``vital_sqi`` toolbox — i.e. *no* ``-3`` correction, so
Gaussian noise scores 3.0 rather than 0.0:

    KSQI = (1/N) * sum_i [ (x_i - mu) / sigma ] ^ 4

with ``mu`` the sample mean and ``sigma`` the *population* standard deviation
(``ddof=0``), matching ``Ssqi`` so the two indices are computed on an identical
normalisation.

Interpretation for PPG
----------------------
Kurtosis measures how much of the amplitude variance comes from rare extreme
samples. For a PPG the reference points are:

    ~1.5   pure sinusoid (theoretical floor for a smooth periodic wave)
    ~2.0   clean pulsatile PPG — Elgendi 2016 measured 2.06 +/- 0.16 on
           adjudicator-rated "excellent" 60 s finger PPG
    3.0    Gaussian — the pulse no longer dominates the amplitude
           distribution, i.e. the trace is noise-shaped
    >>3    leptokurtic: motion spikes / contact transients contribute
           heavy tails
    <1.5   sub-sinusoidal: a bimodal, clipped or saturated trace (rail-to-rail
           ADC, square-ish waveform)

So unlike SSQI (higher = better, monotonic), KSQI is a *two-sided* index: the
clean band sits around 2 and both directions away from it indicate a different
failure mode.

Caveat worth knowing before leaning on it
-----------------------------------------
Elgendi 2016 ranked KSQI *last* of eight PPG SQIs for discriminating rated
quality classes (F1 38.9-73.7% vs SSQI's 74.7-85.8%) — the three quality
classes' KSQI means (2.06 / 1.97 / 2.01) overlap heavily. It is reported here
as an outcome variable alongside SSQI and ZSQI, never as an inclusion filter
(see ``sqi/validity.py`` for why quality metrics must not gate this study).
KSQI does carry information the other two miss — it is the only one of the
three that reacts to *impulsive* artifact (a single large motion spike barely
moves ZSQI and can even raise SSQI) — which is why it earns a column.

References
----------
Elgendi, M. (2016). Optimal signal quality index for photoplethysmogram
signals. Bioengineering, 3(4), 21. doi:10.3390/bioengineering3040021
    -- Equation for KSQI; per-class values in Table 2.

Nguyen, K. et al. (2022). vital_sqi: A Python package for physiological signal
quality control. Frontiers in Physiology, 13, 1020458.
doi:10.3389/fphys.2022.1020458
    -- Same Pearson-form kurtosis SQI, per segment or per beat.

Selvaraj, N. et al. (2011). Statistical approach for the detection of
motion/noise artifacts in photoplethysmogram. IEEE EMBC 2011, 4972-4975.
doi:10.1109/IEMBS.2011.6091232
    -- Kurtosis + Shannon entropy for PPG motion-artifact detection.
"""

import glob

import numpy as np
import pandas as pd


def Ksqi(reading_array):
    """Pearson (non-excess) kurtosis of ``reading_array``.

    Mirrors :func:`sqi.SSQI_algorithm.Ssqi` exactly apart from the exponent,
    including the population sigma (``ddof=0``). Gaussian input -> ~3.0.
    """
    mu = reading_array.mean()
    stddev = reading_array.std(ddof=0)
    return (np.mean(((reading_array - mu) / stddev) ** 4))


if __name__ == "__main__":
    csv_files = glob.glob(r"D:\Study\Lab\fingerTests\*.csv")
    for file in csv_files:
        df = pd.read_csv(file)
        data_np = df.to_numpy()
        score = Ksqi(data_np[:, 1])
        print(file, score)
