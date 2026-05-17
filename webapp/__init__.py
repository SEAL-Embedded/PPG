"""
SEAL PPG acquisition + analysis web frontend.

The webapp is a thin FastAPI shell around the existing CLI scripts:
    - PPG_ECG_Full_Unpacking.py  (serial receiver)
    - sqi/ccc.py                  (RR vs PPI agreement)
    - sqi/SSQI_algorithm.py       (skewness SQI)
    - sqi/zcr_sqi.py              (zero-crossing SQI)
    - sqi/bland_altman.py         (standalone Bland-Altman)

It does not re-implement any of that logic — it imports the public
functions and runs them on the session_<ts>/ folders the receiver
already produces.
"""
