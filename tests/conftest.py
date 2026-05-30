"""
Shared pytest fixtures for the SEAL PPG test bench.

Strategy
--------
We never touch the real MDPIdata/ folder. Instead, every test that
needs disk state monkeypatches ``webapp.sessions.sessions_root`` to a
``tmp_path`` so the sessions module, the analysis module, the recorder
and the FastAPI endpoints all agree on the same isolated root.

Synthetic signals
-----------------
``make_session`` writes a 30 s ECG + N-channel PPG into one
``session_<ts>/`` folder. The ECG is a 1 Hz cosine (60 BPM) with a
small Gaussian-shaped R-peak comb on top of it so
``sqi.ccc.detect_r_peaks`` finds well-separated peaks. The PPG is the
same cosine, smoothed and delayed by ~200 ms (a realistic PTT), so
``detect_ppg_peaks`` returns peaks that ``match_intervals`` happily
pairs with the ECG R-peaks. Sample rate is 200 Hz to keep tests fast
while still beating filtfilt's padlen safety check.
"""

import os
import sys

import numpy as np
import pytest

# Make the repo root importable as a package root so ``import webapp`` /
# ``import sqi`` work regardless of where pytest is launched from.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ── Synthetic signal builders ───────────────────────────────────────────────

def _beat_times(duration_s=30.0, hr_bpm=60.0, hrv_ms=15.0, seed=42):
    """Generate beat times with realistic millisecond-level RR jitter so
    the matched RR/PPI arrays have non-zero variance — otherwise
    compute_ccc divides by zero and Pearson is NaN."""
    rng = np.random.default_rng(seed)
    period_s = 60.0 / hr_bpm
    times = []
    t = period_s
    while t < duration_s:
        times.append(t)
        t += period_s + (hrv_ms / 1000.0) * rng.standard_normal()
    return np.array(times)


def _ecg_signal(duration_s=30.0, fs=200.0, hr_bpm=60.0, noise=0.01,
                 seed=42):
    """Build a synthetic ECG: slow baseline cosine + sharp Gaussian
    R-peaks at the requested heart-rate (with realistic jitter so
    interval statistics are non-degenerate). detect_r_peaks bandpasses
    0.5-40 Hz with auto-flip and a 90th-percentile-based height
    threshold; these peaks clear that easily."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    baseline = 0.05 * np.cos(2 * np.pi * 0.3 * t)
    sig = baseline + noise * rng.standard_normal(n)

    peak_times = _beat_times(duration_s=duration_s, hr_bpm=hr_bpm, seed=seed)
    sigma_s = 0.012
    for pt in peak_times:
        sig += 1.0 * np.exp(-((t - pt) ** 2) / (2 * sigma_s ** 2))
    return t, sig


def _ppg_signal(duration_s=30.0, fs=200.0, hr_bpm=60.0, ptt_ms=200.0,
                 amp=1.0, noise=0.005, seed=42):
    """Build a synthetic PPG: a smooth Gaussian hump per beat, delayed
    ``ptt_ms`` after the corresponding ECG R-peak. Sharing the seed
    with the ECG keeps the per-beat RR/PPI relationship 1-to-1."""
    rng = np.random.default_rng(seed + 1)
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    peak_times = _beat_times(duration_s=duration_s, hr_bpm=hr_bpm,
                              seed=seed) + (ptt_ms / 1000.0)
    sig = np.zeros_like(t)
    sigma_s = 0.15
    for pt in peak_times:
        sig += amp * np.exp(-((t - pt) ** 2) / (2 * sigma_s ** 2))
    sig += noise * rng.standard_normal(n)
    return t, sig


def _write_csv(path, ts_us, values, extra_col=None):
    """Write the headerless CSV layout the receiver uses:
    col0=ts_us (int), col1=sample (float), optional col2."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        if extra_col is None:
            for ts, v in zip(ts_us, values):
                f.write(f"{int(ts)},{float(v)}\n")
        else:
            for ts, v, e in zip(ts_us, values, extra_col):
                f.write(f"{int(ts)},{float(v)},{int(e)}\n")


def write_synthetic_session(session_dir, n_channels=3, duration_s=30.0,
                             fs=200.0, hr_bpm=60.0):
    """Materialise ``session_<ts>/ecg_data.csv`` and N PPG CSVs into
    ``session_dir`` (which must already exist). Returns metadata about
    what was written so tests can assert against expected counts."""
    os.makedirs(session_dir, exist_ok=True)
    t, ecg = _ecg_signal(duration_s=duration_s, fs=fs, hr_bpm=hr_bpm)
    # Timestamps are in microseconds in the file (the loader converts
    # to ms by /1000). Use a non-zero origin so we exercise the t0
    # subtraction.
    t0_us = 1_000_000_000
    ts_us = (t0_us + t * 1e6).astype(np.int64)

    leads_off = np.zeros_like(t, dtype=int)
    _write_csv(os.path.join(session_dir, "ecg_data.csv"),
               ts_us, ecg, extra_col=leads_off)

    for ch in range(n_channels):
        _, ppg = _ppg_signal(duration_s=duration_s, fs=fs, hr_bpm=hr_bpm,
                              ptt_ms=200.0 + 20.0 * ch, seed=100 + ch)
        _write_csv(os.path.join(session_dir, f"ppg_data_ch{ch}.csv"),
                   ts_us, ppg)

    return {
        "session_dir": session_dir,
        "duration_s": duration_s,
        "fs": fs,
        "n_ecg_samples": len(t),
        "n_channels": n_channels,
        "expected_r_peaks_approx": int(duration_s * hr_bpm / 60.0),
    }


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def isolated_sessions_root(tmp_path, monkeypatch):
    """Redirect every consumer of ``webapp.sessions.sessions_root`` to a
    pristine tmp_path. This is the single hook that isolates the
    sessions, analysis, recorder and API tests from MDPIdata/."""
    root = tmp_path / "MDPIdata"
    root.mkdir()
    monkeypatch.setattr("webapp.sessions.sessions_root",
                        lambda: str(root))
    return str(root)


@pytest.fixture
def synth_session(isolated_sessions_root):
    """Create one synthetic session inside the isolated root.

    Returns ``(session_name, session_dir, meta_dict)``.
    """
    name = "session_20260101_120000"
    sdir = os.path.join(isolated_sessions_root, name)
    meta = write_synthetic_session(sdir)
    return name, sdir, meta


@pytest.fixture
def synth_session_with_metadata(synth_session):
    """A synthetic session that already has a participant.json with
    a non-default channel-site map so we can exercise the merge logic."""
    import json
    from webapp import sessions as _sessions
    name, sdir, meta = synth_session
    participant = {
        "participant_id": "P001",
        "fitzpatrick": 3,
        "notes": "synthetic",
        "channel_sites": {"0": "finger", "1": "earlobe", "2": "wrist"},
    }
    with open(os.path.join(sdir, _sessions.PARTICIPANT_FILENAME), "w") as f:
        json.dump(participant, f)
    return name, sdir, meta, participant


@pytest.fixture
def synth_signal_arrays():
    """Raw (ts_ms, signal) arrays the SQI smoke tests need without
    going through the disk loader. Same physiology as the on-disk
    synth session."""
    t, ecg = _ecg_signal()
    _, ppg = _ppg_signal()
    ts_ms = (t * 1000.0).astype(float)
    return {"ts_ms": ts_ms, "ecg": ecg, "ppg": ppg, "fs": 200.0}


# ── Adversarial fixtures for sqi/ccc.py bug fixes ─────────────────────────────

@pytest.fixture
def synth_short_signal():
    """A 1-sample signal — too short for any filtfilt order."""
    return np.array([1.0])


@pytest.fixture
def synth_motion_burst_ecg(synth_signal_arrays):
    """Clean 60 BPM ECG + a 5-s motion burst at +5x amplitude in the middle."""
    ts_ms = synth_signal_arrays["ts_ms"].copy()
    ecg = synth_signal_arrays["ecg"].copy()
    fs = synth_signal_arrays["fs"]
    burst_start = int(12.0 * fs)
    burst_end = int(17.0 * fs)
    rng = np.random.default_rng(99)
    ecg[burst_start:burst_end] += 5.0 * rng.standard_normal(burst_end - burst_start)
    return {"ts_ms": ts_ms, "ecg": ecg, "fs": fs}


@pytest.fixture
def synth_inverted_ecg(synth_signal_arrays):
    """ECG flipped — peaks become troughs."""
    return {
        "ts_ms": synth_signal_arrays["ts_ms"],
        "ecg": -synth_signal_arrays["ecg"],
        "fs": synth_signal_arrays["fs"],
    }


@pytest.fixture
def synth_single_spike_ecg(synth_signal_arrays):
    """Clean ECG plus ONE huge negative spike — must NOT auto-invert the signal."""
    ts_ms = synth_signal_arrays["ts_ms"].copy()
    ecg = synth_signal_arrays["ecg"].copy()
    fs = synth_signal_arrays["fs"]
    spike_idx = int(5.0 * fs)
    ecg[spike_idx] = -10.0 * float(np.max(ecg))
    return {"ts_ms": ts_ms, "ecg": ecg, "fs": fs}


@pytest.fixture
def synth_fast_hr_ecg():
    """180 BPM (3 Hz) synthetic ECG at 200 Hz fs, 30 s."""
    fs = 200.0
    duration_s = 30.0
    hr_bpm = 180.0
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    sig = 0.05 * np.cos(2 * np.pi * 0.3 * t)
    rng = np.random.default_rng(13)
    period_s = 60.0 / hr_bpm
    pt = period_s
    sigma_s = 0.012
    while pt < duration_s:
        sig += 1.0 * np.exp(-((t - pt) ** 2) / (2 * sigma_s ** 2))
        pt += period_s + 0.005 * rng.standard_normal()
    ts_ms = (t * 1000.0).astype(float)
    return {"ts_ms": ts_ms, "ecg": sig, "fs": fs}


@pytest.fixture
def synth_amplitude_drift_ppg(synth_signal_arrays):
    """Clean PPG that doubles in amplitude in the second half."""
    ppg = synth_signal_arrays["ppg"].copy()
    half = len(ppg) // 2
    ppg[half:] = ppg[half:] * 2.0
    return {
        "ts_ms": synth_signal_arrays["ts_ms"],
        "ppg": ppg,
        "fs": synth_signal_arrays["fs"],
    }


@pytest.fixture
def synth_inverted_ppg(synth_signal_arrays):
    """PPG flipped — systolic peaks become troughs."""
    return {
        "ts_ms": synth_signal_arrays["ts_ms"],
        "ppg": -synth_signal_arrays["ppg"],
        "fs": synth_signal_arrays["fs"],
    }
