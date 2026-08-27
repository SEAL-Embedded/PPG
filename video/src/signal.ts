// Signal helpers shared by the waveform rows and the sidebar.
// Everything is indexed by time: the exported JSON puts each trace on a
// uniform grid, so sample index == t * rate.

export type Channel = {
  ch: number;
  site: string;
  rate: number;
  fs_native: number;
  n_native: number;
  mean_hr_bpm: number;
  ssqi: number;
  ccc: number;
  values: number[];
  peaks_s: number[];
};

export type SessionData = {
  session: string;
  participant_id: string;
  fitzpatrick: number | null;
  duration_s: number;
  ecg: {
    rate: number;
    fs_native: number;
    n_native: number;
    mean_hr_bpm: number;
    values: number[];
    peaks_s: number[];
    leads_off: [number, number][];
  };
  channels: Channel[];
  hrv: {
    t_s: number[];
    sdnn_window_s: number;
    lfhf_window_s: number;
    /** Stream order matches [ECG, ...channels]. */
    series: {
      key: string;
      hr: (number | null)[];
      sdnn: (number | null)[];
      lfhf: (number | null)[];
    }[];
  };
};

/** Max |value| the exporter clips to. Amplitude is derived from this so a
 * clipped sample still lands inside the panel instead of off the top edge,
 * which silently swallowed the R-peak markers. */
export const CLIP = 1.5;

/** Vertical gain for a row of height h, leaving a small margin. */
export const ampFor = (h: number, margin = 8): number => (h / 2 - margin) / CLIP;

/** Number of peaks at or before t. Binary search — called every frame. */
export const beatsUpTo = (peaks: number[], t: number): number => {
  let lo = 0;
  let hi = peaks.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (peaks[mid] <= t) lo = mid + 1;
    else hi = mid;
  }
  return lo;
};

/**
 * Live heart rate the way a bedside monitor derives it: the median of the
 * beat-to-beat intervals inside a trailing window. Median rather than mean so
 * one missed or doubled beat doesn't yank the number around.
 */
export const liveHr = (
  peaks: number[],
  t: number,
  lookback = 20,
): number | null => {
  const end = beatsUpTo(peaks, t);
  const win: number[] = [];
  for (let i = end - 1; i >= 0 && peaks[i] > t - lookback; i--) win.push(peaks[i]);
  if (win.length < 4) return null;
  win.reverse();
  const iv: number[] = [];
  for (let i = 1; i < win.length; i++) iv.push(win[i] - win[i - 1]);
  iv.sort((a, b) => a - b);
  const med = iv[Math.floor(iv.length / 2)];
  return med > 0 ? 60 / med : null;
};

/** SVG path for the slice of `values` covering [tStart, tEnd]. */
export const tracePath = (
  values: number[],
  rate: number,
  tStart: number,
  tEnd: number,
  w: number,
  h: number,
  amp: number,
): string => {
  const i0 = Math.max(0, Math.ceil(tStart * rate));
  const i1 = Math.min(values.length - 1, Math.floor(tEnd * rate));
  if (i1 <= i0) return '';
  const span = tEnd - tStart;
  // Never emit more points than the panel has pixels to show them in.
  const step = Math.max(1, Math.floor((i1 - i0) / Math.max(w, 1)));
  const mid = h / 2;
  let d = '';
  for (let i = i0; i <= i1; i += step) {
    const x = ((i / rate - tStart) / span) * w;
    const y = mid - values[i] * amp;
    d += `${d === '' ? 'M' : 'L'}${x.toFixed(1)} ${y.toFixed(1)}`;
  }
  return d;
};

/** Peaks falling inside the visible window, with their plotted position. */
export const peaksInWindow = (
  peaks: number[],
  values: number[],
  rate: number,
  tStart: number,
  tEnd: number,
  w: number,
  h: number,
  amp: number,
) => {
  const out: {x: number; y: number; t: number}[] = [];
  const span = tEnd - tStart;
  const from = beatsUpTo(peaks, tStart);
  const to = beatsUpTo(peaks, tEnd);
  for (let i = from; i < to; i++) {
    const t = peaks[i];
    const idx = Math.round(t * rate);
    if (idx < 0 || idx >= values.length) continue;
    out.push({
      x: ((t - tStart) / span) * w,
      y: h / 2 - values[idx] * amp,
      t,
    });
  }
  return out;
};

export const fmtClock = (s: number): string => {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
};
