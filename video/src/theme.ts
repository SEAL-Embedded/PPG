// Palette validated with the dataviz validator against the #151517 dark
// surface (order = the vertical stack order, which is what adjacency means
// in this layout): all six checks pass, worst adjacent normal-vision ΔE 19.3.
//   node scripts/validate_palette.js "#0ca30c,#3987e5,#d95926,#9085e9,#d55181,#c98500" --mode dark --surface "#151517"

export const FPS = 30;

export const C = {
  page: '#0d0d0d',
  surface: '#151517',
  surfaceHi: '#1c1c1f',
  ink: '#ffffff',
  ink2: '#c3c2b7',
  muted: '#898781',
  grid: '#242427',
  baseline: '#33333a',
  border: 'rgba(255,255,255,0.10)',
  rec: '#d03b3b',
  good: '#0ca30c',
};

export const ECG_COLOR = '#0ca30c';

// Slot order follows channel order ch0..ch4, which is also the stack order.
export const SITE_COLORS = ['#3987e5', '#d95926', '#9085e9', '#d55181', '#c98500'];

export const FONT =
  'system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif';
