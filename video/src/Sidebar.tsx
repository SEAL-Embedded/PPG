import React from 'react';
import {C, ECG_COLOR, SITE_COLORS} from './theme';
import {beatsUpTo, SessionData} from './signal';

const Card: React.FC<{title: string; value?: string; children: React.ReactNode}> = ({
  title,
  value,
  children,
}) => (
  <div
    style={{
      background: C.surface,
      border: `1px solid ${C.border}`,
      borderRadius: 10,
      padding: '13px 16px',
    }}
  >
    <div
      style={{
        display: 'flex',
        alignItems: 'baseline',
        justifyContent: 'space-between',
        marginBottom: 9,
      }}
    >
      <span
        style={{color: C.muted, fontSize: 11, letterSpacing: 1.5, fontWeight: 600}}
      >
        {title}
      </span>
      {value ? (
        <span
          style={{
            color: C.ink,
            fontSize: 16,
            fontWeight: 650,
            fontVariantNumeric: 'tabular-nums',
          }}
        >
          {value}
        </span>
      ) : null}
    </div>
    {children}
  </div>
);

const Row: React.FC<{k: string; v: string}> = ({k, v}) => (
  <div style={{display: 'flex', justifyContent: 'space-between', padding: '4px 0'}}>
    <span style={{color: C.muted, fontSize: 14}}>{k}</span>
    <span style={{color: C.ink2, fontSize: 14, fontVariantNumeric: 'tabular-nums'}}>
      {v}
    </span>
  </div>
);

type Scale = {
  toY: (v: number, h: number) => number;
  ticks: number[];
  fmt: (v: number) => string;
};

const linScale = (lo: number, hi: number, ticks: number[]): Scale => ({
  toY: (v, h) => h - ((Math.min(Math.max(v, lo), hi) - lo) / (hi - lo)) * h,
  ticks,
  fmt: (v) => String(v),
});

/** LF/HF spans well under 1 to ~10; a linear axis buries the sub-1 half. */
const logScale = (lo: number, hi: number, ticks: number[]): Scale => {
  const l0 = Math.log10(lo);
  const l1 = Math.log10(hi);
  return {
    toY: (v, h) =>
      h - ((Math.log10(Math.min(Math.max(v, lo), hi)) - l0) / (l1 - l0)) * h,
    ticks,
    fmt: (v) => (v < 1 ? v.toFixed(2) : String(v)),
  };
};

/**
 * Shared mini chart. Draws each stream only up to `t`, so the trace grows as
 * the recording plays. Nulls (window not yet full) break the line.
 */
const MiniChart: React.FC<{
  grid: number[];
  series: {color: string; values: (number | null)[]}[];
  t: number;
  duration: number;
  scale: Scale;
  w: number;
  h: number;
}> = ({grid, series, t, duration, scale, w, h}) => {
  const PAD_Y = 9;
  const ih = h - PAD_Y * 2;
  const yOf = (v: number) => PAD_Y + scale.toY(v, ih);
  return (
  <svg width={w} height={h} style={{display: 'block'}}>
    {scale.ticks.map((v) => {
      const y = yOf(v);
      return (
        <g key={v}>
          <line x1={26} y1={y} x2={w} y2={y} stroke={C.grid} strokeWidth={1} />
          <text x={0} y={y + 3.5} fill={C.muted} fontSize={9.5}>
            {scale.fmt(v)}
          </text>
        </g>
      );
    })}
    {series.map((s, si) => {
      const segs: string[] = [];
      let cur = '';
      for (let i = 0; i < grid.length; i++) {
        const gt = grid[i];
        const v = s.values[i];
        if (gt > t || v == null) {
          if (cur) segs.push(cur);
          cur = '';
          continue;
        }
        const x = 26 + (gt / duration) * (w - 26);
        const y = yOf(v);
        cur += `${cur ? 'L' : 'M'}${x.toFixed(1)} ${y.toFixed(1)}`;
      }
      if (cur) segs.push(cur);
      return (
        <path
          key={si}
          d={segs.join(' ')}
          fill="none"
          stroke={s.color}
          strokeWidth={si === 0 ? 2.2 : 1.5}
          strokeLinejoin="round"
          opacity={si === 0 ? 1 : 0.8}
        />
      );
    })}
    <line
      x1={26 + (Math.min(t, duration) / duration) * (w - 26)}
      y1={2}
      x2={26 + (Math.min(t, duration) / duration) * (w - 26)}
      y2={h - 2}
      stroke={C.ink2}
      strokeWidth={1}
      opacity={0.45}
    />
  </svg>
  );
};

export const Sidebar: React.FC<{data: SessionData; t: number; width: number}> = ({
  data,
  t,
  width,
}) => {
  const frac = Math.min(t / data.duration_s, 1);
  const ecgSamples = Math.round(data.ecg.n_native * frac);
  const ppgSamples = data.channels.reduce(
    (a, c) => a + Math.round(c.n_native * frac),
    0,
  );
  const ecgBeats = beatsUpTo(data.ecg.peaks_s, t);
  const chartW = width - 34;

  const colors = [ECG_COLOR, ...data.channels.map((_, i) => SITE_COLORS[i])];
  const legend = [
    {site: 'ECG', color: ECG_COLOR},
    ...data.channels.map((c, i) => ({site: c.site, color: SITE_COLORS[i]})),
  ];

  const grid = data.hrv.t_s;
  // Index of the newest HRV sample at or before now — drives the headline value.
  let gi = -1;
  for (let i = 0; i < grid.length; i++) if (grid[i] <= t) gi = i;
  const ecgHrv = data.hrv.series[0];
  const curSdnn = gi >= 0 ? ecgHrv.sdnn[gi] : null;
  const curLfhf = gi >= 0 ? ecgHrv.lfhf[gi] : null;

  const hrSeries = data.hrv.series.map((s, i) => ({
    color: colors[i],
    values: s.hr,
  }));

  const ecgHr = gi >= 0 ? ecgHrv.hr[gi] : null;

  return (
    <div style={{width, display: 'flex', flexDirection: 'column', gap: 11}}>
      <Card title="SESSION">
        <div style={{color: C.ink, fontSize: 17, fontWeight: 650, marginBottom: 7}}>
          {data.session.replace('session_', '')}
        </div>
        <Row k="Participant" v={data.participant_id || '—'} />
        <Row k="Fitzpatrick" v={data.fitzpatrick ? `Type ${data.fitzpatrick}` : '—'} />
        <Row k="PPG sites" v={String(data.channels.length)} />
      </Card>

      <Card title="ACQUISITION">
        <Row k="ECG samples" v={ecgSamples.toLocaleString()} />
        <Row k="PPG samples" v={ppgSamples.toLocaleString()} />
        <Row k="R-peaks" v={String(ecgBeats)} />
        <Row k="Sample rate" v={`${data.ecg.fs_native.toFixed(0)} / 400 Hz`} />
      </Card>

      <Card title="HEART RATE · BPM" value={ecgHr == null ? '—' : `${Math.round(ecgHr)}`}>
        <MiniChart
          grid={grid}
          series={hrSeries}
          t={t}
          duration={data.duration_s}
          scale={linScale(45, 75, [45, 55, 65, 75])}
          w={chartW}
          h={125}
        />
      </Card>

      <Card
        title={`HRV SDNN · ${data.hrv.sdnn_window_s}s WINDOW`}
        value={curSdnn == null ? '—' : `${Math.round(curSdnn)} ms`}
      >
        <MiniChart
          grid={grid}
          series={data.hrv.series.map((s, i) => ({
            color: colors[i],
            values: s.sdnn,
          }))}
          t={t}
          duration={data.duration_s}
          scale={linScale(40, 180, [40, 80, 120, 160])}
          w={chartW}
          h={116}
        />
      </Card>

      <Card
        title={`LF/HF RATIO · ${data.hrv.lfhf_window_s}s WINDOW`}
        value={curLfhf == null ? '—' : curLfhf.toFixed(2)}
      >
        <MiniChart
          grid={grid}
          series={data.hrv.series.map((s, i) => ({
            color: colors[i],
            values: s.lfhf,
          }))}
          t={t}
          duration={data.duration_s}
          scale={logScale(0.2, 10, [0.25, 1, 4])}
          w={chartW}
          h={116}
        />
      </Card>

      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '6px 14px',
          padding: '0 3px',
        }}
      >
        {legend.map((s) => (
          <div key={s.site} style={{display: 'flex', alignItems: 'center', gap: 5}}>
            <div style={{width: 9, height: 9, borderRadius: 2, background: s.color}} />
            <span style={{color: C.ink2, fontSize: 12}}>{s.site}</span>
          </div>
        ))}
      </div>
    </div>
  );
};
