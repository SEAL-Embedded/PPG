import React from 'react';
import {C, FONT} from './theme';
import {liveHr, peaksInWindow, tracePath} from './signal';

const LABEL_W = 168;
const NUM_W = 150;

export const WaveRow: React.FC<{
  label: string;
  sublabel: string;
  color: string;
  values: number[];
  rate: number;
  peaks: number[];
  t: number;
  windowSec: number;
  height: number;
  width: number;
  amp: number;
  strokeWidth?: number;
  refHr?: number | null;
}> = ({
  label,
  sublabel,
  color,
  values,
  rate,
  peaks,
  t,
  windowSec,
  height,
  width,
  amp,
  strokeWidth = 2,
  refHr,
}) => {
  const plotW = width - LABEL_W - NUM_W;
  const tEnd = t;
  const tStart = t - windowSec;

  const d = tracePath(values, rate, tStart, tEnd, plotW, height, amp);
  const marks = peaksInWindow(peaks, values, rate, tStart, tEnd, plotW, height, amp);
  const hr = liveHr(peaks, t);

  // Most recent beat drives the pulse chip — same cue a monitor gives you.
  const lastBeat = marks.length ? marks[marks.length - 1].t : -99;
  const sinceBeat = t - lastBeat;
  const beatGlow = sinceBeat >= 0 && sinceBeat < 0.28 ? 1 - sinceBeat / 0.28 : 0;

  const delta = hr != null && refHr != null ? hr - refHr : null;

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'stretch',
        height,
        background: C.surface,
        borderRadius: 10,
        border: `1px solid ${C.border}`,
        overflow: 'hidden',
      }}
    >
      {/* identity: name in text, colour only as a supporting chip */}
      <div
        style={{
          width: LABEL_W,
          padding: '10px 14px',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          gap: 6,
          borderRight: `1px solid ${C.border}`,
          position: 'relative',
        }}
      >
        <div
          style={{
            position: 'absolute',
            left: 0,
            top: 0,
            bottom: 0,
            width: 4,
            background: color,
            opacity: 0.55 + 0.45 * beatGlow,
          }}
        />
        <div style={{display: 'flex', alignItems: 'center', gap: 9}}>
          <div
            style={{
              width: 11,
              height: 11,
              borderRadius: 3,
              background: color,
              boxShadow: beatGlow ? `0 0 ${10 * beatGlow}px ${color}` : 'none',
            }}
          />
          <div
            style={{
              color: C.ink,
              fontSize: 21,
              fontWeight: 650,
              letterSpacing: 0.2,
            }}
          >
            {label}
          </div>
        </div>
        <div style={{color: C.muted, fontSize: 13, letterSpacing: 0.3}}>{sublabel}</div>
      </div>

      {/* trace */}
      <svg width={plotW} height={height} style={{display: 'block'}}>
        <defs>
          <linearGradient id={`fade-${label}`} x1="0" x2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.12" />
            <stop offset="18%" stopColor={color} stopOpacity="0.85" />
            <stop offset="100%" stopColor={color} stopOpacity="1" />
          </linearGradient>
        </defs>

        <line
          x1={0}
          y1={height / 2}
          x2={plotW}
          y2={height / 2}
          stroke={C.grid}
          strokeWidth={1}
        />
        {[0.25, 0.75].map((f) => (
          <line
            key={f}
            x1={0}
            y1={height * f}
            x2={plotW}
            y2={height * f}
            stroke={C.grid}
            strokeWidth={1}
            strokeDasharray="2 8"
          />
        ))}

        <path
          d={d}
          fill="none"
          stroke={`url(#fade-${label})`}
          strokeWidth={strokeWidth}
          strokeLinejoin="round"
          strokeLinecap="round"
        />

        {marks.map((m, i) => (
          <circle
            key={i}
            cx={m.x}
            cy={m.y}
            r={i === marks.length - 1 ? 4 + 3 * beatGlow : 3.2}
            fill={C.surface}
            stroke={color}
            strokeWidth={2}
          />
        ))}

        {/* leading edge — the "now" write head */}
        <line
          x1={plotW - 1}
          y1={4}
          x2={plotW - 1}
          y2={height - 4}
          stroke={color}
          strokeWidth={2}
          opacity={0.5}
        />
      </svg>

      {/* live numeric */}
      <div
        style={{
          width: NUM_W,
          borderLeft: `1px solid ${C.border}`,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'flex-end',
          justifyContent: 'center',
          padding: '0 16px',
          background: C.surfaceHi,
        }}
      >
        <div
          style={{
            color: C.ink,
            fontSize: 40,
            fontWeight: 700,
            lineHeight: 1,
            fontVariantNumeric: 'tabular-nums',
          }}
        >
          {hr == null ? '--' : Math.round(hr)}
        </div>
        <div style={{color: C.muted, fontSize: 12, letterSpacing: 1.4, marginTop: 3}}>
          BPM
        </div>
        {delta != null ? (
          <div
            style={{
              marginTop: 5,
              fontSize: 13,
              fontVariantNumeric: 'tabular-nums',
              color: Math.abs(delta) <= 5 ? C.good : C.ink2,
            }}
          >
            {delta >= 0 ? '+' : '−'}
            {Math.abs(delta).toFixed(1)} vs ECG
          </div>
        ) : null}
      </div>
    </div>
  );
};
