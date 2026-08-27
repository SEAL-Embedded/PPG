import React from 'react';
import {AbsoluteFill, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, ECG_COLOR, FONT, SITE_COLORS} from './theme';
import {ampFor, fmtClock, liveHr, SessionData} from './signal';
import {WaveRow} from './WaveRow';
import {Sidebar} from './Sidebar';

const WINDOW_SEC = 6;
const PAD = 28;
const SIDEBAR_W = 430;

export const LiveAcquisition: React.FC<{data: SessionData}> = ({data}) => {
  const frame = useCurrentFrame();
  const {fps, width, height} = useVideoConfig();
  const t = frame / fps;

  const headerH = 76;
  const bodyTop = headerH + PAD;
  const bodyH = height - bodyTop - PAD;
  const stackW = width - PAD * 2 - SIDEBAR_W - PAD;

  const rows = 1 + data.channels.length;
  const gap = 10;
  const rowH = (bodyH - gap * (rows - 1)) / rows;

  const ecgHr = liveHr(data.ecg.peaks_s, t);
  const recPulse = 0.55 + 0.45 * Math.sin((frame / fps) * Math.PI * 2);

  return (
    <AbsoluteFill style={{background: C.page, fontFamily: FONT}}>
      {/* header */}
      <div
        style={{
          height: headerH,
          padding: `0 ${PAD}px`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          borderBottom: `1px solid ${C.border}`,
        }}
      >
        <div style={{display: 'flex', alignItems: 'baseline', gap: 16}}>
          <span style={{color: C.ink, fontSize: 25, fontWeight: 700, letterSpacing: 0.2}}>
            SEAL PPG
          </span>
          <span style={{color: C.muted, fontSize: 16}}>
            Multi-site PPG vs ECG acquisition
          </span>
        </div>

        <div style={{display: 'flex', alignItems: 'center', gap: 26}}>
          <div style={{display: 'flex', alignItems: 'center', gap: 9}}>
            <div
              style={{
                width: 13,
                height: 13,
                borderRadius: '50%',
                background: C.rec,
                opacity: recPulse,
                boxShadow: `0 0 ${12 * recPulse}px ${C.rec}`,
              }}
            />
            <span
              style={{
                color: C.rec,
                fontSize: 15,
                fontWeight: 700,
                letterSpacing: 2.2,
              }}
            >
              RECORDING
            </span>
          </div>
          <span
            style={{
              color: C.ink,
              fontSize: 27,
              fontWeight: 650,
              fontVariantNumeric: 'tabular-nums',
            }}
          >
            {fmtClock(t)}
            <span style={{color: C.muted, fontSize: 17, fontWeight: 400}}>
              {' / '}
              {fmtClock(data.duration_s)}
            </span>
          </span>
        </div>
      </div>

      {/* body */}
      <div
        style={{
          position: 'absolute',
          top: bodyTop,
          left: PAD,
          right: PAD,
          height: bodyH,
          display: 'flex',
          gap: PAD,
        }}
      >
        <div
          style={{
            width: stackW,
            display: 'flex',
            flexDirection: 'column',
            gap,
          }}
        >
          <WaveRow
            label="ECG"
            sublabel="reference · lead II"
            color={ECG_COLOR}
            values={data.ecg.values}
            rate={data.ecg.rate}
            peaks={data.ecg.peaks_s}
            t={t}
            windowSec={WINDOW_SEC}
            height={rowH}
            width={stackW}
            amp={ampFor(rowH)}
            strokeWidth={2}
          />
          {data.channels.map((c, i) => (
            <WaveRow
              key={c.ch}
              label={c.site}
              sublabel={`PPG ch${c.ch} · MAX30102`}
              color={SITE_COLORS[i]}
              values={c.values}
              rate={c.rate}
              peaks={c.peaks_s}
              t={t}
              windowSec={WINDOW_SEC}
              height={rowH}
              width={stackW}
              amp={ampFor(rowH)}
              strokeWidth={2}
              refHr={ecgHr}
            />
          ))}
        </div>

        <Sidebar data={data} t={t} width={SIDEBAR_W} />
      </div>
    </AbsoluteFill>
  );
};
