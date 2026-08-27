import React from 'react';
import {Composition} from 'remotion';
import {LiveAcquisition} from './LiveAcquisition';
import {SessionData} from './signal';
import {FPS} from './theme';
import raw from '../public/session.json';

const data = raw as unknown as SessionData;

export const RemotionRoot: React.FC = () => (
  <Composition
    id="LiveAcquisition"
    component={LiveAcquisition}
    durationInFrames={Math.ceil(data.duration_s * FPS)}
    fps={FPS}
    width={1920}
    height={1080}
    defaultProps={{data}}
  />
);
