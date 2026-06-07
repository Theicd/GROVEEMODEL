/** Per-frame motion analysis — pixel diff by region, no model inference. */

import { frameDiffScore, type FrameSample } from "./sceneChangeDetector";

export type MotionSnapshot = {
  motionLevel: number;
  upperMotion: number;
  lowerMotion: number;
  armMovement: boolean;
  waveLike: boolean;
  bodyMovement: boolean;
};

const regionDiffScore = (
  prev: ImageData,
  next: ImageData,
  yStart: number,
  yEnd: number,
): number => {
  if (prev.data.length !== next.data.length) return 1;
  const w = prev.width;
  let changed = 0;
  let total = 0;
  for (let y = yStart; y < yEnd; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      total++;
      const dr = Math.abs(prev.data[i] - next.data[i]);
      const dg = Math.abs(prev.data[i + 1] - next.data[i + 1]);
      const db = Math.abs(prev.data[i + 2] - next.data[i + 2]);
      if (dr + dg + db > 42) changed++;
    }
  }
  return total ? changed / total : 0;
};

export const analyzeMotion = (prev: ImageData, next: ImageData): MotionSnapshot => {
  const motionLevel = frameDiffScore(prev, next);
  const h = prev.height;
  const mid = Math.floor(h / 2);
  const upperMotion = regionDiffScore(prev, next, 0, mid);
  const lowerMotion = regionDiffScore(prev, next, mid, h);
  const armMovement = upperMotion >= 0.12 && upperMotion > lowerMotion * 1.4;
  const bodyMovement = motionLevel >= 0.06;
  const waveLike = armMovement && motionLevel >= 0.18 && upperMotion >= 0.15;
  return { motionLevel, upperMotion, lowerMotion, armMovement, waveLike, bodyMovement };
};

export const motionFromSamples = (prev: FrameSample, next: FrameSample): MotionSnapshot =>
  analyzeMotion(prev.imageData, next.imageData);
