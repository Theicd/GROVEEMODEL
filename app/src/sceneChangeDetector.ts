/** Lightweight frame change detection — no model inference. */

const DIFF_WIDTH = 64;
const DIFF_HEIGHT = 48;

export type FrameSample = {
  imageData: ImageData;
  score: number;
};

export const captureDownscaledFrame = (
  video: HTMLVideoElement,
  width = DIFF_WIDTH,
  height = DIFF_HEIGHT,
): FrameSample | null => {
  if (video.readyState < 2 || video.videoWidth <= 0 || video.videoHeight <= 0) return null;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) return null;
  ctx.drawImage(video, 0, 0, width, height);
  const imageData = ctx.getImageData(0, 0, width, height);
  return { imageData, score: 0 };
};

export const frameDiffScore = (prev: ImageData, next: ImageData): number => {
  if (prev.data.length !== next.data.length) return 1;
  const pixels = prev.data.length / 4;
  let changed = 0;
  for (let i = 0; i < prev.data.length; i += 4) {
    const dr = Math.abs(prev.data[i] - next.data[i]);
    const dg = Math.abs(prev.data[i + 1] - next.data[i + 1]);
    const db = Math.abs(prev.data[i + 2] - next.data[i + 2]);
    if (dr + dg + db > 42) changed++;
  }
  return changed / pixels;
};

export const isSignificantChange = (score: number, threshold = 0.08): boolean => score >= threshold;
