/** Best-frame selection for deep vision — ported from JARVIS-VISION frame_stability.py. */

export type ScoredFrame = {
  imageData: ImageData;
  score: number;
};

const toGray = (imageData: ImageData, targetW = 160, targetH = 90): Float32Array => {
  const { width, height, data } = imageData;
  const gray = new Float32Array(targetW * targetH);
  for (let ty = 0; ty < targetH; ty++) {
    for (let tx = 0; tx < targetW; tx++) {
      const sx = Math.floor((tx / targetW) * width);
      const sy = Math.floor((ty / targetH) * height);
      const i = (sy * width + sx) * 4;
      gray[ty * targetW + tx] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    }
  }
  return gray;
};

const simpleSsim = (a: Float32Array, b: Float32Array): number => {
  if (a.length !== b.length) return 0;
  const c1 = (0.01 * 255) ** 2;
  const c2 = (0.03 * 255) ** 2;
  let muA = 0;
  let muB = 0;
  for (let i = 0; i < a.length; i++) {
    muA += a[i];
    muB += b[i];
  }
  muA /= a.length;
  muB /= b.length;
  let varA = 0;
  let varB = 0;
  let cov = 0;
  for (let i = 0; i < a.length; i++) {
    const da = a[i] - muA;
    const db = b[i] - muB;
    varA += da * da;
    varB += db * db;
    cov += da * db;
  }
  varA /= a.length;
  varB /= b.length;
  cov /= a.length;
  const num = (2 * muA * muB + c1) * (2 * cov + c2);
  const den = (muA * muA + muB * muB + c1) * (varA + varB + c2);
  return num / Math.max(den, 1e-10);
};

const frameDiffMean = (a: Float32Array, b: Float32Array): number => {
  if (a.length !== b.length) return 0;
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
  return sum / a.length;
};

export class FrameStabilityScorer {
  private prevGray: Float32Array | null = null;
  private buffer: ScoredFrame[] = [];
  private readonly bufferSize: number;

  constructor(bufferSize = 12) {
    this.bufferSize = bufferSize;
  }

  reset(): void {
    this.prevGray = null;
    this.buffer = [];
  }

  /** Score frame and keep in rolling buffer. */
  scoreFrame(imageData: ImageData): number {
    const gray = toGray(imageData);

    if (!this.prevGray) {
      this.prevGray = gray;
      this.buffer.push({ imageData, score: 0.5 });
      return 0.5;
    }

    const ssim = Math.max(0, Math.min(1, simpleSsim(this.prevGray, gray)));
    const flowMag = frameDiffMean(this.prevGray, gray);
    const flowNorm = Math.min(1, flowMag / 15);
    const score = 0.5 * ssim + 0.5 * flowNorm;
    this.prevGray = gray;

    this.buffer.push({ imageData, score });
    if (this.buffer.length > this.bufferSize) this.buffer.shift();
    return score;
  }

  getBestFrame(): ImageData | null {
    if (!this.buffer.length) return null;
    return this.buffer.reduce((best, cur) => (cur.score >= best.score ? cur : best)).imageData;
  }
}
