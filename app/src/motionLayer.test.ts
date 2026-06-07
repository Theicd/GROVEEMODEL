import { describe, expect, it } from "vitest";
import { analyzeMotion } from "./motionLayer";

const makeImage = (w: number, h: number, fill: (x: number, y: number) => [number, number, number]) => {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const [r, g, b] = fill(x, y);
      const i = (y * w + x) * 4;
      data[i] = r;
      data[i + 1] = g;
      data[i + 2] = b;
      data[i + 3] = 255;
    }
  }
  return { data, width: w, height: h } as ImageData;
};

describe("motionLayer", () => {
  it("detects zero motion on identical frames", () => {
    const a = makeImage(8, 8, () => [10, 20, 30]);
    const snap = analyzeMotion(a, a);
    expect(snap.motionLevel).toBe(0);
    expect(snap.waveLike).toBe(false);
  });

  it("detects upper-region motion as arm movement", () => {
    const prev = makeImage(8, 8, (_x, y) => (y < 4 ? [0, 0, 0] : [100, 100, 100]));
    const next = makeImage(8, 8, (_x, y) =>
      y < 4 ? [255, 255, 255] : [100, 100, 100],
    );
    const snap = analyzeMotion(prev, next);
    expect(snap.upperMotion).toBeGreaterThan(0.5);
    expect(snap.lowerMotion).toBe(0);
    expect(snap.armMovement).toBe(true);
  });
});
