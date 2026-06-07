import { describe, expect, it } from "vitest";
import { inferPoseState, inferPoseStateWithConfidence, detectWaveGesture, attachHoldingObjects } from "./poseHeuristics";

const makeKps = (overrides: Record<number, { x: number; y: number; score: number }>) => {
  const kps = Array.from({ length: 17 }, () => ({ x: 0, y: 0, score: 0 }));
  for (const [idx, v] of Object.entries(overrides)) {
    kps[Number(idx)] = v;
  }
  for (let i = 0; i < kps.length; i++) {
    if (kps[i].score === 0) kps[i] = { x: 100, y: 100 + i * 5, score: 0.9 };
  }
  return kps;
};

describe("poseHeuristics", () => {
  it("detects sitting from knee geometry", () => {
    const kps = makeKps({
      5: { x: 90, y: 50, score: 0.9 },
      6: { x: 110, y: 50, score: 0.9 },
      11: { x: 95, y: 100, score: 0.9 },
      12: { x: 105, y: 100, score: 0.9 },
      13: { x: 95, y: 115, score: 0.9 },
      14: { x: 105, y: 115, score: 0.9 },
    });
    expect(inferPoseState(kps)).toBe("sitting");
  });

  it("infers standing with high confidence when legs extended", () => {
    const kps = makeKps({
      5: { x: 90, y: 50, score: 0.9 },
      6: { x: 110, y: 50, score: 0.9 },
      11: { x: 95, y: 100, score: 0.9 },
      12: { x: 105, y: 100, score: 0.9 },
      13: { x: 95, y: 170, score: 0.9 },
      14: { x: 105, y: 170, score: 0.9 },
    });
    const inf = inferPoseStateWithConfidence(kps);
    expect(inf.poseState).toBe("standing");
    expect(inf.confidence).toBeGreaterThan(0.45);
  });

  it("detects wave when wrist above shoulders", () => {
    const kps = makeKps({
      5: { x: 90, y: 100, score: 0.9 },
      6: { x: 110, y: 100, score: 0.9 },
      11: { x: 95, y: 180, score: 0.9 },
      12: { x: 105, y: 180, score: 0.9 },
      9: { x: 70, y: 40, score: 0.9 },
      10: { x: 130, y: 110, score: 0.9 },
    });
    expect(detectWaveGesture(kps)).toBe(true);
  });

  it("attaches objects inside person bbox", () => {
    const held = attachHoldingObjects(
      { x: 0, y: 0, width: 200, height: 300 },
      [{ label: "phone", bbox: { x: 50, y: 50, width: 30, height: 40 } }],
    );
    expect(held).toContain("phone");
  });

  it("prefers wrist regions when keypoints available", () => {
    const kps = makeKps({
      5: { x: 90, y: 50, score: 0.9 },
      6: { x: 110, y: 50, score: 0.9 },
      9: { x: 60, y: 80, score: 0.9 },
      10: { x: 140, y: 120, score: 0.9 },
    });
    const held = attachHoldingObjects(
      { x: 0, y: 0, width: 200, height: 300 },
      [
        { label: "cup", bbox: { x: 55, y: 75, width: 20, height: 25 } },
        { label: "chair", bbox: { x: 300, y: 200, width: 40, height: 40 } },
      ],
      0.2,
      kps,
    );
    expect(held).toContain("cup");
    expect(held).not.toContain("chair");
  });
});
