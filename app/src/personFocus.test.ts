import { describe, expect, it } from "vitest";
import { formatFreshPersonBlock } from "./personFocus";

describe("personFocus", () => {
  it("marks low confidence as uncertain", () => {
    const block = formatFreshPersonBlock({
      personPresent: true,
      poseState: "unknown",
      confidence: 0.3,
      holding: [],
      gestures: [],
      focusHint: "",
      poseSource: "bbox",
      capturedAt: Date.now(),
      validationFrames: 2,
    });
    expect(block).toMatch(/uncertain|קשה לדעת/);
  });

  it("includes standing when confidence is high", () => {
    const block = formatFreshPersonBlock({
      personPresent: true,
      poseState: "standing",
      confidence: 0.72,
      holding: ["cup"],
      gestures: [],
      focusHint: "standing with a drink",
      poseSource: "movenet",
      capturedAt: Date.now(),
      validationFrames: 2,
    });
    expect(block).toMatch(/Posture: standing/);
    expect(block).toMatch(/Holding: cup/);
  });
});
