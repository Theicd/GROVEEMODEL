import { describe, expect, it } from "vitest";
import { buildFingerCountBlock, mapLabGestures, mapPoseActionToState } from "./visionBridge";
import type { VisionResult } from "./vision-lab/core/types";

describe("visionBridge", () => {
  it("maps standing pose action", () => {
    const pose = mapPoseActionToState([{ name: "Standing", confidence: 0.9 }]);
    expect(pose.poseState).toBe("standing");
    expect(pose.confidence).toBeGreaterThan(0.8);
  });

  it("collects gesture names from lab result shape", () => {
    const gestures = mapLabGestures({
      staticGestures: [{ name: "Thumbs Up", confidence: 0.9, hand: "Right" }],
      motionGestures: [{ name: "Waving", confidence: 0.8 }],
      poseActions: [{ name: "Right Hand Raised", confidence: 0.85 }],
    } as never);
    expect(gestures).toContain("thumbs_up");
    expect(gestures).toContain("waving");
  });

  it("builds finger count block from hand sensor", () => {
    const block = buildFingerCountBlock({
      hands: [{ handedness: "Right", landmarks: [], bbox: { x: 0, y: 0, width: 0.1, height: 0.1 } }],
      fingerStates: [
        {
          hand: "Right",
          count: 1,
          fingers: { thumb: "Closed", index: "Open", middle: "Closed", ring: "Closed", pinky: "Closed" },
        },
      ],
      staticGestures: [],
    } as unknown as VisionResult);
    expect(block).toContain("Total extended fingers visible: 1");
    expect(block).toContain("Right hand: 1 finger");
  });
});
