import { describe, expect, it } from "vitest";
import { detectSituationEvents } from "./situationInference";
import type { Keypoint } from "./poseHeuristics";

const noopCanEmit = () => true;

const sittingKps = (): Keypoint[] => {
  const kps: Keypoint[] = Array.from({ length: 17 }, () => ({ x: 100, y: 100, score: 0.9 }));
  kps[5] = { x: 90, y: 50, score: 0.9 };
  kps[6] = { x: 110, y: 50, score: 0.9 };
  kps[11] = { x: 95, y: 100, score: 0.9 };
  kps[12] = { x: 105, y: 100, score: 0.9 };
  kps[13] = { x: 95, y: 115, score: 0.9 };
  kps[14] = { x: 105, y: 115, score: 0.9 };
  return kps;
};

describe("situationInference", () => {
  it("emits object_held when cup newly attached", () => {
    const personBbox = { x: 50, y: 20, width: 100, height: 180 };
    const { events } = detectSituationEvents(
      {
        keypoints: sittingKps(),
        personBbox,
        objectBoxes: [{ label: "cup", bbox: { x: 80, y: 90, width: 30, height: 40 } }],
        motion: {
          motionLevel: 0.1,
          upperMotion: 0.1,
          lowerMotion: 0.05,
          armMovement: false,
          waveLike: false,
          bodyMovement: true,
        },
        personInFrame: true,
      },
      { prevPose: null, prevHolding: [] },
      noopCanEmit,
    );
    expect(events.some((e) => e.subject === "object_held:cup")).toBe(true);
  });

  it("emits stood_with_drink when sitting to standing with cup", () => {
    const personBbox = { x: 50, y: 20, width: 100, height: 180 };
    const standing = sittingKps();
    standing[13] = { x: 95, y: 160, score: 0.9 };
    standing[14] = { x: 105, y: 160, score: 0.9 };

    const { events } = detectSituationEvents(
      {
        keypoints: standing,
        personBbox,
        objectBoxes: [{ label: "cup", bbox: { x: 80, y: 90, width: 30, height: 40 } }],
        motion: {
          motionLevel: 0.15,
          upperMotion: 0.1,
          lowerMotion: 0.12,
          armMovement: false,
          waveLike: false,
          bodyMovement: true,
        },
        personInFrame: true,
      },
      {
        prevPose: {
          poseState: "sitting",
          confidence: 0.8,
          gestures: [],
          holding: ["cup"],
          focusHint: "",
        },
        prevHolding: ["cup"],
      },
      noopCanEmit,
    );
    expect(events.some((e) => e.subject === "stood_with_drink")).toBe(true);
  });
});
