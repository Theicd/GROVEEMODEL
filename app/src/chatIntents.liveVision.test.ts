import { describe, expect, it } from "vitest";
import { isCameraContextQuestion, isConsciousnessQuestion, needsLiveCameraContext } from "./chatIntents";
import { buildLiveVisionChatBrief } from "./vision2/liveVisionChatBrief";
import { WorldMemory } from "./worldMemory";
import type { VisionResult } from "./vision-lab/core/types";

describe("camera chat intents", () => {
  it("detects מה קורה and consciousness questions", () => {
    expect(isCameraContextQuestion("מה קורה?")).toBe(true);
    expect(isCameraContextQuestion("מה אתה מזהה כרגע במרחב?")).toBe(true);
    expect(isConsciousnessQuestion("מה רמת הוודאות שלך?")).toBe(true);
    expect(needsLiveCameraContext("מה אתה מזהה כרגע במרחב?")).toBe(true);
  });
});

describe("buildLiveVisionChatBrief", () => {
  it("includes yolo and face facts", () => {
    const world = new WorldMemory();
    const vision = {
      objects: [{ label: "person", displayLabel: "person", confidence: 0.9, bbox: { x: 0, y: 0, width: 0.1, height: 0.2 } }],
      faces: [{ id: 1, estimatedAge: 44, estimatedGender: "Male", gazeDirection: "Center", bbox: { x: 0, y: 0, width: 0.1, height: 0.1 } }],
    } as unknown as VisionResult;
    const brief = buildLiveVisionChatBrief({ vision, dialogue: null, world, cameraActive: true });
    expect(brief).toMatch(/personVisible=yes/);
    expect(brief).toMatch(/\[INTERNAL VISION CONTEXT/);
  });
});
