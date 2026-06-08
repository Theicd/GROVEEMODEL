import { describe, expect, it } from "vitest";
import type { VisionResult } from "./vision-lab/core/types";
import {
  createSituationTriggerState,
  evaluateSituationTriggers,
  registrySubjectFromLabEvent,
} from "./situationTriggerEngine";
import { DEFAULT_SITUATION_RULES } from "./situationRegistry";

const emptyVision = (): VisionResult => ({
  objects: [],
  poseLandmarks: [],
  poseActions: [],
  hands: [],
  fingerStates: [],
  staticGestures: [],
  motionGestures: [],
  faces: [],
  emotion: null,
  interactions: [],
  events: [],
  bodyLanguage: [],
  environment: "Unknown",
  sceneDescription: "",
  vlmDescription: "",
  fps: 0,
  backend: "wasm",
});

describe("situationTriggerEngine", () => {
  it("fires wave rule on motion gesture", () => {
    const result = emptyVision();
    result.motionGestures.push({ name: "Waving", confidence: 0.9 });
    const state = createSituationTriggerState();
    const events = evaluateSituationTriggers(result, DEFAULT_SITUATION_RULES, state);
    expect(events.some((e) => e.subject === "wave")).toBe(true);
  });

  it("respects cooldown on repeated wave", () => {
    const result = emptyVision();
    result.motionGestures.push({ name: "Waving", confidence: 0.9 });
    const state = createSituationTriggerState();
    const first = evaluateSituationTriggers(result, DEFAULT_SITUATION_RULES, state);
    const second = evaluateSituationTriggers(result, DEFAULT_SITUATION_RULES, state);
    expect(first.length).toBeGreaterThan(0);
    expect(second.length).toBe(0);
  });

  it("maps lab event to registry subject", () => {
    const subject = registrySubjectFromLabEvent("Phone Usage", DEFAULT_SITUATION_RULES);
    expect(subject).toBe("focused_work");
  });
});
