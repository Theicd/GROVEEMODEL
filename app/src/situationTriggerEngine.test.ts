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

  it("fires one finger rule from finger count", () => {
    const result = emptyVision();
    result.fingerStates.push({
      hand: "Right",
      count: 1,
      fingers: { thumb: "Closed", index: "Open", middle: "Closed", ring: "Closed", pinky: "Closed" },
    });
    const state = createSituationTriggerState();
    const events = evaluateSituationTriggers(result, DEFAULT_SITUATION_RULES, state);
    expect(events.some((e) => e.subject === "gesture:one_finger")).toBe(true);
  });

  it("fires hand on head from body language", () => {
    const result = emptyVision();
    result.bodyLanguage.push({
      signal: "Hand on head",
      meaning: "Touching head",
      category: "self-touch",
      confidence: 0.82,
    });
    const state = createSituationTriggerState();
    const events = evaluateSituationTriggers(result, DEFAULT_SITUATION_RULES, state);
    expect(events.some((e) => e.subject === "hand_on_head")).toBe(true);
  });
});
