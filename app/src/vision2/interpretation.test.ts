import { describe, expect, it } from "vitest";
import { fuseMetaEvents } from "./interpretation/eventFusion";
import { computeSceneState } from "./interpretation/sceneStateEngine";
import { buildNarrativeFrame, formatNarrativeForGemma } from "./interpretation/narrativeBuilder";
import { createAgentState, updateAgentState } from "./interpretation/attentionManager";
import { EMPTY_BODY_LANGUAGE, EMPTY_SITUATION } from "./types";

describe("interpretation brain layers", () => {
  it("fuses waving + person into social_interaction_attempt", () => {
    const scene = computeSceneState({
      obs: {
        timestamp: Date.now(),
        personPresent: true,
        waving: true,
        pointing: false,
        motionLevel: 0.2,
        touchingFace: false,
        touchingHead: false,
        handsOnHead: false,
        handNearEyes: false,
        handOnChin: false,
        raisedHand: false,
        thumbsUp: false,
        thumbsDown: false,
        holdingCup: false,
        usingPhone: false,
        gazeDown: false,
        gazeAtCamera: true,
      },
      human: {
        posture: "standing",
        attention: "camera",
        activity: "social",
        energy: "medium",
        engagement: 0.7,
        updatedAt: Date.now(),
      },
      body: EMPTY_BODY_LANGUAGE,
      situation: { ...EMPTY_SITUATION, primary: "greeting", confidence: 0.8, updatedAt: Date.now() },
      recentChanges: [],
      prevActivity: null,
      motionDelta: 0.1,
    });
    const meta = fuseMetaEvents({
      obs: {
        timestamp: Date.now(),
        personPresent: true,
        waving: true,
        pointing: false,
        motionLevel: 0.2,
        touchingFace: false,
        touchingHead: false,
        handsOnHead: false,
        handNearEyes: false,
        handOnChin: false,
        raisedHand: false,
        thumbsUp: false,
        thumbsDown: false,
        holdingCup: false,
        usingPhone: false,
        gazeDown: false,
        gazeAtCamera: true,
      },
      human: {
        posture: "standing",
        attention: "camera",
        activity: "social",
        energy: "medium",
        engagement: 0.7,
        updatedAt: Date.now(),
      },
      body: EMPTY_BODY_LANGUAGE,
      scene,
      recentChanges: [],
      faceTouchSec: 0,
      waveRising: true,
      personJustEntered: false,
    });
    expect(meta.some((e) => e.type === "social_interaction_attempt")).toBe(true);
  });

  it("builds narrative with internal monologue for Gemma", () => {
    const agent = updateAgentState(createAgentState(), {
      scene: {
        activity: "social",
        intensity: 0.7,
        stability: "changing",
        focusTarget: "agent",
        engagementLevel: 65,
        updatedAt: Date.now(),
      },
      metaEvents: [
        {
          type: "social_interaction_attempt",
          confidence: 0.85,
          components: ["waving"],
          meaning: "User tries to engage attention",
          ageSec: 0,
        },
      ],
      personPresent: true,
      stressLevel: 0.1,
      msSinceUserChat: 60_000,
    });
    const narrative = buildNarrativeFrame({
      scene: {
        activity: "social",
        intensity: 0.7,
        stability: "changing",
        focusTarget: "agent",
        engagementLevel: 65,
        updatedAt: Date.now(),
      },
      prevScene: null,
      metaEvents: [
        {
          type: "social_interaction_attempt",
          confidence: 0.85,
          components: ["waving"],
          meaning: "User tries to engage attention",
          ageSec: 0,
        },
      ],
      agent,
      recentChanges: [],
      situation: EMPTY_SITUATION,
      hal: {
        mood: "curious",
        tone: "warm",
        moodLabelHe: "סקרן",
        personPresent: true,
        sceneLabel: "פנייה ישירה",
        activePackId: null,
        interpretation: "Social attention",
        stressLevel: 0.1,
        engagement: 0.65,
      },
      personPresent: true,
    });
    const block = formatNarrativeForGemma({
      scene: {
        activity: "social",
        intensity: 0.7,
        stability: "changing",
        focusTarget: "agent",
        engagementLevel: 65,
        updatedAt: Date.now(),
      },
      metaEvents: [
        {
          type: "social_interaction_attempt",
          confidence: 0.85,
          components: ["waving"],
          meaning: "User tries to engage attention",
          ageSec: 0,
        },
      ],
      agent,
      narrative,
      personPresent: true,
      halMood: "curious",
    });
    expect(narrative.internalMonologue.length).toBeGreaterThan(0);
    expect(block).toContain("HAL NARRATIVE FRAME");
    expect(block).toContain("INTERNAL MONOLOGUE");
    expect(block).toContain("Person in frame: YES");
  });
});
