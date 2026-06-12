import { describe, expect, it } from "vitest";
import { isPackBlockedDuringBoot } from "./bootGate";
import type { DialogueContext } from "../vision2/types";
import { EMPTY_BODY_LANGUAGE, EMPTY_SITUATION } from "../vision2/types";
import { buildAcquaintanceMessage } from "../vision2/acquaintanceEngine";
import type { EntityProfile } from "../vision2/entityProfile";

const mockDialogue = (sceneAgeSec: number): DialogueContext =>
  ({
    worldState: {
      room: {
        hasLaptop: false,
        hasCup: false,
        hasPhone: false,
        hasTv: false,
        environment: "unknown",
        stableObjects: [],
        semanticNotes: "",
      },
      person: { present: true, absentDurationSec: 0 },
      session: { sceneAgeSec, workSessionMin: 0, lastBreakAt: 0, lastGreetingAt: 0, faceTouchDurationSec: 0 },
    },
    consciousness: {
      soul: "STABLE_PRESENCE",
      phase: "stable",
      confidence: 0.9,
      stabilitySec: 2,
      personStable: true,
      rawDetected: true,
      interpretation: "",
      evolution: "STABLE",
      affect: { curiosity: 0.5, certainty: 0.8, alertness: 0.4 },
      perception: { certainty: 0.8, ambiguity: 0.2, noiseFloor: 0.1 },
      gemmaBlock: "",
    },
    bodyLanguage: EMPTY_BODY_LANGUAGE,
    situation: EMPTY_SITUATION,
  }) as unknown as DialogueContext;

describe("bootGate", () => {
  it("blocks stress pack during warmup", () => {
    const d = mockDialogue(5);
    expect(
      isPackBlockedDuringBoot(
        {
          id: "stress-moderate",
          name: "x",
          enabled: true,
          triggers: {},
          interpretation: "",
          tone: "supportive",
          priority: "medium",
          cooldownMs: 1000,
          responses: [],
          sceneTags: ["stress"],
          proactive: true,
        },
        d,
      ),
    ).toBe(true);
  });

  it("allows greeting during warmup", () => {
    const d = mockDialogue(3);
    expect(
      isPackBlockedDuringBoot(
        {
          id: "greeting-entry",
          name: "x",
          enabled: true,
          triggers: {},
          interpretation: "",
          tone: "warm",
          priority: "high",
          cooldownMs: 1000,
          responses: [],
          sceneTags: ["social", "boot"],
          proactive: true,
        },
        d,
      ),
    ).toBe(false);
  });
});

describe("acquaintanceEngine", () => {
  it("builds child greeting with age", () => {
    const entity: EntityProfile = {
      ageEstimate: 7,
      ageRawEstimate: 7,
      gender: "female",
      segment: "child",
      emotion: "happy",
      emotionScore: 0.6,
      gazeAtCamera: true,
      engagement: "neutral",
      confidence: 0.7,
      faceObservations: 3,
      updatedAt: Date.now(),
    };
    const msg = buildAcquaintanceMessage(entity);
    expect(msg).toMatch(/ילדה/);
    expect(msg).toMatch(/7/);
    expect(msg).toMatch(/שחק/);
  });
});
