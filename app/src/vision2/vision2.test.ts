import { describe, expect, it } from "vitest";
import { CharacterBrain } from "../characterBrain";
import { WorldMemory } from "../worldMemory";
import type { VisionResult } from "../vision-lab/core/types";
import {
  createBodyLanguageModelState,
  updateBodyLanguage,
} from "./bodyLanguageModel";
import { evaluateCoach } from "./coachEngine";
import {
  buildDialogueContext,
  buildFingerAnswerBlock,
  DIALOGUE_CONTEXT_SYSTEM_HINT,
  serializeDialogueContext,
} from "./dialogueContext";
import {
  createHumanStateEngineState,
  updateHumanState,
} from "./humanStateEngine";
import { perceiveFromVisionResult } from "./perceptionEngine";
import {
  createSituationEngineState,
  updateSituation,
} from "./situationEngine";
import { EMPTY_BODY_LANGUAGE, EMPTY_SITUATION } from "./types";
import { buildWorldSnapshot, createSessionTracker } from "./worldModel";
import { evaluateSocialAwareness } from "./capabilities/socialAwarenessModel";
import {
  createProductivityTracker,
  evaluateProductivity,
} from "./capabilities/productivityCoach";
import {
  createTeachingTracker,
  evaluateTeachingState,
} from "./capabilities/teachingAssistant";
import { evaluateEmotionalCoach, evaluateEmotionalState } from "./capabilities/emotionalCompanion";
import { filterEventsForVision2 } from "./eventFilter";
import { evaluateCoachDecision } from "./characterBrainVision2";
import { EMPTY_INTERPRETATION } from "./interpretationDefaults";
import { assertVision2PromptClean } from "./promptAudit";
import { makeSemanticEvent } from "../worldMemory";
import { Vision2Engine } from "./Vision2Engine";

const emptyCapabilities = (now = Date.now()) => ({
  social: {
    greeting: 0,
    interest: 0,
    agreement: 0,
    disagreement: 0,
    confusion: 0,
    updatedAt: now,
  },
  productivity: {
    workSessionMin: 0,
    focusStreakMin: 0,
    breaksTaken: 0,
    fatigueLevel: 0,
    needsBreak: false,
    updatedAt: now,
  },
  teaching: {
    attentionLoss: 0,
    engagementDrop: 0,
    likelyDistracted: false,
    ageSec: 0,
    updatedAt: now,
  },
  emotional: {
    supportNeeded: 0,
    encouragementOpportunity: 0,
    motivation: 0,
    reflection: 0,
    updatedAt: now,
  },
});

const minimalVisionResult = (): VisionResult => ({
  objects: [
    {
      label: "person",
      displayLabel: "Person",
      confidence: 0.9,
      bbox: { x: 0.2, y: 0.1, width: 0.4, height: 0.8 },
    },
  ],
  poseLandmarks: [],
  poseActions: [{ name: "Sitting", confidence: 0.85 }],
  hands: [],
  fingerStates: [],
  staticGestures: [],
  motionGestures: [],
  faces: [],
  emotion: null,
  faceModule: {
    status: "ready",
    message: "",
    lastScanAt: 0,
    lastFaceCount: 0,
    modelSource: "local",
  },
  interactions: [],
  events: [],
  bodyLanguage: [
    {
      signal: "Hand on chin/jaw",
      meaning: "thinking",
      category: "self-touch",
      confidence: 0.83,
    },
  ],
  environment: "Office",
  sceneDescription: "Person sitting.",
  vlmDescription: "",
  fps: 12,
  backend: "wasm",
});

describe("perceptionEngine", () => {
  it("maps body language to handOnChin", () => {
    const obs = perceiveFromVisionResult(minimalVisionResult(), true, 0.04);
    expect(obs.handOnChin).toBe(true);
    expect(obs.personPresent).toBe(true);
  });
});

describe("bodyLanguageModel", () => {
  it("raises thinking score for chin touch + low motion", () => {
    const humanEngine = createHumanStateEngineState();
    const world = new WorldMemory();
    world.personPresent = true;
    world.poseState = "sitting";
    const obs = perceiveFromVisionResult(minimalVisionResult(), true, 0.03);
    const human = updateHumanState(obs, world, humanEngine);
    const blState = createBodyLanguageModelState();
    const body = updateBodyLanguage(obs, human, { faceTouchSec: 12, handsOnHeadSec: 0 }, blState);
    expect(body.thinking).toBeGreaterThan(0.5);
  });
});

describe("situationEngine", () => {
  it("infers reflecting for thinking activity", () => {
    const humanEngine = createHumanStateEngineState();
    const world = new WorldMemory();
    world.personPresent = true;
    world.poseState = "sitting";
    const obs = perceiveFromVisionResult(minimalVisionResult(), true, 0.03);
    const human = updateHumanState(obs, world, humanEngine);
    human.activity = "thinking";
    human.attention = "internal";
    const sitState = createSituationEngineState();
    const situation = updateSituation(obs, human, sitState);
    expect(["reflecting", "idle", "unknown"]).toContain(situation.primary);
  });
});

describe("dialogueContext", () => {
  it("serializes without landmark keys", () => {
    const world = new WorldMemory();
    world.personPresent = true;
    world.baselineEstablished = true;
    const humanEngine = createHumanStateEngineState();
    const obs = perceiveFromVisionResult(minimalVisionResult(), true, 0.03);
    const human = updateHumanState(obs, world, humanEngine);
    const blState = createBodyLanguageModelState();
    const body = updateBodyLanguage(obs, human, { faceTouchSec: 5, handsOnHeadSec: 0 }, blState);
    const sitState = createSituationEngineState();
    const situation = updateSituation(obs, human, sitState);
    const snapshot = buildWorldSnapshot(
      world,
      human,
      body,
      situation,
      createSessionTracker(),
      5,
    );
    const brain = new CharacterBrain();
    const ctx = buildDialogueContext({
      world,
      snapshot,
      human,
      coach: { intent: "none", reason: "", urgency: 0 },
      capabilities: emptyCapabilities(),
      audio: { available: false, level: 0, speechDetected: false },
      episodicSummary: [],
      character: brain,
    });
    const json = serializeDialogueContext(ctx);
    expect(json).not.toMatch(/landmark|bbox|fingerStates|YOLO/i);
    expect(json).toContain("personState");
    expect(json).toContain("bodyLanguage");
    expect(json).toContain("hal");
  });

  it("buildFingerAnswerBlock avoids landmark dump", () => {
    const block = buildFingerAnswerBlock(2, "Peace Sign");
    expect(block).toContain("Total extended fingers visible: 2");
    expect(block).not.toContain("thumb=");
  });
});

describe("coachEngine", () => {
  it("suggests break after long work session", () => {
    const human = {
      posture: "sitting" as const,
      attention: "screen" as const,
      activity: "working" as const,
      energy: "medium" as const,
      engagement: 0.8,
      updatedAt: Date.now(),
    };
    const body = { ...EMPTY_BODY_LANGUAGE, focused: 0.85, ageSec: 30, updatedAt: Date.now() };
    const situation = {
      ...EMPTY_SITUATION,
      primary: "working" as const,
      confidence: 0.8,
      updatedAt: Date.now(),
    };
    const session = {
      sceneAgeSec: 3000,
      workSessionMin: 50,
      lastBreakAt: 0,
      lastGreetingAt: 0,
      faceTouchDurationSec: 0,
    };
    const coach = evaluateCoach(human, body, situation, session);
    expect(coach.intent).toBe("suggest_break");
  });
});

describe("Phase 7 capabilities", () => {
  it("social awareness detects thumbs up agreement", () => {
    const result = minimalVisionResult();
    result.staticGestures = [{ name: "Thumbs Up", confidence: 0.9, hand: "Right" }];
    const obs = perceiveFromVisionResult(result, true, 0.05);
    const humanEngine = createHumanStateEngineState();
    const world = new WorldMemory();
    world.personPresent = true;
    const human = updateHumanState(obs, world, humanEngine);
    const body = updateBodyLanguage(
      obs,
      human,
      { faceTouchSec: 0, handsOnHeadSec: 0 },
      createBodyLanguageModelState(),
    );
    const social = evaluateSocialAwareness(obs, human, body, null);
    expect(social.agreement).toBeGreaterThan(0.6);
  });

  it("teaching assistant flags attention loss", () => {
    const obs = perceiveFromVisionResult(minimalVisionResult(), true, 0.02);
    const human = {
      posture: "sitting" as const,
      attention: "away" as const,
      activity: "unknown" as const,
      energy: "low" as const,
      engagement: 0.2,
      updatedAt: Date.now(),
    };
    const body = { ...EMPTY_BODY_LANGUAGE, bored: 0.6, updatedAt: Date.now(), ageSec: 15 };
    const trackers = createTeachingTracker();
    for (let i = 0; i < 20; i++) {
      evaluateTeachingState(obs, human, body, trackers, Date.now() + i * 1000);
    }
    const teaching = evaluateTeachingState(obs, human, body, trackers, Date.now() + 25_000);
    expect(teaching.attentionLoss).toBeGreaterThan(0.5);
  });

  it("productivity coach marks needsBreak on long focus", () => {
    const tracker = createProductivityTracker();
    tracker.focusStartedAt = Date.now() - 50 * 60_000;
    const session = {
      sceneAgeSec: 4000,
      workSessionMin: 50,
      lastBreakAt: 0,
      lastGreetingAt: 0,
      faceTouchDurationSec: 0,
    };
    const body = { ...EMPTY_BODY_LANGUAGE, focused: 0.85, ageSec: 40, updatedAt: Date.now() };
    const human = {
      posture: "sitting" as const,
      attention: "screen" as const,
      activity: "working" as const,
      energy: "medium" as const,
      engagement: 0.75,
      updatedAt: Date.now(),
    };
    const productivity = evaluateProductivity(session, tracker, body, human, {
      ...EMPTY_SITUATION,
      primary: "working",
      confidence: 0.8,
      updatedAt: Date.now(),
    });
    expect(productivity.needsBreak).toBe(true);
  });

  it("emotional coach offers support under stress", () => {
    const human = {
      posture: "sitting" as const,
      attention: "internal" as const,
      activity: "thinking" as const,
      energy: "low" as const,
      engagement: 0.4,
      updatedAt: Date.now(),
    };
    const body = { ...EMPTY_BODY_LANGUAGE, stressed: 0.75, thinking: 0.6, ageSec: 20, updatedAt: Date.now() };
    const social = { greeting: 0, interest: 0.2, agreement: 0, disagreement: 0, confusion: 0.3, updatedAt: Date.now() };
    const productivity = {
      workSessionMin: 10,
      focusStreakMin: 10,
      breaksTaken: 0,
      fatigueLevel: 0.5,
      needsBreak: false,
      updatedAt: Date.now(),
    };
    const emotional = evaluateEmotionalState(human, body, social, productivity);
    const coach = evaluateEmotionalCoach(
      human,
      body,
      { ...EMPTY_SITUATION, primary: "reflecting", confidence: 0.7, updatedAt: Date.now() },
      {
        sceneAgeSec: 600,
        workSessionMin: 10,
        lastBreakAt: 0,
        lastGreetingAt: 0,
        faceTouchDurationSec: 20,
      },
      social,
      emotional,
      productivity,
    );
    expect(coach.intent).toBe("offer_support");
  });
});

describe("eventFilter", () => {
  it("suppresses hand_on_face for vision2", () => {
    const events = [
      makeSemanticEvent("activity_change", "Hand on face", "hand_on_face", true),
      makeSemanticEvent("activity_change", "Wave", "wave", true),
    ];
    const filtered = filterEventsForVision2(events);
    expect(filtered.some((e) => e.subject === "hand_on_face")).toBe(false);
    expect(filtered.some((e) => e.subject === "wave")).toBe(true);
  });
});

describe("characterBrainVision2", () => {
  it("returns coach decision for suggest_break", () => {
    const brain = new CharacterBrain();
    brain.lastProactiveAt = 0;
    const decision = evaluateCoachDecision(brain, {
      worldState: {
        room: {
          hasLaptop: true,
          hasCup: false,
          hasPhone: false,
          hasTv: false,
          environment: "office",
          stableObjects: [],
          semanticNotes: "",
        },
        person: { present: true, absentDurationSec: 0 },
        session: {
          sceneAgeSec: 100,
          workSessionMin: 50,
          lastBreakAt: 0,
          lastGreetingAt: 0,
          faceTouchDurationSec: 0,
        },
      },
      personState: {
        posture: "sitting",
        attention: "screen",
        activity: "working",
        energy: "medium",
        engagement: 0.8,
        updatedAt: Date.now(),
      },
      bodyLanguage: { ...EMPTY_BODY_LANGUAGE, focused: 0.85, ageSec: 30, updatedAt: Date.now() },
      situation: { ...EMPTY_SITUATION, primary: "working", confidence: 0.8, updatedAt: Date.now() },
      recentChanges: [],
      coach: { intent: "suggest_break", reason: "long focus", urgency: 0.7 },
      capabilities: emptyCapabilities(),
      audio: { available: false, level: 0, speechDetected: false },
      character: { mood: "observing", shouldSpeak: true, speakReason: "coach:suggest_break" },
      hal: {
        mood: "observing",
        tone: "soft",
        moodLabelHe: "תצפית",
        personPresent: true,
        sceneLabel: null,
        activePackId: null,
        interpretation: "work",
        stressLevel: 0,
        engagement: 0.8,
      },
      interpretation: EMPTY_INTERPRETATION,
      episodicSummary: [],
    });
    expect(decision?.reason).toContain("coach:");
    expect(decision?.message).toMatch(/מים|נשום|רגע/);
  });
});

describe("promptAudit", () => {
  it("passes clean HAL JSON prompt", () => {
    const prompt = `${DIALOGUE_CONTEXT_SYSTEM_HINT}\nHAL_PERCEPTION_JSON:{"personState":{},"capabilities":{}}`;
    expect(() => assertVision2PromptClean(prompt)).not.toThrow();
  });

  it("fails on raw sensor dump", () => {
    expect(() => assertVision2PromptClean("Finger counts: Left:3")).toThrow();
  });
});

describe("Vision2Engine integration", () => {
  it("process returns capabilities in dialogue", () => {
    const engine = new Vision2Engine();
    const world = new WorldMemory();
    world.personPresent = true;
    world.baselineEstablished = true;
    world.poseState = "sitting";
    const { dialogue } = engine.process(minimalVisionResult(), world, new CharacterBrain());
    expect(dialogue.capabilities.social).toBeDefined();
    expect(dialogue.capabilities.productivity).toBeDefined();
    expect(dialogue.capabilities.teaching).toBeDefined();
    expect(dialogue.capabilities.emotional).toBeDefined();
    expect(dialogue.audio).toBeDefined();
    const json = serializeDialogueContext(dialogue);
    expect(json).toContain("capabilities");
  });
});
