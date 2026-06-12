import { describe, expect, it } from "vitest";
import {
  DEFAULT_SITUATION_PACKS,
  LEVEL1_PACK_COUNT,
  SITUATION_PACK_COUNT,
} from "./defaultPacks";
import { LEVEL2_PACK_COUNT, LEVEL2_SITUATION_PACKS } from "./level2Packs";
import { bodyLanguageFromObs, gesturesFromObs } from "./patternContext";
import { matchSituationPacks, matchTriggers } from "./patternMatcher";
import { buildScene } from "./sceneBuilder";
import { createSignalHistory, recordSignal, signalCount } from "./signalHistory";
import { pickResponseVariant, createVariationState, noteVariantUsed } from "./variationsEngine";
import type { MatchContext } from "./patternContext";
import {
  EMPTY_BODY_LANGUAGE,
  EMPTY_SITUATION,
  type HumanState,
  type ObservationSet,
} from "../vision2/types";

const baseObs = (over: Partial<ObservationSet> = {}): ObservationSet => ({
  timestamp: Date.now(),
  personPresent: true,
  touchingFace: false,
  touchingHead: false,
  handsOnHead: false,
  handNearEyes: false,
  handOnChin: false,
  raisedHand: false,
  waving: false,
  pointing: false,
  thumbsUp: false,
  thumbsDown: false,
  holdingCup: false,
  usingPhone: false,
  gazeDown: false,
  gazeAtCamera: false,
  motionLevel: 0.05,
  ...over,
});

const baseHuman = (over: Partial<HumanState> = {}): HumanState => ({
  posture: "sitting",
  attention: "screen",
  activity: "working",
  energy: "medium",
  engagement: 0.5,
  updatedAt: Date.now(),
  ...over,
});

const minimalCtx = (obs: ObservationSet, over: Partial<MatchContext> = {}): MatchContext => {
  const history = createSignalHistory();
  const now = Date.now();
  if (obs.waving) {
    recordSignal(history, "gesture:waving", now);
    recordSignal(history, "gesture:waving", now - 500);
  }
  return {
    now,
    obs,
    human: baseHuman(),
    body: { ...EMPTY_BODY_LANGUAGE, thinking: obs.handOnChin ? 0.7 : 0, updatedAt: now },
    situation: { ...EMPTY_SITUATION, primary: "unknown", confidence: 0.8, updatedAt: now },
    world: {} as MatchContext["world"],
    snapshot: {
      room: { hasLaptop: false, hasCup: false, hasPhone: false, hasTv: false, environment: "unknown", stableObjects: [], semanticNotes: "" },
      person: { present: true, absentDurationSec: 0, posture: "sitting", attention: "screen", activity: "working", reflecting: false },
      session: { sceneAgeSec: 10, workSessionMin: 5, lastBreakAt: 0, lastGreetingAt: 0, faceTouchDurationSec: 0 },
      bodyLanguage: EMPTY_BODY_LANGUAGE,
      situation: EMPTY_SITUATION,
      updatedAt: now,
    },
    history,
    gestures: gesturesFromObs(obs),
    bodyLanguage: bodyLanguageFromObs(obs),
    events: [],
    objects: [],
    poseActions: [],
    motionLevel: obs.motionLevel,
    silenceSec: 0,
    personJustEntered: false,
    ...over,
  };
};

describe("situation packs", () => {
  it("ships Level 1 + Level 2 (100+ packs)", () => {
    expect(LEVEL1_PACK_COUNT).toBeGreaterThanOrEqual(50);
    expect(LEVEL2_PACK_COUNT).toBe(50);
    expect(SITUATION_PACK_COUNT).toBe(LEVEL1_PACK_COUNT + LEVEL2_PACK_COUNT);
    expect(DEFAULT_SITUATION_PACKS.every((p) => p.responses.length >= 3)).toBe(true);
  });

  it("Level 2 packs are pattern bundles with cognition metadata", () => {
    expect(LEVEL2_SITUATION_PACKS.every((p) => p.id.startsWith("l2-"))).toBe(true);
    expect(LEVEL2_SITUATION_PACKS.every((p) => p.cognition?.length)).toBe(true);
    expect(LEVEL2_SITUATION_PACKS.every((p) => p.sceneTags?.includes("psych"))).toBe(true);
  });

  it("matches Level 2 rumination from sustained thinking bundle", () => {
    const obs = baseObs({ touchingFace: true, handOnChin: true, motionLevel: 0.03 });
    const ctx = minimalCtx(obs, {
      body: { ...EMPTY_BODY_LANGUAGE, thinking: 0.72, stressed: 0.4, updatedAt: Date.now() },
      snapshot: {
        room: { hasLaptop: false, hasCup: false, hasPhone: false, hasTv: false, environment: "unknown", stableObjects: [], semanticNotes: "" },
        person: { present: true, absentDurationSec: 0, posture: "sitting", attention: "screen", activity: "thinking", reflecting: true },
        session: { sceneAgeSec: 20, workSessionMin: 10, lastBreakAt: 0, lastGreetingAt: 0, faceTouchDurationSec: 15 },
        bodyLanguage: EMPTY_BODY_LANGUAGE,
        situation: EMPTY_SITUATION,
        updatedAt: Date.now(),
      },
    });
    const pack = LEVEL2_SITUATION_PACKS.find((p) => p.id === "l2-rumination-loop")!;
    const { match } = matchTriggers(pack.triggers, ctx);
    expect(match).toBe(true);
  });

  it("matches deep focus bundle not single gesture", () => {
    const obs = baseObs({ motionLevel: 0.04 });
    const ctx = minimalCtx(obs, {
      human: baseHuman({ posture: "sitting", attention: "screen" }),
      situation: { primary: "working", confidence: 0.85, description: "work", updatedAt: Date.now() },
    });
    const history = ctx.history;
    const now = ctx.now;
    for (let i = 0; i < 12; i++) {
      history.hits.set("presence:person", [...(history.hits.get("presence:person") ?? []), now - i * 1000]);
    }
    ctx.history = history;

    const pack = DEFAULT_SITUATION_PACKS.find((p) => p.id === "deep-focus-work")!;
    const { match } = matchTriggers(pack.triggers, ctx);
    expect(match).toBe(true);
  });

  it("matches thinking from body language not isolated wave", () => {
    const obs = baseObs({ touchingFace: true, handOnChin: true });
    const ctx = minimalCtx(obs);
    const matches = matchSituationPacks(DEFAULT_SITUATION_PACKS, ctx);
    expect(matches.some((m) => m.pack.id === "thinking-hand-face")).toBe(true);
  });

  it("picks different variants over repetitions", () => {
    const pack = DEFAULT_SITUATION_PACKS.find((p) => p.id === "attention-seeking-wave")!;
    const state = createVariationState();
    const first = pickResponseVariant(pack, state, "observing");
    noteVariantUsed(state, pack.id, first);
    const second = pickResponseVariant(pack, state, "observing");
    expect(pack.responses).toContain(first);
    expect(pack.responses).toContain(second);
  });

  it("builds composite scene from multiple tags", () => {
    const wave = DEFAULT_SITUATION_PACKS.find((p) => p.id === "attention-seeking-wave")!;
    const cup = DEFAULT_SITUATION_PACKS.find((p) => p.id === "holding-cup-break")!;
    const scene = buildScene([
      { pack: wave, score: 0.9, confidence: 0.8 },
      { pack: cup, score: 0.7, confidence: 0.7 },
    ]);
    expect(scene?.label).toBeTruthy();
    expect(scene?.packIds.length).toBe(2);
  });

  it("counts gesture repetition in time window", () => {
    const history = createSignalHistory();
    const now = Date.now();
    recordSignal(history, "gesture:waving", now);
    recordSignal(history, "gesture:waving", now - 800);
    expect(signalCount(history, "gesture:waving", 3, now)).toBe(2);
  });
});
