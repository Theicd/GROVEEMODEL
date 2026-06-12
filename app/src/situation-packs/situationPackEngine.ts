/**
 * Pattern-based situation engine — multi-signal → scene → variant response.
 * Replaces flat gesture → single reply when Vision 2.0 is active.
 */

import type { CharacterBrain, CharacterDecision } from "../characterBrain";
import { CHARACTER_CONFIG } from "../characterBrain";
import type { VisionResult } from "../vision-lab/core/types";
import type { SemanticEvent, WorldMemory } from "../worldMemory";
import type { DialogueContext } from "../vision2/types";
import { perceiveFromVisionResult } from "../vision2/perceptionEngine";
import { loadSituationPacks } from "./situationPackStorage";
import {
  buildMatchContext,
  recordObservationsToHistory,
  collectSignalsFromResult,
} from "./patternContext";
import { matchSituationPacks } from "./patternMatcher";
import { buildScene } from "./sceneBuilder";
import { isPackBlockedDuringBoot } from "./bootGate";
import {
  createSignalHistory,
  resetSignalHistory,
  updateMotionHistory,
  type SignalHistory,
} from "./signalHistory";
import type { MatchedSituation, SituationPack, SituationPackDecision } from "./types";
import {
  createVariationState,
  noteVariantUsed,
  pickResponseVariant,
  resetVariationState,
  type VariationState,
} from "./variationsEngine";

export type SituationPackEngineState = {
  history: SignalHistory;
  variations: VariationState;
  lastFiredAt: Map<string, number>;
  lastPackId: string | null;
};

export const createSituationPackEngineState = (): SituationPackEngineState => ({
  history: createSignalHistory(),
  variations: createVariationState(),
  lastFiredAt: new Map(),
  lastPackId: null,
});

const toneToMood = (tone: string): "curious" | "excited" | "observing" => {
  if (tone === "positive" || tone === "playful" || tone === "warm") return "excited";
  if (tone === "curious" || tone === "analytical") return "curious";
  return "observing";
};

const canFirePack = (state: SituationPackEngineState, packId: string, cooldownMs: number): boolean => {
  const last = state.lastFiredAt.get(packId) ?? 0;
  return Date.now() - last >= cooldownMs;
};

const markPackFired = (state: SituationPackEngineState, packId: string): void => {
  state.lastFiredAt.set(packId, Date.now());
};

export class SituationPackEngine {
  private state = createSituationPackEngineState();

  reset(): void {
    resetSignalHistory(this.state.history);
    resetVariationState(this.state.variations);
    this.state.lastFiredAt.clear();
    this.state.lastPackId = null;
  }

  ingest(
    result: VisionResult,
    world: WorldMemory,
    freshEvents: SemanticEvent[],
    personJustEntered: boolean,
  ): void {
    const now = Date.now();
    const obs = perceiveFromVisionResult(result, world.personPresent, world.lastMotionLevel, now);
    const signals = collectSignalsFromResult(result);
    recordObservationsToHistory(this.state.history, obs, signals, now);
    updateMotionHistory(this.state.history, obs.motionLevel, obs.personPresent, now);
    if (personJustEntered) {
      this.state.history.hits.set("event:person_entered", [now]);
      this.state.history.lastInteractionAt = now;
    }
    for (const ev of freshEvents) {
      const key = `event:${(ev.text ?? ev.subject ?? "").trim().toLowerCase().replace(/\s+/g, "_")}`;
      if (key.length > 7) {
        const prev = this.state.history.hits.get(key) ?? [];
        this.state.history.hits.set(key, [...prev.slice(-20), now]);
      }
    }
  }

  evaluate(
    brain: CharacterBrain,
    dialogue: DialogueContext,
    result: VisionResult,
    world: WorldMemory,
    freshEvents: SemanticEvent[],
    personJustEntered: boolean,
  ): SituationPackDecision | null {
    const snapshot = {
      room: dialogue.worldState.room,
      person: {
        present: dialogue.worldState.person.present,
        absentDurationSec: dialogue.worldState.person.absentDurationSec,
        posture: dialogue.personState.posture,
        attention: dialogue.personState.attention,
        activity: dialogue.personState.activity,
        reflecting: dialogue.bodyLanguage.thinking >= 0.5,
      },
      session: dialogue.worldState.session,
      bodyLanguage: dialogue.bodyLanguage,
      situation: dialogue.situation,
      updatedAt: Date.now(),
    };

    const obs = perceiveFromVisionResult(
      result,
      world.personPresent,
      world.lastMotionLevel,
      Date.now(),
    );

    const ctx = buildMatchContext(
      result,
      obs,
      dialogue.personState,
      dialogue.bodyLanguage,
      dialogue.situation,
      world,
      snapshot,
      this.state.history,
      freshEvents,
      { capabilities: dialogue.capabilities, personJustEntered },
    );

    const packs = loadSituationPacks();
    const rawMatches = matchSituationPacks(packs, ctx);
    if (!rawMatches.length) return null;

    const eligible = rawMatches.filter(
      (m) => canFirePack(this.state, m.pack.id, m.pack.cooldownMs) && !isPackBlockedDuringBoot(m.pack, dialogue),
    );
    if (!eligible.length) return null;

    const top = eligible[0];
    markPackFired(this.state, top.pack.id);
    const secondary = eligible.slice(1, 3);
    const matched: MatchedSituation[] = [
      { pack: top.pack, score: top.score, confidence: top.confidence },
      ...secondary.map((s) => ({ pack: s.pack, score: s.score, confidence: s.confidence })),
    ];

    const scene = buildScene(matched);
    const message = pickResponseVariant(top.pack, this.state.variations, brain.mood);
    noteVariantUsed(this.state.variations, top.pack.id, message);
    this.state.lastPackId = top.pack.id;

    return {
      packId: top.pack.id,
      message,
      topic: `pack:${top.pack.id}`,
      mood: toneToMood(top.pack.tone),
      reason: `situation-pack:${top.pack.id}`,
      scene,
      interpretation: top.pack.interpretation,
      tone: top.pack.tone,
      priority: top.pack.priority,
    };
  }

  peekTopMatch(
    dialogue: DialogueContext,
    result: VisionResult,
    world: WorldMemory,
    freshEvents: SemanticEvent[],
    personJustEntered: boolean,
  ): { packId: string; tone: SituationPack["tone"]; interpretation: string; sceneLabel: string | null } | null {
    const snapshot = {
      room: dialogue.worldState.room,
      person: {
        present: dialogue.worldState.person.present,
        absentDurationSec: dialogue.worldState.person.absentDurationSec,
        posture: dialogue.personState.posture,
        attention: dialogue.personState.attention,
        activity: dialogue.personState.activity,
        reflecting: dialogue.bodyLanguage.thinking >= 0.5,
      },
      session: dialogue.worldState.session,
      bodyLanguage: dialogue.bodyLanguage,
      situation: dialogue.situation,
      updatedAt: Date.now(),
    };
    const obs = perceiveFromVisionResult(result, world.personPresent, world.lastMotionLevel, Date.now());
    const ctx = buildMatchContext(
      result,
      obs,
      dialogue.personState,
      dialogue.bodyLanguage,
      dialogue.situation,
      world,
      snapshot,
      this.state.history,
      freshEvents,
      { capabilities: dialogue.capabilities, personJustEntered },
    );
    const raw = matchSituationPacks(loadSituationPacks(), ctx);
    if (!raw.length) return null;
    const top = raw[0].pack;
    const scene = buildScene([{ pack: top, score: raw[0].score, confidence: raw[0].confidence }]);
    return {
      packId: top.id,
      tone: top.tone,
      interpretation: top.interpretation,
      sceneLabel: scene?.label ?? top.nameHe ?? top.name,
    };
  }
}

export const evaluateSituationPackDecision = (
  engine: SituationPackEngine,
  brain: CharacterBrain,
  dialogue: DialogueContext,
  result: VisionResult,
  world: WorldMemory,
  freshEvents: SemanticEvent[],
  personJustEntered: boolean,
): CharacterDecision | null => {
  const now = Date.now();
  const msSinceProactive = brain.lastProactiveAt ? now - brain.lastProactiveAt : Number.POSITIVE_INFINITY;
  if (msSinceProactive < CHARACTER_CONFIG.generalCooldownMs) return null;

  const decision = engine.evaluate(brain, dialogue, result, world, freshEvents, personJustEntered);
  if (!decision) return null;

  if (brain.wasTopicMentionedRecently(decision.topic)) return null;

  const urgent = decision.priority === "critical" || decision.priority === "high";
  if (!urgent && msSinceProactive < CHARACTER_CONFIG.urgentCooldownMs / 2) return null;

  return {
    mood: decision.mood,
    message: decision.message,
    topic: decision.topic,
    reason: decision.reason,
  };
};
