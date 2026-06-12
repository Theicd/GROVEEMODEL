/** L10 — build + serialize DialogueContext for Gemma (no raw sensors). */

import type { CharacterBrain, CharacterMood } from "../characterBrain";
import type { SemanticEvent, WorldMemory } from "../worldMemory";
import { buildHalChatFactsBlock } from "./halMoodEngine";
import { formatEntityForGemma } from "./entityProfile";
import { EMPTY_INTERPRETATION } from "./interpretationDefaults";
import type {
  AudioContext,
  CapabilityContext,
  CoachState,
  DialogueContext,
  HalLayer,
  HumanState,
  RecentChange,
  WorldSnapshot,
} from "./types";

export const DIALOGUE_CONTEXT_SYSTEM_HINT = `You receive HAL CONSCIOUSNESS + NARRATIVE — temporal interpreted reality, NOT raw per-frame sensors.
personStable=YES / worldState.person.present=true means a human has STABILIZED over time (authoritative).
Raw sensor flicker (phantom/forming) = tentative language only — never deny stable presence.
Use META EVENTS and INTERPRETATION for meaning. INTERNAL MONOLOGUE is for reasoning only — never quote it.
Respond as cold HAL: perceptive, slightly unsettling, precise. Hebrew when user writes Hebrew.
Never mention YOLO, landmarks, bounding boxes, or deny camera access when consciousness says personStable=YES.`;

export const serializeDialogueContext = (ctx: DialogueContext): string =>
  JSON.stringify(ctx, null, 0);

export const buildRecentChanges = (
  world: WorldMemory,
  snapshot: WorldSnapshot,
  prevSituation: string | null,
  now = Date.now(),
): RecentChange[] => {
  const out: RecentChange[] = [];

  for (const ev of world.lastChanges.slice(0, 6)) {
    const ageSec = Math.max(0, Math.floor((now - ev.ts) / 1000));
    let kind: RecentChange["kind"] = "activity_change";
    if (ev.type === "person_entered" || ev.type === "user_returned") kind = "entered";
    else if (ev.type === "person_left") kind = "left";
    else if (ev.subject === "wave") kind = "greeting";
    out.push({
      kind,
      ageSec,
      confidence: ev.significant ? 0.85 : 0.55,
      text: ev.text,
    });
  }

  if (prevSituation && prevSituation !== snapshot.situation.primary) {
    out.unshift({
      kind: "shifted_focus",
      ageSec: 0,
      confidence: snapshot.situation.confidence,
      text: `Situation shifted toward ${snapshot.situation.primary}.`,
    });
  }

  if (snapshot.bodyLanguage.stressed >= 0.7 && snapshot.bodyLanguage.ageSec >= 10) {
    out.unshift({
      kind: "stress_rising",
      ageSec: Math.round(snapshot.bodyLanguage.ageSec),
      confidence: snapshot.bodyLanguage.stressed,
      text: "Stress signals building over time.",
    });
  }

  if (snapshot.session.workSessionMin >= 45 && snapshot.situation.primary === "working") {
    out.unshift({
      kind: "break_needed",
      ageSec: snapshot.session.workSessionMin * 60,
      confidence: 0.75,
      text: "Extended work session without break.",
    });
  }

  return out.slice(0, 8);
};

export const buildDialogueContext = (params: {
  world: WorldMemory;
  snapshot: WorldSnapshot;
  human: HumanState;
  coach: CoachState;
  capabilities: CapabilityContext;
  audio: AudioContext;
  episodicSummary: string[];
  character: CharacterBrain;
  shouldSpeak?: boolean;
  speakReason?: string;
  recentChanges?: RecentChange[];
  hal?: HalLayer;
  entity?: import("./entityProfile").EntityProfile | null;
}): DialogueContext => {
  const { snapshot, human, coach, capabilities, audio, episodicSummary, character } = params;

  const hal: HalLayer = params.hal ?? {
    mood: character.mood as CharacterMood,
    tone: "neutral",
    moodLabelHe: "תצפית",
    personPresent: snapshot.person.present,
    sceneLabel: null,
    activePackId: null,
    interpretation: snapshot.situation.description,
    stressLevel: snapshot.bodyLanguage.stressed,
    engagement: human.engagement,
  };

  return {
    worldState: {
      room: snapshot.room,
      person: {
        present: snapshot.person.present,
        absentDurationSec: snapshot.person.absentDurationSec,
      },
      session: snapshot.session,
    },
    personState: human,
    bodyLanguage: snapshot.bodyLanguage,
    situation: snapshot.situation,
    recentChanges: params.recentChanges ?? [],
    coach,
    capabilities,
    audio,
    character: {
      mood: hal.mood,
      shouldSpeak: params.shouldSpeak ?? false,
      speakReason: params.speakReason,
    },
    hal,
    interpretation: EMPTY_INTERPRETATION,
    episodicSummary,
    entity: params.entity ?? null,
  };
};

/** Finger count answer block — derived state only, no landmarks. */
export const buildFingerAnswerBlock = (extendedTotal: number, gestureHint?: string): string => {
  if (extendedTotal <= 0) {
    return "HAND STATE: no clear extended fingers visible. Ask user to show hand closer to camera.";
  }
  const lines = [
    "FINGER ANSWER (from perception state only):",
    `Total extended fingers visible: ${extendedTotal}`,
  ];
  if (gestureHint) lines.push(`Gesture hint: ${gestureHint}`);
  return lines.join("\n");
};

export const buildHalFactsFromDialogue = (ctx: DialogueContext): string =>
  buildHalChatFactsBlock({
    mood: ctx.hal.mood,
    tone: ctx.hal.tone as import("../situation-packs/types").SituationTone,
    moodLabelHe: ctx.hal.moodLabelHe,
    situationPrimary: ctx.situation.primary,
    situationConfidence: ctx.situation.confidence,
    personPresent: ctx.hal.personPresent,
    sceneLabel: ctx.hal.sceneLabel,
    activePackId: ctx.hal.activePackId,
    interpretation: ctx.hal.interpretation,
    engagement: ctx.hal.engagement,
    stressLevel: ctx.hal.stressLevel,
    updatedAt: Date.now(),
  });

export const buildDeepVisionContextBlock = (ctx: DialogueContext | null): string => {
  if (!ctx) return "HAL perception state: not ready yet.";
  if (ctx.interpretation?.gemmaBlock) return ctx.interpretation.gemmaBlock;
  return `HAL perception state (structured):\n${serializeDialogueContext(ctx)}`;
};

/** Primary chat injection — consciousness-first, then entity, then narrative. */
export const buildGemmaContextBlock = (ctx: DialogueContext): string => {
  const parts: string[] = [];
  if (ctx.consciousness?.gemmaBlock) parts.push(ctx.consciousness.gemmaBlock);
  if (ctx.entity) parts.push(formatEntityForGemma(ctx.entity));
  parts.push(buildHalFactsFromDialogue(ctx));
  if (ctx.interpretation?.gemmaBlock) parts.push(ctx.interpretation.gemmaBlock);
  else parts.push(buildDeepVisionContextBlock(ctx));
  return parts.join("\n\n");
};

export const eventsToSpeakReason = (events: SemanticEvent[]): string | undefined => {
  const primary = events.find((e) => e.significant);
  return primary ? `situation:${primary.subject ?? primary.type}` : undefined;
};
