/** HAL dynamic mood — derives character mood + tone from perception layers (smoothed). */

import type { CharacterMood } from "../characterBrain";
import type { SituationTone } from "../situation-packs/types";
import type { BodyLanguageVector, HumanState, SituationState } from "./types";

export type HalMoodState = {
  mood: CharacterMood;
  tone: SituationTone;
  moodLabelHe: string;
  situationPrimary: string;
  situationConfidence: number;
  personPresent: boolean;
  sceneLabel: string | null;
  activePackId: string | null;
  interpretation: string | null;
  engagement: number;
  stressLevel: number;
  updatedAt: number;
};

export type HalMoodInput = {
  human: HumanState;
  body: BodyLanguageVector;
  situation: SituationState;
  personPresent: boolean;
  packHint?: {
    packId: string;
    tone: SituationTone;
    interpretation: string;
    sceneLabel: string | null;
  } | null;
};

const MOOD_LABELS: Record<CharacterMood, string> = {
  observing: "תצפית",
  curious: "סקרן",
  bored: "משועמם",
  excited: "מעורב",
};

const toneFromSignals = (
  body: BodyLanguageVector,
  human: HumanState,
  packTone?: SituationTone,
): SituationTone => {
  if (packTone) return packTone;
  if (body.stressed >= 0.55) return "calm";
  if (body.bored >= 0.5) return "soft";
  if (body.thinking >= 0.55) return "quiet";
  if (body.focused >= 0.6) return "soft";
  if (human.engagement >= 0.6) return "curious";
  return "neutral";
};

const moodFromSignals = (
  body: BodyLanguageVector,
  human: HumanState,
  situation: SituationState,
  packTone?: SituationTone,
): CharacterMood => {
  if (packTone === "positive" || packTone === "playful" || packTone === "warm") return "excited";
  if (packTone === "curious" || packTone === "analytical") return "curious";
  if (body.bored >= 0.55 && human.engagement < 0.4) return "bored";
  if (body.thinking >= 0.5 || situation.primary === "reflecting") return "curious";
  if (body.stressed >= 0.5 || packTone === "supportive" || packTone === "calm") return "observing";
  if (human.engagement >= 0.65 || situation.primary === "greeting") return "curious";
  if (body.focused >= 0.55) return "observing";
  return "observing";
};

export class HalMoodEngine {
  private state: HalMoodState = {
    mood: "observing",
    tone: "neutral",
    moodLabelHe: MOOD_LABELS.observing,
    situationPrimary: "unknown",
    situationConfidence: 0,
    personPresent: false,
    sceneLabel: null,
    activePackId: null,
    interpretation: null,
    engagement: 0,
    stressLevel: 0,
    updatedAt: 0,
  };

  private lastMoodChangeAt = 0;

  reset(): void {
    this.state = {
      mood: "observing",
      tone: "neutral",
      moodLabelHe: MOOD_LABELS.observing,
      situationPrimary: "unknown",
      situationConfidence: 0,
      personPresent: false,
      sceneLabel: null,
      activePackId: null,
      interpretation: null,
      engagement: 0,
      stressLevel: 0,
      updatedAt: 0,
    };
    this.lastMoodChangeAt = 0;
  }

  getState(): HalMoodState {
    return this.state;
  }

  update(input: HalMoodInput, now = Date.now()): HalMoodState {
    const packTone = input.packHint?.tone;
    const nextMood = moodFromSignals(input.body, input.human, input.situation, packTone);
    const nextTone = toneFromSignals(input.body, input.human, packTone);

    let mood = this.state.mood;
    if (nextMood !== mood) {
      const holdMs = 2_500;
      const urgent =
        input.packHint != null ||
        input.situation.primary === "greeting" ||
        input.body.stressed >= 0.7;
      if (urgent || now - this.lastMoodChangeAt >= holdMs) {
        mood = nextMood;
        this.lastMoodChangeAt = now;
      }
    }

    this.state = {
      mood,
      tone: nextTone,
      moodLabelHe: MOOD_LABELS[mood],
      situationPrimary: input.situation.primary,
      situationConfidence: input.situation.confidence,
      personPresent: input.personPresent,
      sceneLabel: input.packHint?.sceneLabel ?? this.state.sceneLabel,
      activePackId: input.packHint?.packId ?? null,
      interpretation: input.packHint?.interpretation ?? input.situation.description,
      engagement: input.human.engagement,
      stressLevel: input.body.stressed,
      updatedAt: now,
    };

    return this.state;
  }
}

/** Authoritative facts for chat — Gemma must not contradict these. */
export const buildHalChatFactsBlock = (hal: HalMoodState): string => {
  const lines = [
    "HAL LIVE FACTS (authoritative — from camera pipeline, NOT from uploaded files):",
    `personInFrame: ${hal.personPresent}`,
    `situation: ${hal.situationPrimary} (${Math.round(hal.situationConfidence * 100)}%)`,
    `posture/attention: use HAL_PERCEPTION_JSON personState`,
    `halMood: ${hal.mood} (${hal.moodLabelHe}), tone: ${hal.tone}`,
  ];
  if (hal.sceneLabel) lines.push(`activeScene: ${hal.sceneLabel}`);
  if (hal.interpretation) lines.push(`interpretation: ${hal.interpretation}`);
  if (hal.personPresent) {
    lines.push(
      "VISIBILITY RULE: If user asks whether you see them / a person — answer YES. Describe posture/focus tentatively from personState.",
    );
  } else {
    lines.push(
      "VISIBILITY RULE: If user asks whether you see them — answer NO (person not confirmed in frame).",
    );
  }
  return lines.join("\n");
};
