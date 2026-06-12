/** L4 — probabilistic body language vector with temporal age. */

import { DominantStateTracker } from "./temporalTracker";
import type { BodyLanguageVector, HumanState, ObservationSet } from "./types";

export type BodyLanguageModelState = {
  dominant: DominantStateTracker<"focused" | "thinking" | "stressed" | "bored">;
  lastVector: BodyLanguageVector;
};

export const createBodyLanguageModelState = (): BodyLanguageModelState => ({
  dominant: new DominantStateTracker(),
  lastVector: {
    focused: 0,
    thinking: 0,
    stressed: 0,
    bored: 0,
    ageSec: 0,
    updatedAt: 0,
  },
});

const clamp01 = (n: number) => Math.max(0, Math.min(1, n));

export const updateBodyLanguage = (
  obs: ObservationSet,
  human: HumanState,
  durations: {
    faceTouchSec: number;
    handsOnHeadSec: number;
  },
  state: BodyLanguageModelState,
  now = Date.now(),
): BodyLanguageVector => {
  let thinking = 0.08;
  let stressed = 0.06;
  let focused = 0.1;
  let bored = 0.08;

  if (obs.personPresent) {
    if (obs.handOnChin || obs.touchingFace) thinking += 0.35;
    if (obs.gazeDown || human.attention === "internal") thinking += 0.22;
    if (durations.faceTouchSec >= 8) thinking += 0.18;
    if (obs.motionLevel < 0.1 && human.activity === "thinking") thinking += 0.12;

    if (obs.handsOnHead || obs.touchingHead) stressed += 0.38;
    if (durations.handsOnHeadSec >= 5) stressed += 0.22;
    if (obs.motionLevel >= 0.25) stressed += 0.18;
    if (human.energy === "high" && obs.handsOnHead) stressed += 0.12;

    if (human.attention === "screen" && human.activity === "working") focused += 0.42;
    if (human.posture === "sitting" && obs.motionLevel < 0.08) focused += 0.2;
    if (human.engagement >= 0.55) focused += 0.15;

    if (human.engagement < 0.35 && obs.motionLevel < 0.06) bored += 0.35;
    if (human.attention === "away" || human.attention === "unknown") bored += 0.12;
    if (human.energy === "low" && human.activity !== "working") bored += 0.15;
  }

  const raw = {
    thinking: clamp01(thinking),
    stressed: clamp01(stressed),
    focused: clamp01(focused),
    bored: clamp01(bored),
  };

  const entries = Object.entries(raw) as Array<[keyof typeof raw, number]>;
  entries.sort((a, b) => b[1] - a[1]);
  const dominantKey = entries[0][0] as "focused" | "thinking" | "stressed" | "bored";
  const { ageSec } = state.dominant.update(dominantKey, now);

  const vector: BodyLanguageVector = {
    ...raw,
    ageSec,
    updatedAt: now,
  };
  state.lastVector = vector;
  return vector;
};

export const resetBodyLanguageModel = (state: BodyLanguageModelState): void => {
  state.dominant.reset();
  state.lastVector = {
    focused: 0,
    thinking: 0,
    stressed: 0,
    bored: 0,
    ageSec: 0,
    updatedAt: 0,
  };
};
