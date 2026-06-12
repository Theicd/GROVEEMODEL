/** L5 — situation state with hysteresis (min dwell). */

import type { HumanState, ObservationSet, SituationPrimary, SituationState } from "./types";

const MIN_DWELL_MS = 2_500;

export type SituationEngineState = {
  primary: SituationPrimary;
  since: number;
  confidence: number;
};

export const createSituationEngineState = (): SituationEngineState => ({
  primary: "unknown",
  since: 0,
  confidence: 0,
});

const describe = (primary: SituationPrimary): string => {
  switch (primary) {
    case "working":
      return "Person appears focused on work or a screen.";
    case "drinking":
      return "Person appears to be taking a drink break.";
    case "using_phone":
      return "Person appears to be using a phone.";
    case "greeting":
      return "Person appears to be greeting or seeking attention.";
    case "reflecting":
      return "Person appears thoughtful or internally focused.";
    case "idle":
      return "Person is present with low activity.";
    default:
      return "Situation unclear.";
  }
};

const inferCandidate = (obs: ObservationSet, human: HumanState): { primary: SituationPrimary; confidence: number } => {
  if (!obs.personPresent) return { primary: "unknown", confidence: 0 };

  if (obs.waving || obs.raisedHand) {
    return { primary: "greeting", confidence: 0.88 };
  }
  if (obs.usingPhone) {
    return { primary: "using_phone", confidence: 0.84 };
  }
  if (obs.holdingCup) {
    return { primary: "drinking", confidence: 0.8 };
  }
  if (human.activity === "working" || (human.attention === "screen" && human.activity !== "resting")) {
    return { primary: "working", confidence: 0.78 };
  }
  if (human.activity === "thinking" || obs.touchingFace || obs.handOnChin) {
    return { primary: "reflecting", confidence: 0.72 };
  }
  if (human.engagement < 0.3 && obs.motionLevel < 0.08) {
    return { primary: "idle", confidence: 0.65 };
  }
  return { primary: "unknown", confidence: 0.4 };
};

export const updateSituation = (
  obs: ObservationSet,
  human: HumanState,
  state: SituationEngineState,
  now = Date.now(),
): SituationState => {
  const candidate = inferCandidate(obs, human);

  if (state.primary === "unknown" || state.since === 0) {
    state.primary = candidate.primary;
    state.confidence = candidate.confidence;
    state.since = now;
  } else if (candidate.primary !== state.primary) {
    if (now - state.since >= MIN_DWELL_MS && candidate.confidence >= state.confidence - 0.05) {
      state.primary = candidate.primary;
      state.confidence = candidate.confidence;
      state.since = now;
    }
  } else {
    state.confidence = Math.max(state.confidence * 0.92, candidate.confidence);
  }

  return {
    primary: state.primary,
    confidence: state.confidence,
    description: describe(state.primary),
    updatedAt: now,
  };
};

export const resetSituationEngine = (state: SituationEngineState): void => {
  state.primary = "unknown";
  state.since = 0;
  state.confidence = 0;
};
