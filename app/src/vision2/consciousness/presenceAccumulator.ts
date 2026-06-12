/** Temporal presence accumulation — stable truth only over time. */

import type { PresencePhase, SoulState } from "./types";

const clamp = (n: number, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, n));

export type PresenceAccumConfig = {
  gainPerSec: number;
  decayPerSec: number;
  stableConfidenceMin: number;
  stableDurationSec: number;
  absentConfidenceMax: number;
  phantomMaxDurationSec: number;
};

export const DEFAULT_PRESENCE_CONFIG: PresenceAccumConfig = {
  gainPerSec: 0.55,
  decayPerSec: 0.35,
  stableConfidenceMin: 0.75,
  stableDurationSec: 1.5,
  absentConfidenceMax: 0.25,
  phantomMaxDurationSec: 1.2,
};

export type PresenceAccumState = {
  confidence: number;
  stabilitySec: number;
  stableSince: number | null;
  lastSeenAt: number | null;
  prevSoul: SoulState;
};

export const createPresenceAccumState = (): PresenceAccumState => ({
  confidence: 0,
  stabilitySec: 0,
  stableSince: null,
  lastSeenAt: null,
  prevSoul: "VOID_IDLE",
});

export type PresenceTickResult = {
  confidence: number;
  stabilitySec: number;
  stableSince: number | null;
  phase: PresencePhase;
  soul: SoulState;
  transition: string | null;
};

export const tickPresence = (
  state: PresenceAccumState,
  rawDetected: boolean,
  dtSec: number,
  cfg: PresenceAccumConfig = DEFAULT_PRESENCE_CONFIG,
  now = Date.now(),
): PresenceTickResult => {
  if (rawDetected) {
    state.confidence = clamp(state.confidence + cfg.gainPerSec * dtSec);
    state.stabilitySec += dtSec;
    state.lastSeenAt = now;
  } else {
    state.confidence = clamp(state.confidence - cfg.decayPerSec * dtSec);
    state.stabilitySec = 0;
  }

  const prevSoul = state.prevSoul;
  let soul: SoulState = "VOID_IDLE";
  let phase: PresencePhase = "absent";
  let transition: string | null = null;

  const wasStable = prevSoul === "STABLE_PRESENCE";

  if (state.confidence >= cfg.stableConfidenceMin && state.stabilitySec >= cfg.stableDurationSec) {
    soul = "STABLE_PRESENCE";
    phase = "stable";
    if (!state.stableSince) state.stableSince = now;
  } else if (wasStable && state.confidence < cfg.stableConfidenceMin) {
    soul = "PRESENCE_COLLAPSE";
    phase = "weak";
    state.stableSince = null;
  } else if (rawDetected && state.stabilitySec < cfg.phantomMaxDurationSec) {
    soul = "PHANTOM_DETECTION";
    phase = "weak";
    state.stableSince = null;
  } else if (rawDetected || state.confidence >= cfg.absentConfidenceMax) {
    soul = "PRESENCE_FORMING";
    phase = "unknown";
    state.stableSince = null;
  } else if (state.confidence <= cfg.absentConfidenceMax) {
    soul = "VOID_IDLE";
    phase = "absent";
    state.stableSince = null;
  }

  if (soul !== prevSoul) {
    transition = `${prevSoul} → ${soul}`;
    state.prevSoul = soul;
  }

  return {
    confidence: state.confidence,
    stabilitySec: state.stabilitySec,
    stableSince: state.stableSince,
    phase,
    soul,
    transition,
  };
};

export const interpretPresence = (soul: SoulState, confidence: number): string => {
  switch (soul) {
    case "STABLE_PRESENCE":
      return "נוכחות אנושית התייצבה בשדה — אין סטייה בפרשנות.";
    case "PRESENCE_FORMING":
      return "דפוס חוזר. המציאות נוטה להכריע לטובת נוכחות.";
    case "PHANTOM_DETECTION":
      return "סימנים חלקיים של נוכחות. לא ניתן לאשר זהות מציאותית.";
    case "PRESENCE_COLLAPSE":
      return "קריסה פתאומית של ישות שהייתה יציבה.";
    default:
      return confidence < 0.3
        ? "המרחב קיים ללא עדות לקיום אנושי מתמשך."
        : "אין תודעה מתמשכת מזוהה — רק רעש.";
  }
};
