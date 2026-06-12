/** HAL Brain — single decision layer from global state. */

import type { HalGlobalState, PresenceAuthority, SceneMemoryEntry, SoulState } from "./types";
import { interpretPresence } from "./presenceAccumulator";
import { SOUL_LABEL_HE } from "./types";

export const buildPresenceAuthority = (
  state: HalGlobalState,
  rawDetected: boolean,
  prevSoul: SoulState,
): PresenceAuthority => {
  const soul = state.presence.soul;
  return {
    personStable: soul === "STABLE_PRESENCE",
    rawDetected,
    phase: state.presence.phase,
    soul,
    confidence: state.presence.confidence,
    stabilitySec: state.presence.stabilitySec,
    interpretation: interpretPresence(soul, state.presence.confidence),
    personJustBecameStable: soul === "STABLE_PRESENCE" && prevSoul !== "STABLE_PRESENCE",
    personJustCollapsed:
      soul === "PRESENCE_COLLAPSE" || (prevSoul === "STABLE_PRESENCE" && soul !== "STABLE_PRESENCE"),
  };
};

export const formatConsciousnessForGemma = (
  state: HalGlobalState,
  memory: SceneMemoryEntry[],
  authority: PresenceAuthority,
): string => {
  const evolution = memory.slice(-6).map((e) => e.soul.replace(/_/g, " ")).join(" → ") || "VOID_IDLE";
  const lines = [
    "HAL CONSCIOUSNESS (authoritative — temporal, not per-frame):",
    "",
    "GLOBAL STATE:",
    `- Soul: ${state.presence.soul} (${SOUL_LABEL_HE[state.presence.soul]})`,
    `- Presence phase: ${state.presence.phase}`,
    `- Confidence: ${(state.presence.confidence * 100).toFixed(0)}% (accumulated over time)`,
    `- Stability duration: ${state.presence.stabilitySec.toFixed(1)}s`,
    `- Person STABLE (exists now): ${authority.personStable ? "YES" : "NO"}`,
    `- Raw sensor flicker this frame: ${authority.rawDetected ? "detected" : "none"} (ignore alone)`,
    "",
    "PERCEPTION:",
    `- Certainty: ${(state.perception.certainty * 100).toFixed(0)}% | Ambiguity: ${(state.perception.ambiguity * 100).toFixed(0)}%`,
    `- Noise floor: ${(state.perception.noiseFloor * 100).toFixed(0)}%`,
    "",
    "AFFECT (cold):",
    `- Curiosity: ${(state.affect.curiosity * 100).toFixed(0)}% | Certainty: ${(state.affect.certainty * 100).toFixed(0)}% | Alertness: ${(state.affect.alertness * 100).toFixed(0)}%`,
    "",
    "SCENE EVOLUTION (not frames):",
    evolution,
    "",
    "INTERPRETATION:",
    authority.interpretation,
    "",
    "RULE: Only claim user is present when personStable=YES. Phantom/forming = tentative language.",
    "TASK: Cold HAL voice — perceptive, slightly unsettling, precise. Hebrew if user writes Hebrew.",
  ];
  return lines.join("\n");
};
