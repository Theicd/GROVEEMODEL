/** Cold affect — certainty/curiosity/alertness from temporal state. */

import type { HalGlobalState, SoulState } from "./types";

const clamp01 = (n: number) => Math.max(0, Math.min(1, n));

export const updateAffect = (
  affect: HalGlobalState["affect"],
  perception: HalGlobalState["perception"],
  soul: SoulState,
  stabilitySec: number,
  dtSec: number,
): HalGlobalState["affect"] => {
  let curiosity = affect.curiosity;
  let certainty = affect.certainty;
  let alertness = affect.alertness;

  certainty = clamp01(perception.certainty * 0.7 + (soul === "STABLE_PRESENCE" ? 0.25 : 0));

  if (soul === "PHANTOM_DETECTION" || soul === "PRESENCE_FORMING") {
    curiosity = clamp01(curiosity + 0.08 * dtSec);
    alertness = clamp01(alertness + 0.12 * dtSec);
  }

  if (soul === "STABLE_PRESENCE") {
    curiosity = clamp01(curiosity + 0.03 * dtSec);
    alertness = clamp01(alertness - 0.02 * dtSec);
  }

  if (soul === "VOID_IDLE") {
    curiosity = clamp01(curiosity + 0.02 * dtSec);
    alertness = clamp01(alertness - 0.04 * dtSec);
  }

  if (soul === "PRESENCE_COLLAPSE") {
    alertness = clamp01(alertness + 0.25);
    certainty = clamp01(certainty - 0.15);
  }

  if (stabilitySec > 3) curiosity = clamp01(curiosity - 0.02 * dtSec);

  return {
    curiosity: clamp01(curiosity * 0.995 + 0.002),
    certainty,
    alertness: clamp01(alertness * 0.99 + 0.01),
  };
};

export const updatePerception = (
  confidence: number,
  soul: SoulState,
  flickerRate: number,
): HalGlobalState["perception"] => {
  const ambiguity = clamp01(1 - confidence + (soul === "PHANTOM_DETECTION" ? 0.35 : 0));
  const certainty = clamp01(confidence * (soul === "STABLE_PRESENCE" ? 1 : 0.65));
  const noiseFloor = clamp01(0.1 + flickerRate * 0.4);
  return { certainty, ambiguity, noiseFloor };
};
