/** L3 — smoothed human state from observations + world hints. */

import type { WorldMemory } from "../worldMemory";
import { EmaTracker } from "./temporalTracker";
import type { ActivityKind, AttentionKind, EnergyLevel, HumanState, ObservationSet, PostureKind } from "./types";

export type HumanStateEngineState = {
  engagementEma: EmaTracker;
  motionEma: EmaTracker;
};

export const createHumanStateEngineState = (): HumanStateEngineState => ({
  engagementEma: new EmaTracker(),
  motionEma: new EmaTracker(),
});

const mapPosture = (world: WorldMemory): PostureKind => {
  if (world.poseState === "sitting" || world.poseState === "standing") return world.poseState;
  return "unknown";
};

const inferAttention = (obs: ObservationSet, world: WorldMemory): AttentionKind => {
  if (!obs.personPresent) return "unknown";
  if (obs.usingPhone) return "screen";
  if (world.objects.some((o) => /laptop|tv|keyboard|monitor/.test(o)) && obs.motionLevel < 0.12) {
    return "screen";
  }
  if (obs.gazeAtCamera || obs.waving) return "camera";
  if (obs.touchingFace || obs.handOnChin || obs.gazeDown) return "internal";
  if (world.focusHint && /phone|cup/.test(world.focusHint)) return "screen";
  return "unknown";
};

const inferActivity = (obs: ObservationSet, world: WorldMemory, attention: AttentionKind): ActivityKind => {
  if (!obs.personPresent) return "unknown";
  if (obs.waving || obs.thumbsUp) return "social";
  if (obs.holdingCup && !obs.usingPhone) return "resting";
  if (attention === "internal" || obs.touchingFace || obs.handOnChin) return "thinking";
  if (obs.usingPhone || (attention === "screen" && world.objects.some((o) => /laptop|keyboard/.test(o)))) {
    return "working";
  }
  if (obs.motionLevel >= 0.2) return "social";
  return "unknown";
};

const inferEnergy = (motionEma: number): EnergyLevel => {
  if (motionEma >= 0.22) return "high";
  if (motionEma <= 0.06) return "low";
  return "medium";
};

export const updateHumanState = (
  obs: ObservationSet,
  world: WorldMemory,
  engine: HumanStateEngineState,
  now = Date.now(),
): HumanState => {
  const posture = mapPosture(world);
  const attention = inferAttention(obs, world);
  const activity = inferActivity(obs, world, attention);
  const motionEma = engine.motionEma.update(obs.motionLevel);
  const energy = inferEnergy(motionEma);

  let engagementSample = 0;
  if (obs.personPresent) {
    engagementSample = 0.45;
    if (attention === "screen" || attention === "camera") engagementSample += 0.25;
    if (obs.waving || obs.pointing) engagementSample += 0.2;
    if (activity === "working") engagementSample += 0.15;
    if (energy === "low" && activity === "thinking") engagementSample += 0.1;
    engagementSample = Math.min(1, engagementSample);
  }

  const engagement = obs.personPresent ? engine.engagementEma.update(engagementSample) : 0;

  return {
    posture,
    attention,
    activity,
    energy,
    engagement,
    updatedAt: now,
  };
};

export const resetHumanStateEngine = (engine: HumanStateEngineState): void => {
  engine.engagementEma.reset();
  engine.motionEma.reset();
};
