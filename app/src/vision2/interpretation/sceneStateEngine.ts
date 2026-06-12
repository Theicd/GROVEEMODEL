/** L9 — Scene State Engine: fused activity, not single triggers. */

import type { ObservationSet } from "../types";
import type { BodyLanguageVector, HumanState, RecentChange, SituationState } from "../types";

export type SceneActivity =
  | "idle"
  | "interaction"
  | "attention"
  | "confusion"
  | "social"
  | "movement"
  | "focused_work";

export type SceneStability = "stable" | "changing" | "chaotic";

export type FocusTarget = "person" | "object" | "agent" | "multiple" | "none";

export type SceneState = {
  activity: SceneActivity;
  intensity: number;
  stability: SceneStability;
  focusTarget: FocusTarget;
  engagementLevel: number;
  updatedAt: number;
};

export type SceneStateInput = {
  obs: ObservationSet;
  human: HumanState;
  body: BodyLanguageVector;
  situation: SituationState;
  recentChanges: RecentChange[];
  prevActivity: SceneActivity | null;
  motionDelta: number;
  now?: number;
};

const clamp = (n: number, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, n));

export const computeSceneState = (input: SceneStateInput): SceneState => {
  const { obs, human, body, situation, recentChanges, prevActivity, motionDelta } = input;
  const now = input.now ?? Date.now();

  let activity: SceneActivity = "idle";
  let intensity = 0.2;
  let focusTarget: FocusTarget = obs.personPresent ? "person" : "none";

  if (!obs.personPresent) {
    return {
      activity: "idle",
      intensity: 0.1,
      stability: "stable",
      focusTarget: "none",
      engagementLevel: 0,
      updatedAt: now,
    };
  }

  if (obs.waving || obs.pointing || obs.gazeAtCamera) {
    activity = obs.waving ? "social" : "attention";
    intensity = Math.max(intensity, obs.waving ? 0.75 : 0.6);
    focusTarget = obs.gazeAtCamera || obs.waving ? "agent" : "person";
  } else if (obs.usingPhone) {
    activity = "attention";
    intensity = 0.55;
    focusTarget = "object";
  } else if (body.thinking >= 0.5 || obs.touchingFace || obs.handOnChin) {
    activity = "confusion";
    intensity = Math.max(intensity, body.thinking);
    if (body.thinking >= 0.55 && obs.motionLevel < 0.15) activity = "confusion";
    else activity = "interaction";
  } else if (situation.primary === "working" || body.focused >= 0.55) {
    activity = "focused_work";
    intensity = Math.max(body.focused, 0.5);
    focusTarget = human.attention === "screen" ? "object" : "person";
  } else if (obs.motionLevel >= 0.45) {
    activity = "movement";
    intensity = obs.motionLevel;
  } else if (human.engagement >= 0.45) {
    activity = "interaction";
    intensity = human.engagement;
  }

  const recentCount = recentChanges.filter((c) => c.ageSec <= 8).length;
  let stability: SceneStability = "stable";
  if (motionDelta >= 0.25 || recentCount >= 4) stability = "chaotic";
  else if (recentCount >= 2 || (prevActivity && prevActivity !== activity)) stability = "changing";

  const engagementLevel = Math.round(clamp(human.engagement, 0, 1) * 100);

  return {
    activity,
    intensity: clamp(intensity),
    stability,
    focusTarget,
    engagementLevel,
    updatedAt: now,
  };
};
