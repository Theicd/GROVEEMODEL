/** Phase 7 — social signal scores (0–1). */

import type { AudioSample } from "../sensors/audioSensor";
import type { BodyLanguageVector, HumanState, ObservationSet, SocialAwarenessVector } from "../types";

export const EMPTY_SOCIAL: SocialAwarenessVector = {
  greeting: 0,
  interest: 0,
  agreement: 0,
  disagreement: 0,
  confusion: 0,
  updatedAt: 0,
};

export const evaluateSocialAwareness = (
  obs: ObservationSet,
  human: HumanState,
  body: BodyLanguageVector,
  audio: AudioSample | null,
  now = Date.now(),
): SocialAwarenessVector => {
  if (!obs.personPresent) {
    return { ...EMPTY_SOCIAL, updatedAt: now };
  }

  let greeting = 0.1;
  let interest = 0.15;
  let agreement = 0.05;
  let disagreement = 0.05;
  let confusion = 0.08;

  if (obs.waving || obs.raisedHand) greeting += 0.65;
  if (obs.gazeAtCamera) interest += 0.35;
  if (obs.pointing) interest += 0.25;
  if (human.engagement >= 0.55) interest += 0.2;
  if (obs.thumbsUp) agreement += 0.75;
  if (obs.thumbsDown) disagreement += 0.7;
  if (body.focused >= 0.6 && human.attention === "screen") interest += 0.15;

  if (body.thinking >= 0.55 && body.ageSec >= 8 && human.attention === "internal") {
    confusion += 0.25;
  }
  if (body.bored >= 0.5 && human.engagement < 0.35) {
    confusion += 0.2;
    interest = Math.max(0, interest - 0.1);
  }
  if (audio?.speechDetected) interest += 0.15;

  const clamp = (n: number) => Math.max(0, Math.min(1, n));

  return {
    greeting: clamp(greeting),
    interest: clamp(interest),
    agreement: clamp(agreement),
    disagreement: clamp(disagreement),
    confusion: clamp(confusion),
    updatedAt: now,
  };
};
