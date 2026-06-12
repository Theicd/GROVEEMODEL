/** Phase 7 — teaching / learning assistant: attention loss detection. */

import type { BodyLanguageVector, HumanState, ObservationSet, TeachingState } from "../types";
import { BoolTracker } from "../temporalTracker";

export type TeachingTrackerState = {  lowEngagement: BoolTracker;
  awayGaze: BoolTracker;
};

export const createTeachingTracker = (): TeachingTrackerState => ({
  lowEngagement: new BoolTracker(),
  awayGaze: new BoolTracker(),
});

export const evaluateTeachingState = (
  obs: ObservationSet,
  human: HumanState,
  body: BodyLanguageVector,
  trackers: TeachingTrackerState,
  now = Date.now(),
): TeachingState => {
  if (!obs.personPresent) {
    return {
      attentionLoss: 0,
      engagementDrop: 0,
      likelyDistracted: false,
      ageSec: 0,
      updatedAt: now,
    };
  }

  const lowEng = human.engagement < 0.32 || body.bored >= 0.55;
  const away = human.attention === "away" || human.attention === "unknown";
  const usingPhone = obs.usingPhone;

  const lowTrack = trackers.lowEngagement.update(lowEng, now);
  const awayTrack = trackers.awayGaze.update(away || usingPhone, now);

  let attentionLoss = 0.1;
  if (lowEng) attentionLoss += 0.35;
  if (away || usingPhone) attentionLoss += 0.35;
  if (lowTrack.durationSec >= 15) attentionLoss += 0.2;
  if (body.bored >= 0.5) attentionLoss += 0.15;
  attentionLoss = Math.min(1, attentionLoss);

  const engagementDrop = Math.max(0, 0.65 - human.engagement);
  const ageSec = Math.max(lowTrack.durationSec, awayTrack.durationSec);
  const likelyDistracted = attentionLoss >= 0.55 && ageSec >= 12;

  return {
    attentionLoss,
    engagementDrop,
    likelyDistracted,
    ageSec,
    updatedAt: now,
  };
};

export const resetTeachingTracker = (trackers: TeachingTrackerState): void => {
  trackers.lowEngagement.reset();
  trackers.awayGaze.reset();
};
