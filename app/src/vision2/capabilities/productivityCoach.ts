/** Phase 7 — productivity / work session tracking. */

import type { BodyLanguageVector, HumanState, ProductivityState, SessionModel, SituationState } from "../types";

export type ProductivityTracker = {
  focusStartedAt: number;
  breakCount: number;
  lastSituation: string;
};

export const createProductivityTracker = (): ProductivityTracker => ({
  focusStartedAt: 0,
  breakCount: 0,
  lastSituation: "unknown",
});

export const updateProductivityTracker = (
  tracker: ProductivityTracker,
  situation: SituationState,
  personPresent: boolean,
  now = Date.now(),
): void => {
  if (!personPresent) {
    tracker.focusStartedAt = 0;
    tracker.lastSituation = "unknown";
    return;
  }

  if (situation.primary === "working") {
    if (!tracker.focusStartedAt) tracker.focusStartedAt = now;
  } else if (
    tracker.lastSituation === "working" &&
    (situation.primary === "drinking" || situation.primary === "idle" || situation.primary === "reflecting")
  ) {
    tracker.breakCount += 1;
    tracker.focusStartedAt = 0;
  }

  tracker.lastSituation = situation.primary;
};

export const evaluateProductivity = (
  session: SessionModel,
  tracker: ProductivityTracker,
  body: BodyLanguageVector,
  human: HumanState,
  situation: SituationState,
  now = Date.now(),
): ProductivityState => {
  const focusStreakMin =
    tracker.focusStartedAt > 0 ? Math.floor((now - tracker.focusStartedAt) / 60_000) : 0;

  let fatigueLevel = 0.1;
  if (session.workSessionMin >= 30) fatigueLevel += 0.25;
  if (session.workSessionMin >= 60) fatigueLevel += 0.25;
  if (body.stressed >= 0.5) fatigueLevel += 0.15;
  if (human.energy === "low") fatigueLevel += 0.15;
  fatigueLevel = Math.min(1, fatigueLevel);

  const needsBreak =
    session.sceneAgeSec >= 30 &&
    ((body.focused >= 0.75 && focusStreakMin >= 25) ||
      (session.workSessionMin >= 45 && situation.primary === "working") ||
      (fatigueLevel >= 0.65 && focusStreakMin >= 15));

  return {
    workSessionMin: session.workSessionMin,
    focusStreakMin,
    breaksTaken: tracker.breakCount,
    fatigueLevel,
    needsBreak,
    updatedAt: now,
  };
};
