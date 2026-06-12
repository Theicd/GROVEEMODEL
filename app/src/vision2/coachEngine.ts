/** L6 — coaching intents from state (not raw detections). */

import type {
  BodyLanguageVector,
  CoachState,
  HumanState,
  SessionModel,
  SituationState,
} from "./types";

export const evaluateCoach = (
  human: HumanState,
  body: BodyLanguageVector,
  situation: SituationState,
  session: SessionModel,
): CoachState => {
  if (session.sceneAgeSec < 25) {
    return { intent: "none", reason: "", urgency: 0 };
  }

  if (!human.updatedAt || (human.engagement === 0 && human.posture === "unknown")) {
    return { intent: "none", reason: "", urgency: 0 };
  }

  if (body.stressed >= 0.75 && session.faceTouchDurationSec >= 0) {
    const headStress = body.stressed >= 0.8;
    if (headStress) {
      return {
        intent: "offer_support",
        reason: "Elevated stress signals with sustained tension.",
        urgency: clamp(body.stressed),
      };
    }
  }

  if (body.stressed >= 0.72 && body.ageSec >= 12 && session.sceneAgeSec >= 20) {
    return {
      intent: "offer_support",
      reason: "Prolonged stress pattern detected.",
      urgency: clamp(body.stressed * 0.9),
    };
  }

  if (
    body.focused >= 0.78 &&
    session.workSessionMin >= 45 &&
    situation.primary === "working"
  ) {
    return {
      intent: "suggest_break",
      reason: "Long focused work without break cues.",
      urgency: clamp(0.55 + session.workSessionMin / 120),
    };
  }

  if (body.focused >= 0.8 && session.workSessionMin >= 25 && situation.primary === "working") {
    return {
      intent: "suggest_break",
      reason: "Extended focus — hydration or pause may help.",
      urgency: 0.45,
    };
  }

  if (situation.primary === "greeting" || (human.activity === "social" && body.bored < 0.4)) {
    if (body.stressed < 0.5) {
      return { intent: "none", reason: "", urgency: 0 };
    }
  }

  if (situation.primary === "greeting" && human.engagement >= 0.5) {
    return { intent: "encourage", reason: "Positive social engagement.", urgency: 0.35 };
  }

  return { intent: "none", reason: "", urgency: 0 };
};

const clamp = (n: number) => Math.max(0, Math.min(1, n));
