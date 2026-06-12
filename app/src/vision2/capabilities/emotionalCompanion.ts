/** Phase 7 — emotional companion: support / encouragement thresholds. */

import type {
  BodyLanguageVector,
  CoachState,
  EmotionalState,
  HumanState,
  ProductivityState,
  SessionModel,
  SituationState,
  SocialAwarenessVector,
} from "../types";
import { evaluateCoach } from "../coachEngine";

export const evaluateEmotionalState = (  human: HumanState,
  body: BodyLanguageVector,
  social: SocialAwarenessVector,
  productivity: ProductivityState,
  now = Date.now(),
): EmotionalState => {
  let supportNeeded = body.stressed * 0.7 + (1 - human.engagement) * 0.2;
  if (productivity.fatigueLevel >= 0.6) supportNeeded += 0.15;
  supportNeeded = Math.min(1, supportNeeded);

  let encouragementOpportunity = social.agreement * 0.5 + social.greeting * 0.35;
  if (human.activity === "social") encouragementOpportunity += 0.2;
  encouragementOpportunity = Math.min(1, encouragementOpportunity);

  const motivation = Math.min(
    1,
    human.engagement * 0.5 + body.focused * 0.3 + (1 - body.bored) * 0.2,
  );

  const reflection = Math.min(1, body.thinking * 0.85 + (human.activity === "thinking" ? 0.15 : 0));

  return {
    supportNeeded,
    encouragementOpportunity,
    motivation,
    reflection,
    updatedAt: now,
  };
};

/** Merges base coach with emotional + productivity signals. */
export const evaluateEmotionalCoach = (
  human: HumanState,
  body: BodyLanguageVector,
  situation: SituationState,
  session: SessionModel,
  social: SocialAwarenessVector,
  emotional: EmotionalState,
  productivity: ProductivityState,
): CoachState => {
  const base = evaluateCoach(human, body, situation, session);
  if (base.intent !== "none") return base;

  if (session.sceneAgeSec < 30) {
    return { intent: "none", reason: "", urgency: 0 };
  }

  if (emotional.supportNeeded >= 0.72 && body.stressed >= 0.6 && body.ageSec >= 10) {
    return {
      intent: "offer_support",
      reason: "Emotional companion: sustained stress — gentle check-in.",
      urgency: emotional.supportNeeded,
    };
  }

  if (productivity.needsBreak && productivity.fatigueLevel >= 0.55) {
    return {
      intent: "suggest_break",
      reason: "Productivity coach: fatigue rising during focus block.",
      urgency: Math.min(1, productivity.fatigueLevel + 0.2),
    };
  }

  if (emotional.encouragementOpportunity >= 0.65 && social.greeting >= 0.5) {
    return {
      intent: "encourage",
      reason: "Positive social moment — reinforce connection.",
      urgency: emotional.encouragementOpportunity * 0.8,
    };
  }

  if (social.agreement >= 0.7) {
    return {
      intent: "encourage",
      reason: "Agreement signal detected.",
      urgency: 0.4,
    };
  }

  return { intent: "none", reason: "", urgency: 0 };
};
