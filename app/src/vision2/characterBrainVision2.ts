/** Character decisions driven by Vision 2.0 coach / capabilities (meaning-first). */

import type { CharacterBrain, CharacterDecision, CharacterMood } from "../characterBrain";
import { CHARACTER_CONFIG } from "../characterBrain";
import type { DialogueContext } from "./types";

const coachUtterance = (ctx: DialogueContext): { message: string; mood: CharacterMood; topic: string } | null => {
  const { coach, situation, bodyLanguage, capabilities } = ctx;

  switch (coach.intent) {
    case "offer_support":
      return {
        mood: "curious",
        topic: "coach_support",
        message:
          bodyLanguage.stressed >= 0.75
            ? "נראה שיש קצת מתח — הכל בסדר? אני כאן אם תרצה לפרוק."
            : "שמתי לב שאתה נראה עמוס — רוצה הפסקה קצרה?",
      };
    case "suggest_break":
      return {
        mood: "curious",
        topic: "coach_break",
        message:
          capabilities?.productivity.focusStreakMin >= 45
            ? "אתה ממוקד כבר זמן רב — אל תשכח לשתות מים ולהתמתח."
            : "נראה שאתה עובד ברצף — רגע קטן לנשום יכול לעזור.",
      };
    case "encourage":
      return {
        mood: "excited",
        topic: "coach_encourage",
        message:
          situation.primary === "greeting"
            ? "שלום — טוב לראות אותך. מה קורה?"
            : "אני רואה אנרגיה חיובית — כל הכבוד.",
      };
    default:
      return null;
  }
};

export const evaluateCoachDecision = (
  brain: CharacterBrain,
  ctx: DialogueContext,
): CharacterDecision | null => {
  if (ctx.coach.intent === "none" || ctx.coach.urgency < 0.35) return null;
  if (ctx.worldState.session.sceneAgeSec < 45) return null;
  if (!brain.acquaintanceDone && ctx.worldState.session.sceneAgeSec < 60) return null;

  const now = Date.now();
  const msSinceProactive = brain.lastProactiveAt ? now - brain.lastProactiveAt : Number.POSITIVE_INFINITY;
  const cooldown =
    ctx.coach.intent === "offer_support"
      ? CHARACTER_CONFIG.urgentCooldownMs
      : CHARACTER_CONFIG.generalCooldownMs;

  if (msSinceProactive < cooldown) return null;

  const line = coachUtterance(ctx);
  if (!line) return null;

  if (brain.wasTopicMentionedRecently(line.topic)) return null;

  return {
    mood: line.mood,
    message: line.message,
    topic: line.topic,
    reason: ctx.character.speakReason ?? `coach:${ctx.coach.intent}`,
  };
};
