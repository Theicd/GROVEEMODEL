/** First contact — identify who is present and open conversation. */

import type { CharacterBrain, CharacterDecision } from "../characterBrain";
import type { EntityProfile } from "./entityProfile";
import type { DialogueContext } from "./types";

const genderWord = (entity: EntityProfile): string => {
  if (entity.gender === "male") return "גבר";
  if (entity.gender === "female") return "אישה";
  return "מישהו";
};

const childWord = (entity: EntityProfile): string => {
  if (entity.gender === "female") return "ילדה";
  if (entity.gender === "male") return "ילד";
  return "ילד/ה";
};

export const buildAcquaintanceMessage = (entity: EntityProfile): string => {
  const age = entity.ageEstimate;

  if (entity.segment === "child" && age) {
    return `היי! אני רואה ${childWord(entity)} בערך בן/בת ${age} — איך קוראים לך? רוצה שנשחק משהו ביחד?`;
  }
  if (entity.segment === "teen" && age) {
    return `שלום — נראה שיש כאן ${genderWord(entity)} בערך בן/בת ${age}. איך קוראים לך? במה תרצה/י לעסוק?`;
  }
  if (entity.segment === "adult" && entity.confidence >= 0.5) {
    return `שלום. אני מזהה ${genderWord(entity)}${age ? ` בערך בן/בת ${age}` : ""} — נעים להכיר. איך קוראים לך, ומה נעשה היום?`;
  }
  if (entity.faceObservations >= 1) {
    return `היי — אני מתחיל להכיר אותך. איך קוראים לך? את/ה ילד/ה או מבוגר/ת?`;
  }
  return `היי 👋 — אני כאן. מי נמצא מולי?`;
};

export const evaluateAcquaintanceDecision = (
  brain: CharacterBrain,
  dialogue: DialogueContext,
  entity: EntityProfile | null,
): CharacterDecision | null => {
  if (brain.acquaintanceDone) return null;
  if (!dialogue.consciousness?.personStable && !dialogue.worldState.person.present) return null;

  const stableSec = dialogue.consciousness?.stabilitySec ?? 0;
  if (stableSec < 1.2) return null;

  const sceneAge = dialogue.worldState.session.sceneAgeSec ?? 0;
  if (sceneAge > 45) {
    brain.acquaintanceDone = true;
    return null;
  }

  const profile = entity ?? dialogue.entity ?? null;
  const ready =
    profile &&
    (profile.faceObservations >= 2 || (profile.confidence >= 0.35 && stableSec >= 2));

  if (!ready && sceneAge < 8) return null;
  if (!ready && sceneAge >= 8) {
    return {
      mood: "curious",
      message: buildAcquaintanceMessage(
        profile ?? {
          ageEstimate: null,
          ageRawEstimate: null,
          gender: "unknown",
          segment: "unknown",
          emotion: null,
          emotionScore: 0,
          gazeAtCamera: false,
          engagement: "neutral",
          confidence: 0,
          faceObservations: 0,
          updatedAt: Date.now(),
        },
      ),
      topic: "acquaintance:intro",
      reason: "acquaintance:no_face_yet",
    };
  }

  if (brain.wasTopicMentionedRecently("acquaintance:intro")) return null;

  return {
    mood: "curious",
    message: buildAcquaintanceMessage(profile!),
    topic: "acquaintance:intro",
    reason: "acquaintance:first_contact",
  };
};
