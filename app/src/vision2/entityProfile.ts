/** Temporal entity profile from face + body — who is in front of HAL. */

import type { EmotionScores } from "../vision-lab/core/types";
import type { BodyLanguageVector, HumanState } from "./types";

export type GenderEstimate = "male" | "female" | "unknown";
export type AgeSegment = "child" | "teen" | "adult" | "unknown";
export type EngagementMode = "focused" | "bored" | "drowsy" | "tense" | "neutral";

export type EntityProfile = {
  /** Display/speech age (bias-corrected). */
  ageEstimate: number | null;
  /** Raw face-model average before correction. */
  ageRawEstimate: number | null;
  gender: GenderEstimate;
  segment: AgeSegment;
  emotion: string | null;
  emotionScore: number;
  gazeAtCamera: boolean;
  engagement: EngagementMode;
  confidence: number;
  faceObservations: number;
  updatedAt: number;
};

export type EntityProfileState = {
  ageSamples: number[];
  maleScore: number;
  femaleScore: number;
  faceCount: number;
  lastEmotion: string | null;
  lastEmotionScore: number;
};

export const createEntityProfileState = (): EntityProfileState => ({
  ageSamples: [],
  maleScore: 0,
  femaleScore: 0,
  faceCount: 0,
  lastEmotion: null,
  lastEmotionScore: 0,
});

export const ADULT_AGE_BIAS_THRESHOLD = 40;
/** Face model over-estimates age above ~40 — show 80% in UI and speech. */
export const ADULT_AGE_BIAS_FACTOR = 0.8;

/** Raw sensor age → age shown in HUD, Gemma, and proactive speech. Youth: unchanged. */
export const correctAgeForDisplay = (rawAge: number): number => {
  if (rawAge <= 0) return 0;
  const rounded = Math.round(rawAge);
  if (rounded > ADULT_AGE_BIAS_THRESHOLD) {
    return Math.max(1, Math.round(rounded * ADULT_AGE_BIAS_FACTOR));
  }
  return rounded;
};

const segmentFromAge = (age: number | null): AgeSegment => {
  if (age == null || age <= 0) return "unknown";
  if (age < 13) return "child";
  if (age < 18) return "teen";
  return "adult";
};

const normGender = (g: string): GenderEstimate => {
  const x = g.toLowerCase();
  if (x.includes("male") && !x.includes("female")) return "male";
  if (x.includes("female") || x.includes("woman")) return "female";
  return "unknown";
};

export const inferEngagementMode = (
  body: BodyLanguageVector,
  human: HumanState,
  emotion: EmotionScores | null | undefined,
): EngagementMode => {
  if (emotion?.dominant === "angry" && (emotion.dominantScore ?? 0) >= 0.45) return "tense";
  if (body.bored >= 0.55 && human.engagement < 0.4) return "bored";
  if (body.focused >= 0.6 && human.engagement >= 0.45) return "focused";
  if (
    human.energy === "low" &&
    body.bored >= 0.35 &&
    (emotion?.dominant === "neutral" || emotion?.dominant === "sad")
  ) {
    return "drowsy";
  }
  return "neutral";
};

export const updateEntityProfile = (
  state: EntityProfileState,
  params: {
    face?: { estimatedAge: number; estimatedGender: string; gazeDirection: string } | null;
    emotion?: EmotionScores | null;
    body: BodyLanguageVector;
    human: HumanState;
    personStable: boolean;
  },
  now = Date.now(),
): EntityProfile => {
  if (params.personStable && params.face && params.face.estimatedAge > 0) {
    state.faceCount += 1;
    state.ageSamples.push(params.face.estimatedAge);
    if (state.ageSamples.length > 12) state.ageSamples.shift();
    const g = normGender(params.face.estimatedGender);
    if (g === "male") state.maleScore += 1;
    else if (g === "female") state.femaleScore += 1;
  }

  if (params.emotion?.dominant) {
    state.lastEmotion = params.emotion.dominant;
    state.lastEmotionScore = params.emotion.dominantScore ?? 0;
  }

  const rawAverage =
    state.ageSamples.length > 0
      ? state.ageSamples.reduce((a, b) => a + b, 0) / state.ageSamples.length
      : null;
  const ageEstimate = rawAverage != null ? correctAgeForDisplay(rawAverage) : null;
  const ageRawEstimate = rawAverage != null ? Math.round(rawAverage) : null;

  let gender: GenderEstimate = "unknown";
  if (state.maleScore > state.femaleScore + 1) gender = "male";
  else if (state.femaleScore > state.maleScore + 1) gender = "female";
  else if (params.face) gender = normGender(params.face.estimatedGender);

  const confidence = Math.min(
    1,
    (state.faceCount >= 2 ? 0.45 : 0.15) +
      (state.ageSamples.length >= 3 ? 0.35 : state.ageSamples.length * 0.1) +
      (gender !== "unknown" ? 0.2 : 0),
  );

  const gazeAtCamera =
    params.face?.gazeDirection === "Center" || params.human.attention === "camera";

  return {
    ageEstimate,
    ageRawEstimate,
    gender,
    segment: segmentFromAge(ageEstimate),
    emotion: state.lastEmotion,
    emotionScore: state.lastEmotionScore,
    gazeAtCamera,
    engagement: inferEngagementMode(params.body, params.human, params.emotion),
    confidence,
    faceObservations: state.faceCount,
    updatedAt: now,
  };
};

export const formatEntityForGemma = (entity: EntityProfile | null): string => {
  if (!entity || entity.faceObservations < 1) {
    return "ENTITY PROFILE: not established yet — ask gently who is present; do not assume age or gender.";
  }
  const genderHe =
    entity.gender === "male" ? "גבר (הערכה)" : entity.gender === "female" ? "אישה (הערכה)" : "לא ברור";
  const segmentHe =
    entity.segment === "child"
      ? "ילד/ה"
      : entity.segment === "teen"
        ? "נער/ה"
        : entity.segment === "adult"
          ? "מבוגר/ת"
          : "לא ידוע";
  return [
    "ENTITY PROFILE (face model — tentative, confirm by conversation):",
    `- Age estimate (display): ${entity.ageEstimate ?? "?"} (${segmentHe})`,
    entity.ageRawEstimate != null && entity.ageRawEstimate > ADULT_AGE_BIAS_THRESHOLD
      ? `- Face model raw age was ~${entity.ageRawEstimate}; corrected −20% for 40+ bias`
      : "",
    `- Gender estimate: ${genderHe}`,
    `- Emotion: ${entity.emotion ?? "unknown"} (${Math.round(entity.emotionScore * 100)}%)`,
    `- Engagement: ${entity.engagement} | gaze at camera: ${entity.gazeAtCamera ? "yes" : "no"}`,
    `- Confidence: ${Math.round(entity.confidence * 100)}% (${entity.faceObservations} face frames)`,
    "",
    "TASK: Use corrected display age in speech. Youth ages are trusted as-is. Always confirm name by asking.",
  ]
    .filter(Boolean)
    .join("\n");
};
