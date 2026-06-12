/** Compose multi-pack matches into a single scene label. */

import type { BuiltScene, MatchedSituation, SituationTone } from "./types";

const SCENE_COMPOSITIONS: Array<{
  tags: string[];
  label: string;
  interpretation: string;
}> = [
  {
    tags: ["psych", "stress", "thinking"],
    label: "עומס קוגניטיבי פנימי",
    interpretation: "Internal cognitive-emotional load without single clear trigger",
  },
  {
    tags: ["psych", "focus"],
    label: "מצב flow / עומק",
    interpretation: "Deep autonomous engagement with reduced social need",
  },
  {
    tags: ["psych", "social", "stress"],
    label: "מתח חברתי-קוגניטיבי",
    interpretation: "Social evaluation pressure combined with task demand",
  },
  {
    tags: ["social", "rest"],
    label: "רגע אינטראקציה רגוע",
    interpretation: "User is socially present while in a break or rest context",
  },
  {
    tags: ["social", "focus"],
    label: "פנייה במהלך עבודה",
    interpretation: "User seeks attention while remaining task-oriented",
  },
  {
    tags: ["social", "direct"],
    label: "פנייה ישירה למערכת",
    interpretation: "User explicitly addresses the system",
  },
  {
    tags: ["stress", "focus"],
    label: "עומס במהלך משימה",
    interpretation: "Cognitive load rising during active work",
  },
  {
    tags: ["thinking", "focus"],
    label: "עיבוד מידע עמוק",
    interpretation: "User is processing while engaged with task",
  },
  {
    tags: ["social", "positive"],
    label: "אינטראקציה חיובית",
    interpretation: "Warm expressive social exchange",
  },
  {
    tags: ["environment", "curious"],
    label: "שינוי בסביבה",
    interpretation: "Environmental shift draws attention",
  },
];

export const buildScene = (matches: MatchedSituation[]): BuiltScene | null => {
  if (!matches.length) return null;

  const packIds = matches.slice(0, 4).map((m) => m.pack.id);
  const tags = new Set(matches.flatMap((m) => m.pack.sceneTags ?? []));
  const interpretations = matches.map((m) => m.pack.interpretation).filter(Boolean);

  for (const rule of SCENE_COMPOSITIONS) {
    if (rule.tags.every((t) => tags.has(t))) {
      return {
        label: rule.label,
        interpretation: rule.interpretation,
        packIds,
        dominantTone: dominantToneFromMatches(matches),
      };
    }
  }

  if (matches.length >= 2) {
    return {
      label: "רגע מורכב",
      interpretation: interpretations.slice(0, 2).join(" · "),
      packIds,
      dominantTone: dominantToneFromMatches(matches),
    };
  }

  const solo = matches[0].pack;
  return {
    label: solo.nameHe ?? solo.name,
    interpretation: solo.interpretation,
    packIds: [solo.id],
    dominantTone: solo.tone,
  };
};

const dominantToneFromMatches = (matches: MatchedSituation[]): SituationTone => {
  const tone = matches[0]?.pack.tone ?? "neutral";
  return tone;
};
