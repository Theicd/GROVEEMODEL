/**
 * Configurable situation detection rules — enable/disable, cooldowns, utterances, LLM tier.
 */

export type SituationTier = "instant" | "llm_boot" | "llm_change";
export type SituationSource =
  | "static_gesture"
  | "motion_gesture"
  | "pose_action"
  | "body_language"
  | "interaction"
  | "event"
  | "object";

export type SituationRule = {
  id: string;
  label: string;
  labelHe: string;
  enabled: boolean;
  source: SituationSource;
  /** Match against normalized signal name (gesture, event, cue signal, object label). */
  match: string;
  tier: SituationTier;
  cooldownMs: number;
  subject: string;
  utteranceHe: string;
  llmHint?: string;
  proactive: boolean;
};

export const DEFAULT_SITUATION_RULES: SituationRule[] = [
  {
    id: "wave",
    label: "Waving",
    labelHe: "ניפנוף",
    enabled: true,
    source: "motion_gesture",
    match: "waving",
    tier: "instant",
    cooldownMs: 12_000,
    subject: "wave",
    utteranceHe: "אני די בטוח שניסית למשוך את תשומת הלב שלי עכשיו.",
    proactive: true,
  },
  {
    id: "thumbs_up",
    label: "Thumbs Up",
    labelHe: "אגודל למעלה",
    enabled: true,
    source: "static_gesture",
    match: "thumbs up",
    tier: "instant",
    cooldownMs: 15_000,
    subject: "gesture:thumbs_up",
    utteranceHe: "ראיתי אגודל למעלה — כל הכבוד.",
    proactive: true,
  },
  {
    id: "clapping",
    label: "Clapping",
    labelHe: "מחיאות כפיים",
    enabled: true,
    source: "motion_gesture",
    match: "clapping",
    tier: "instant",
    cooldownMs: 20_000,
    subject: "arm_movement",
    utteranceHe: "שמעתי מחיאות כפיים — מחווה יפה.",
    proactive: true,
  },
  {
    id: "hands_on_head",
    label: "Hands on Head",
    labelHe: "ידיים על הראש",
    enabled: true,
    source: "body_language",
    match: "hands on head",
    tier: "instant",
    cooldownMs: 25_000,
    subject: "hands_on_head",
    utteranceHe: "נראה שאתה מחזיק את הראש — הכל בסדר?",
    proactive: true,
  },
  {
    id: "hand_on_face",
    label: "Hand on Face",
    labelHe: "יד על הפנים",
    enabled: true,
    source: "body_language",
    match: "hand on face",
    tier: "instant",
    cooldownMs: 30_000,
    subject: "hand_on_face",
    utteranceHe: "שמתי לב שאתה נוגע בפנים — חושב על משהו?",
    proactive: false,
  },
  {
    id: "holding_cup",
    label: "Holding Cup",
    labelHe: "מחזיק כוס",
    enabled: true,
    source: "interaction",
    match: "holding cup",
    tier: "instant",
    cooldownMs: 60_000,
    subject: "stood_with_drink",
    utteranceHe: "נראה שיש כוס ביד — רגע של קפה?",
    proactive: true,
  },
  {
    id: "phone_usage",
    label: "Using Phone",
    labelHe: "שימוש בטלפון",
    enabled: true,
    source: "event",
    match: "phone usage",
    tier: "instant",
    cooldownMs: 90_000,
    subject: "focused_work",
    utteranceHe: "נראה שאתה עם הטלפון — משהו דחוף?",
    proactive: false,
  },
  {
    id: "person_entered",
    label: "Person Entered",
    labelHe: "אדם נכנס לפריים",
    enabled: true,
    source: "event",
    match: "person entered",
    tier: "instant",
    cooldownMs: 25_000,
    subject: "person",
    utteranceHe: "שמתי לב שאתה בפריים — מה קורה?",
    proactive: true,
  },
  {
    id: "guitar",
    label: "Guitar in Scene",
    labelHe: "גיטרה בסצנה",
    enabled: true,
    source: "object",
    match: "guitar",
    tier: "llm_boot",
    cooldownMs: 900_000,
    subject: "object:guitar",
    utteranceHe: "שמתי לב שיש גיטרה בחדר — אתה מנגן?",
    llmHint: "User may play guitar — ask warmly about music.",
    proactive: true,
  },
  {
    id: "jumping",
    label: "Jumping",
    labelHe: "קפיצה",
    enabled: true,
    source: "pose_action",
    match: "jumping",
    tier: "instant",
    cooldownMs: 20_000,
    subject: "motion_burst",
    utteranceHe: "אני רואה הרבה תנועה — בודק שאני עדיין איתך?",
    proactive: true,
  },
];

const STORAGE_KEY = "grovee-situation-registry-v1";

export const loadSituationRegistry = (): SituationRule[] => {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_SITUATION_RULES.map((r) => ({ ...r }));
    const parsed = JSON.parse(raw) as SituationRule[];
    if (!Array.isArray(parsed) || parsed.length === 0) {
      return DEFAULT_SITUATION_RULES.map((r) => ({ ...r }));
    }
    const byId = new Map(DEFAULT_SITUATION_RULES.map((r) => [r.id, r]));
    for (const rule of parsed) {
      if (rule?.id && byId.has(rule.id)) {
        byId.set(rule.id, { ...byId.get(rule.id)!, ...rule, id: rule.id });
      }
    }
    return [...byId.values()];
  } catch {
    return DEFAULT_SITUATION_RULES.map((r) => ({ ...r }));
  }
};

export const saveSituationRegistry = (rules: SituationRule[]): void => {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(rules));
  } catch {
    // ignore quota
  }
};

export const resetSituationRegistry = (): SituationRule[] => {
  const rules = DEFAULT_SITUATION_RULES.map((r) => ({ ...r }));
  saveSituationRegistry(rules);
  return rules;
};

export const findSituationRule = (rules: SituationRule[], subject: string): SituationRule | undefined =>
  rules.find((r) => r.enabled && r.subject === subject);

export const findSituationUtterance = (rules: SituationRule[], subject: string): string | null => {
  const rule = findSituationRule(rules, subject);
  return rule?.utteranceHe?.trim() || null;
};
