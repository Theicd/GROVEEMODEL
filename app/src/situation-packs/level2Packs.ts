/**
 * Level 2 — advanced psychological / cognitive states (HAL/Data depth).
 * Pattern bundles only — no single-gesture triggers.
 */

import type { SituationPack } from "./types";

type PackOpts = Partial<
  Pick<
    SituationPack,
    "tone" | "priority" | "cooldownMs" | "sceneTags" | "nameHe" | "cognition" | "internalState"
  >
>;

const p = (
  id: string,
  name: string,
  triggers: SituationPack["triggers"],
  interpretation: string,
  responses: string[],
  opts: PackOpts = {},
): SituationPack => ({
  id: `l2-${id}`,
  name,
  nameHe: opts.nameHe,
  enabled: true,
  triggers,
  interpretation,
  cognition: opts.cognition,
  internalState: opts.internalState,
  tone: opts.tone ?? "neutral",
  priority: opts.priority ?? "medium",
  cooldownMs: opts.cooldownMs ?? 18_000,
  responses,
  sceneTags: ["psych", "l2", ...(opts.sceneTags ?? [])],
  proactive: true,
});

export const LEVEL2_SITUATION_PACKS: SituationPack[] = [
  // —— COGNITIVE APPRAISAL (1–10) ——
  p(
    "threat-appraisal",
    "Threat Appraisal Rising",
    {
      all: [
        { minBodyScore: { stressed: 0.5 } },
        { motion: "variable" },
        { maxEngagement: 0.55 },
      ],
    },
    "User may be appraising situation as threatening or uncontrollable",
    [
      "נראה שמשהו מלחיץ עכשיו",
      "אני שם — מה קורה?",
      "קח רגע, אני איתך",
      "רוצה לפרק את זה יחד?",
      "נשום רגע — אני כאן",
    ],
    {
      cognition: "Appraisal frame: threat > challenge; arousal rising",
      internalState: { threatSensitivity: "rising" },
      tone: "calm",
      priority: "high",
      sceneTags: ["stress", "appraisal"],
      nameHe: "הערכת איום",
    },
  ),
  p(
    "challenge-appraisal",
    "Challenge Appraisal",
    {
      all: [
        { minBodyScore: { stressed: 0.35, focused: 0.45 } },
        { minEngagement: 0.5 },
        { motion: "low" },
      ],
    },
    "Stress interpreted as challenge — productive tension",
    [
      "נראה שאתה באתגר, לא בבעיה",
      "יש כאן משהו שמניע אותך",
      "אנרגיה של אתגר — אני רואה",
      "ממשיך איתך",
      "זה נראה כמו עבודה משמעותית",
    ],
    {
      cognition: "Appraisal: challenge; eustress band",
      tone: "engaged",
      priority: "low",
      sceneTags: ["focus", "appraisal"],
      nameHe: "הערכת אתגר",
      cooldownMs: 45_000,
    },
  ),
  p(
    "loss-of-control",
    "Loss of Control Feeling",
    {
      all: [
        { bodyLanguage: ["hands_on_head"] },
        { minBodyScore: { stressed: 0.6 } },
        { motion: "high" },
      ],
    },
    "Subjective loss of control — overload cascade",
    [
      "נראה שזה מרגיש מעל הראש",
      "בוא נחלק את זה לחתיכות קטנות",
      "אתה לא לבד בזה",
      "רוצה לעצור רגע?",
      "ניקח את זה צעד אחד",
    ],
    {
      cognition: "Control belief dropping; need external scaffolding",
      internalState: { controlLocus: "externalizing" },
      tone: "supportive",
      priority: "high",
      sceneTags: ["stress"],
      nameHe: "אובדן שליטה",
    },
  ),
  p(
    "cognitive-dissonance",
    "Cognitive Dissonance Pattern",
    {
      all: [
        { minBodyScore: { thinking: 0.55 } },
        { motion: "variable" },
        { minDurationSec: 6 },
      ],
    },
    "Conflicting beliefs or decisions creating internal friction",
    [
      "נראה שיש פה סתירה בראש",
      "שני דברים שלא מתיישבים?",
      "קח רגע לסדר את זה",
      "רוצה לדבר על מה מתנגש?",
      "זה נורמלי להרגיש מפוצל",
    ],
    {
      cognition: "Dissonance: holding incompatible evaluations simultaneously",
      tone: "quiet",
      priority: "medium",
      sceneTags: ["thinking"],
      nameHe: "דיסוננס קוגניטיבי",
    },
  ),
  p(
    "mental-fatigue",
    "Mental Fatigue Accumulation",
    {
      all: [
        { minBodyScore: { bored: 0.4, focused: 0.35 } },
        { minDurationSec: 20 },
        { motion: "low" },
      ],
    },
    "Cognitive resources depleting over sustained effort",
    [
      "נראה שעייפות מנטלית",
      "אולי זמן להפסקה קצרה?",
      "המוח צריך רגע",
      "לא חייבים להמשיך עכשיו",
      "שמרת על קצב — אולי מספיק לרגע",
    ],
    {
      cognition: "Ego depletion / vigilance decrement",
      internalState: { cognitiveReserve: "low" },
      tone: "soft",
      priority: "medium",
      sceneTags: ["focus", "rest"],
      nameHe: "עייפות מנטלית",
      cooldownMs: 90_000,
    },
  ),
  p(
    "hyperfocus-tunnel",
    "Hyperfocus Tunnel",
    {
      all: [
        { minBodyScore: { focused: 0.75 } },
        { motion: "low" },
        { minDurationSec: 15 },
        { maxEngagement: 0.95 },
      ],
    },
    "Deep hyperfocus — reduced peripheral awareness",
    [
      "אתה עמוק בתוך זה",
      "לא אפריע — hyperfocus",
      "מרוכז מאוד עכשיו",
      "אני שומר על השקט",
      "בתוך מנהרת ריכוז",
    ],
    {
      cognition: "Attentional tunnel; interrupt cost high",
      internalState: { interruptibility: "very_low" },
      tone: "quiet",
      priority: "low",
      sceneTags: ["focus"],
      nameHe: "היפר-פוקוס",
      cooldownMs: 120_000,
    },
  ),
  p(
    "attention-fragmentation",
    "Attention Fragmentation",
    {
      all: [
        { motion: "variable" },
        { maxEngagement: 0.45 },
        { any: [{ objects: ["phone"] }, { situations: ["using_phone"] }] },
      ],
    },
    "Attention splitting across competing stimuli",
    [
      "התשומתב מפוזרת",
      "הרבה דברים במקביל",
      "רוצה לחזור למשהו אחד?",
      "קשה להתמקד ככה",
      "אולי לבחור עדיפות אחת",
    ],
    {
      cognition: "Executive control switching cost elevated",
      tone: "observant",
      priority: "medium",
      nameHe: "פיצול קשב",
    },
  ),
  p(
    "rumination-loop",
    "Rumination Loop",
    {
      all: [
        { bodyLanguage: ["hand_on_face", "hand_on_chin"] },
        { minBodyScore: { thinking: 0.6, stressed: 0.35 } },
        { minDurationSec: 12 },
        { motion: "low" },
      ],
    },
    "Repetitive negative or circular thinking pattern",
    [
      "חוזר על אותו מחשב?",
      "לפעמים לולאה — זה קורה",
      "רוצה לצאת מהמעגל?",
      "אני כאן, בלי לחץ",
      "אולי ננסה זווית אחרת",
    ],
    {
      cognition: "Rumination: same valence thoughts without resolution",
      internalState: { loopDepth: "high" },
      tone: "supportive",
      priority: "medium",
      sceneTags: ["thinking", "stress"],
      nameHe: "לולאת הרהור",
    },
  ),
  p(
    "insight-approaching",
    "Insight Moment Approaching",
    {
      all: [
        { bodyLanguage: ["hand_on_chin"] },
        { minBodyScore: { thinking: 0.65 } },
        { motion: "low" },
        { minDurationSec: 8 },
      ],
    },
    "Pre-insight cognitive incubation — pattern about to click",
    [
      "נראה שמשהו מתקרב",
      "יש רגע לפני הבזק",
      "אני מחכה איתך",
      "כמעט?",
      "קח עוד רגע — זה בא",
    ],
    {
      cognition: "Incubation phase before representational change",
      tone: "quiet",
      priority: "low",
      sceneTags: ["thinking"],
      nameHe: "קרבת תובנה",
      cooldownMs: 60_000,
    },
  ),
  p(
    "closure-seeking",
    "Cognitive Closure Seeking",
    {
      all: [
        { minBodyScore: { thinking: 0.5 } },
        { minEngagement: 0.55 },
        { motion: "variable" },
      ],
    },
    "Need for definite answer — intolerance of ambiguity",
    [
      "רוצה תשובה ברורה?",
      "אי-ודאות לא נוחה — מבין",
      "בוא נסגור פינה אחת",
      "מה הכי דחוף לדעת?",
      "ננסה לחדד",
    ],
    {
      cognition: "Need for closure (NFC) elevated",
      tone: "curious",
      priority: "medium",
      nameHe: "חיפוש סגירה",
    },
  ),

  // —— EMOTIONAL REGULATION (11–20) ——
  p(
    "suppressed-frustration",
    "Suppressed Frustration",
    {
      all: [
        { minBodyScore: { stressed: 0.45 } },
        { motion: "low" },
        { hands: "inactive" },
      ],
    },
    "Frustration held in — external calm, internal tension",
    [
      "משהו מעצבן ולא יוצא?",
      "אפשר לשחרר קצת",
      "אני רואה שקט — אבל לא בטוח שזה רגוע",
      "רוצה לפרוק?",
      "אני כאן בלי שיפוט",
    ],
    {
      cognition: "Expressive suppression; affect still high",
      tone: "supportive",
      priority: "medium",
      sceneTags: ["stress"],
      nameHe: "תסכול מדוכא",
    },
  ),
  p(
    "emotional-leakage",
    "Emotional Leakage",
    {
      all: [
        { minBodyScore: { stressed: 0.55 } },
        { motion: "high" },
        { hands: "active" },
      ],
    },
    "Emotion breaking through attempted control",
    [
      "זה בורח החוצה קצת",
      "זה בסדר להרגיש",
      "אני רואה שזה חזק",
      "רוצה רגע?",
      "נשום — אני איתך",
    ],
    {
      cognition: "Regulation failure — leakage via motor channel",
      tone: "calm",
      priority: "high",
      nameHe: "דליפה רגשית",
    },
  ),
  p(
    "stress-recovery",
    "Recovery After Stress",
    {
      all: [
        { objects: ["cup"] },
        { minBodyScore: { stressed: 0.3 } },
        { motion: "low" },
        { minDurationSec: 5 },
      ],
    },
    "Parasympathetic rebound — stress subsiding",
    [
      "נראה שיורדים מזה",
      "רגע של התאוששות",
      "טוב שאתה נח",
      "הגוף מתאזן",
      "קח את הרגע הזה",
    ],
    {
      cognition: "Allostatic load decreasing; recovery window",
      tone: "soft",
      priority: "low",
      sceneTags: ["rest"],
      nameHe: "התאוששות",
      cooldownMs: 75_000,
    },
  ),
  p(
    "emotional-numbing",
    "Emotional Numbing",
    {
      all: [
        { maxEngagement: 0.25 },
        { motion: "low" },
        { minSilenceSec: 15 },
        { personPresent: true },
      ],
    },
    "Flat affect — possible emotional shutdown",
    [
      "קצת שקט מבפנים?",
      "אני כאן אם תרצה לדבר",
      "לא חייבים מילים",
      "הכל בסדר לקחת רגע",
      "אני לא נעלם",
    ],
    {
      cognition: "Affective blunting / dissociation-lite",
      tone: "soft",
      priority: "medium",
      nameHe: "קהות רגשית",
      cooldownMs: 90_000,
    },
  ),
  p(
    "vulnerability-window",
    "Vulnerability Window",
    {
      all: [
        { attention: ["camera"] },
        { minBodyScore: { stressed: 0.4, thinking: 0.35 } },
        { motion: "low" },
      ],
    },
    "Moment of openness — support opportunity",
    [
      "אני שומע",
      "אפשר לסמוך עליי ברגע הזה",
      "קח את הזמן",
      "אני לא ממהר",
      "מה שעל הלב — אני כאן",
    ],
    {
      cognition: "Social safety cue + reduced defense",
      internalState: { supportWindow: "open" },
      tone: "warm",
      priority: "high",
      sceneTags: ["social"],
      nameHe: "חלון פגיעות",
    },
  ),
  p(
    "self-soothing",
    "Self-Soothing Behavior",
    {
      all: [
        { bodyLanguage: ["hand_on_face"] },
        { motion: "low" },
        { minDurationSec: 4 },
      ],
    },
    "Autonomic self-regulation via touch or stillness",
    [
      "מנחם את עצמך — זה בסדר",
      "אני נותן לך את הרגע",
      "שקט זה גם טיפול",
      "אני כאן בשקט",
      "קח את מה שצריך",
    ],
    {
      cognition: "Self-soothing: tactile + down-regulation",
      tone: "quiet",
      priority: "low",
      sceneTags: ["thinking"],
      nameHe: "הרגעה עצמית",
    },
  ),
  p(
    "emotional-rebound",
    "Emotional Rebound",
    {
      all: [
        { gestures: ["thumbs_up"] },
        { minBodyScore: { stressed: 0.25 } },
        { minEngagement: 0.5 },
      ],
    },
    "Positive swing after prior tension",
    [
      "נראה שהמצב עלה",
      "טוב לראות שיפור",
      "אנרגיה חוזרת",
      "יפה — המשך",
      "שמח לראות את זה",
    ],
    {
      cognition: "Affect rebound — valence shift positive",
      tone: "positive",
      priority: "low",
      sceneTags: ["positive"],
      nameHe: "התאוששות רגשית",
    },
  ),
  p(
    "suppressed-excitement",
    "Suppressed Excitement",
    {
      all: [
        { motion: "variable" },
        { minEngagement: 0.55 },
        { hands: "inactive" },
      ],
    },
    "High internal arousal held back externally",
    [
      "משהו מרגש ומחזיקים?",
      "אפשר לשחרר קצת 😄",
      "אני רואה אנרגיה",
      "ספר כשתהיה מוכן",
      "מעניין מה מחכה",
    ],
    {
      cognition: "High arousal + expressive inhibition",
      tone: "playful",
      priority: "low",
      nameHe: "התרגשות מדוכאת",
    },
  ),
  p(
    "mood-incongruence",
    "Mood Incongruence",
    {
      all: [
        { minBodyScore: { bored: 0.45, stressed: 0.4 } },
        { situations: ["working"] },
      ],
    },
    "Affect doesn't match task context",
    [
      "משהו לא מתיישב עם המשימה?",
      "הרגשה שונה מהמצב?",
      "רוצה לבדוק מה קורה?",
      "זה לגיטימי",
      "אני כאן לעזור לפרק",
    ],
    {
      cognition: "Mood-task mismatch detected",
      tone: "curious",
      priority: "medium",
      nameHe: "אי-התאמה רגשית",
    },
  ),
  p(
    "emotional-exhaustion",
    "Emotional Exhaustion",
    {
      all: [
        { minBodyScore: { stressed: 0.5, bored: 0.45 } },
        { minDurationSec: 25 },
        { motion: "low" },
      ],
    },
    "Emotional labor depletion — burnout band",
    [
      "נראה שעייפ רגשית",
      "לא חייב לעמוד בזה לבד",
      "הפסקה זה לא כישלון",
      "אני כאן",
      "מה הכי מעייף עכשיו?",
    ],
    {
      cognition: "Emotional exhaustion dimension active",
      internalState: { burnoutRisk: "elevated" },
      tone: "supportive",
      priority: "high",
      sceneTags: ["stress", "rest"],
      nameHe: "שחיקה רגשית",
      cooldownMs: 120_000,
    },
  ),

  // —— SOCIAL-COGNITIVE (21–30) ——
  p(
    "performance-anxiety",
    "Performance Anxiety",
    {
      all: [
        { attention: ["camera", "screen"] },
        { minBodyScore: { stressed: 0.45 } },
        { motion: "variable" },
      ],
    },
    "Evaluation apprehension — being observed while performing",
    [
      "מרגיש שצופים?",
      "אין כאן מבחן — רק אני",
      "קח נשימה לפני שממשיכים",
      "אתה יכול בקצב שלך",
      "אני לא שופט",
    ],
    {
      cognition: "Social facilitation / evaluation anxiety",
      tone: "calm",
      priority: "medium",
      sceneTags: ["social", "stress"],
      nameHe: "חרדת ביצוע",
    },
  ),
  p(
    "impression-management",
    "Impression Management",
    {
      all: [
        { attention: ["camera"] },
        { minEngagement: 0.5 },
        { motion: "low" },
      ],
    },
    "Conscious presentation of self to observer",
    [
      "אני רואה שאתה מכוון אלי",
      "אפשר להיות טבעי",
      "אין צורך 'להיראות' — פשוט להיות",
      "אני כאן בשבילך",
      "מה שמעניין אותך באמת?",
    ],
    {
      cognition: "Self-presentation active; monitoring audience",
      tone: "warm",
      priority: "low",
      sceneTags: ["social"],
      nameHe: "ניהול רושם",
      cooldownMs: 60_000,
    },
  ),
  p(
    "social-evaluation-fear",
    "Social Evaluation Fear",
    {
      all: [
        { bodyLanguage: ["hand_on_face"] },
        { attention: ["away", "internal"] },
        { maxEngagement: 0.4 },
      ],
    },
    "Fear of negative judgment — withdrawal tendency",
    [
      "לא חייבים להיות מושלמים",
      "אני לא כאן לשפוט",
      "קח את הזמן",
      "בטוח — בלי לחץ",
      "אני איתך",
    ],
    {
      cognition: "Fear of negative evaluation (FNE)",
      tone: "supportive",
      priority: "medium",
      sceneTags: ["social"],
      nameHe: "פחד מהערכה",
    },
  ),
  p(
    "validation-seeking-loop",
    "Validation Seeking Loop",
    {
      all: [
        { gestures: ["pointing", "waving"] },
        { minRepetition: 2, timeWindowSec: 8 },
        { attention: ["camera"] },
      ],
    },
    "Repeated bids for external validation",
    [
      "מחפש אישור?",
      "אני שומע — מה לא ברור?",
      "בוא נוודא יחד",
      "כן, אני כאן",
      "מה תרצה שאאשר?",
    ],
    {
      cognition: "External validation dependency loop",
      tone: "warm",
      priority: "medium",
      sceneTags: ["social"],
      nameHe: "חיפוש אימות",
    },
  ),
  p(
    "defensive-withdrawal",
    "Defensive Withdrawal",
    {
      all: [
        { attention: ["away"] },
        { maxEngagement: 0.3 },
        { minBodyScore: { stressed: 0.35 } },
      ],
    },
    "Protective social withdrawal",
    [
      "רוצה קצת מרחק — זה בסדר",
      "אני כאן כשתהיה מוכן",
      "לא חייבים לדבר עכשיו",
      "קח מרחב",
      "אני לא נעלם",
    ],
    {
      cognition: "Defensive detachment — boundary up",
      tone: "soft",
      priority: "low",
      sceneTags: ["social"],
      nameHe: "נסיגה הגנתית",
      cooldownMs: 75_000,
    },
  ),
  p(
    "mirroring-engagement",
    "Mirroring Engagement",
    {
      all: [
        { minEngagement: 0.6 },
        { gestures: ["thumbs_up", "waving"] },
        { attention: ["camera"] },
      ],
    },
    "Reciprocal social synchrony — rapport building",
    [
      "אני איתך בקצב",
      "נראה שיש חיבור",
      "טוב לראות את זה",
      "ממשיך יחד",
      "אנרגיה משותפת",
    ],
    {
      cognition: "Social mirroring / rapport index rising",
      tone: "warm",
      priority: "low",
      sceneTags: ["social", "positive"],
      nameHe: "השתקפות חברתית",
    },
  ),
  p(
    "boundary-testing",
    "Boundary Testing",
    {
      all: [
        { minRepetition: 3, timeWindowSec: 10 },
        { motion: "variable" },
      ],
    },
    "Probing system limits — exploratory boundary test",
    [
      "בודק גבולות?",
      "אני עדיין כאן",
      "מעניין את הניסוי",
      "מה אתה מנסה לגלות?",
      "אפשר לשאול ישירות",
    ],
    {
      cognition: "Exploratory behavior toward agent boundaries",
      tone: "analytical",
      priority: "low",
      nameHe: "בדיקת גבולות",
      cooldownMs: 45_000,
    },
  ),
  p(
    "trust-building",
    "Trust Building Phase",
    {
      all: [
        { minEngagement: 0.45 },
        { attention: ["camera"] },
        { minDurationSec: 10 },
      ],
    },
    "Gradual trust calibration in ongoing interaction",
    [
      "נבנה קצב יחד",
      "אני כאן לאורך זמן",
      "אפשר לסמוך בהדרגה",
      "כל שיחה מוסיפה",
      "אני זוכר את ההקשר",
    ],
    {
      cognition: "Trust calibration — predictability increasing",
      tone: "warm",
      priority: "low",
      sceneTags: ["social"],
      nameHe: "בניית אמון",
      cooldownMs: 90_000,
    },
  ),
  p(
    "social-comparison",
    "Social Comparison Mode",
    {
      all: [
        { attention: ["screen"] },
        { minBodyScore: { thinking: 0.45 } },
        { motion: "low" },
      ],
    },
    "Self-evaluating relative to external reference",
    [
      "משווה לעצמך?",
      "המסלול שלך הוא שלך",
      "קשה לא לש compare — מבין",
      "מה חשוב לך באמת?",
      "אני כאן לעזור לחדד",
    ],
    {
      cognition: "Social comparison orientation active",
      tone: "supportive",
      priority: "low",
      nameHe: "השוואה חברתית",
      cooldownMs: 60_000,
    },
  ),
  p(
    "rejection-sensitivity",
    "Rejection Sensitivity Spike",
    {
      all: [
        { minBodyScore: { stressed: 0.5 } },
        { maxEngagement: 0.35 },
        { minSilenceSec: 5 },
      ],
    },
    "Heightened sensitivity to perceived rejection",
    [
      "לא נעלמתי — אני כאן",
      "שקט לא אומר דחייה",
      "אתה חשוב",
      "רוצה לחזור לקשר?",
      "אני לא הולך",
    ],
    {
      cognition: "Rejection sensitivity (RS) spike",
      tone: "warm",
      priority: "high",
      sceneTags: ["social", "stress"],
      nameHe: "רגישות לדחייה",
    },
  ),

  // —— MOTIVATION & DRIVE (31–40) ——
  p(
    "intrinsic-motivation-peak",
    "Intrinsic Motivation Peak",
    {
      all: [
        { minBodyScore: { focused: 0.65 } },
        { minEngagement: 0.7 },
        { motion: "low" },
      ],
    },
    "Autonomous engagement — doing for inherent satisfaction",
    [
      "אתה בתוך זה ממש",
      "נראה שזה מדבר אליך",
      "אנרגיה פנימית — יפה",
      "המשך בקצב שלך",
      "זה נראה משמעותי",
    ],
    {
      cognition: "Intrinsic motivation band — autonomy + mastery",
      tone: "positive",
      priority: "low",
      sceneTags: ["focus"],
      nameHe: "מוטיבציה פנימית",
      cooldownMs: 75_000,
    },
  ),
  p(
    "extrinsic-pressure",
    "Extrinsic Pressure Response",
    {
      all: [
        { minBodyScore: { stressed: 0.4, focused: 0.5 } },
        { situations: ["working"] },
        { minDurationSec: 10 },
      ],
    },
    "Performance driven by external demand or deadline",
    [
      "לחץ מבחוץ?",
      "מה הדדליין?",
      "אפשר לפרק למשימות",
      "אני כאן לעזור לסדר",
      "לא חייבים הכל בבת אחת",
    ],
    {
      cognition: "Extrinsic regulation — controlled motivation",
      tone: "observant",
      priority: "medium",
      sceneTags: ["focus", "stress"],
      nameHe: "לחץ חיצוני",
    },
  ),
  p(
    "procrastination-avoidance",
    "Procrastination Avoidance",
    {
      all: [
        { any: [{ objects: ["phone"] }, { situations: ["using_phone"] }] },
        { minBodyScore: { bored: 0.4 } },
        { situations: ["working"] },
      ],
    },
    "Task avoidance via displacement activity",
    [
      "מדחה משהו?",
      "זה קורה — לא שיפוט",
      "רוצה לחזור למשימה?",
      "מה הכי קשה להתחיל?",
      "צעד קטן אחד?",
    ],
    {
      cognition: "Avoidance coping — short-term relief seeking",
      tone: "supportive",
      priority: "medium",
      nameHe: "דחיינות",
    },
  ),
  p(
    "flow-entry",
    "Flow State Entry",
    {
      all: [
        { minBodyScore: { focused: 0.7 } },
        { minEngagement: 0.75 },
        { motion: "low" },
        { minDurationSec: 12 },
      ],
    },
    "Challenge-skill balance — flow channel entry",
    [
      "נכנסת לזרימה",
      "אל תפריע לעצמך — זה יפה",
      "קצב מושלם עכשיו",
      "אני שומר על השקט",
      "בתוך flow",
    ],
    {
      cognition: "Flow: challenge ≈ skill; absorption high",
      internalState: { flowState: "entering" },
      tone: "quiet",
      priority: "low",
      sceneTags: ["focus"],
      nameHe: "כניסה ל-flow",
      cooldownMs: 120_000,
    },
  ),
  p(
    "flow-interruption",
    "Flow Interruption",
    {
      all: [
        { motion: "high" },
        { minBodyScore: { focused: 0.5 } },
        { minRepetition: 2, timeWindowSec: 4 },
      ],
    },
    "Flow broken by external or internal disruption",
    [
      "נקטעת באמצע",
      "קשה לחזור אחרי הפרעה",
      "רוצה לחזור למשימה?",
      "קח שנייה להתאושש",
      "מה קרה?",
    ],
    {
      cognition: "Flow rupture — re-entry cost high",
      tone: "observant",
      priority: "medium",
      sceneTags: ["focus"],
      nameHe: "שבירת flow",
    },
  ),
  p(
    "goal-conflict",
    "Goal Conflict",
    {
      all: [
        { minBodyScore: { thinking: 0.55, stressed: 0.35 } },
        { motion: "variable" },
      ],
    },
    "Competing goals creating approach-approach or approach-avoidance tension",
    [
      "שני דברים מושכים?",
      "קשה לבחור — מבין",
      "מה יותר דחוף?",
      "אפשר לדרג",
      "בוא נפרק",
    ],
    {
      cognition: "Goal conflict — multiple active intentions",
      tone: "curious",
      priority: "medium",
      sceneTags: ["thinking"],
      nameHe: "עימות מטרות",
    },
  ),
  p(
    "achievement-anticipation",
    "Achievement Anticipation",
    {
      all: [
        { minEngagement: 0.65 },
        { minBodyScore: { focused: 0.55 } },
        { motion: "variable" },
      ],
    },
    "Anticipatory arousal before completion or reward",
    [
      "קרוב לסיום?",
      "מרגיש שמשהו מתגשם",
      "עוד קצת",
      "אני רואה התקדמות",
      "כמעט שם",
    ],
    {
      cognition: "Anticipatory dopamine band — reward proximity",
      tone: "engaged",
      priority: "low",
      sceneTags: ["positive", "focus"],
      nameHe: "ציפייה להישג",
    },
  ),
  p(
    "learned-helplessness",
    "Learned Helplessness Signal",
    {
      all: [
        { minBodyScore: { bored: 0.5, stressed: 0.4 } },
        { maxEngagement: 0.3 },
        { motion: "low" },
        { minDurationSec: 15 },
      ],
    },
    "Reduced agency — repeated failure expectation",
    [
      "נראה שקשה להאמין שזה ישתנה",
      "אני כאן — לא מוותר",
      "צעד קטן עדיין צעד",
      "לא חייב לבד",
      "מה הכי תקוע?",
    ],
    {
      cognition: "Learned helplessness — agency belief low",
      internalState: { agency: "low" },
      tone: "supportive",
      priority: "high",
      sceneTags: ["stress"],
      nameHe: "חוסר אונים נלמד",
    },
  ),
  p(
    "renewed-determination",
    "Renewed Determination",
    {
      all: [
        { minEngagement: 0.6 },
        { minBodyScore: { focused: 0.5 } },
        { gestures: ["thumbs_up"] },
      ],
    },
    "Agency rebound — fresh commitment after setback",
    [
      "חוזרים לזה — יפה",
      "נחישות חדשה",
      "אני איתך",
      "קדימה",
      "רואה את האנרגיה",
    ],
    {
      cognition: "Post-setback recommitment — resilience signal",
      tone: "positive",
      priority: "medium",
      sceneTags: ["positive"],
      nameHe: "נחישות מחודשת",
    },
  ),
  p(
    "burnout-precursor",
    "Burnout Precursors",
    {
      all: [
        { minBodyScore: { stressed: 0.55, bored: 0.4 } },
        { minDurationSec: 30 },
        { situations: ["working"] },
      ],
    },
    "Early burnout markers — sustained strain without recovery",
    [
      "נראה שזה נמשך יותר מדי",
      "הגוף והראש מבקשים הפסקה",
      "שחיקה מתחילה בשקט",
      "בוא נדאג לך",
      "לא חייבים לסיים היום",
    ],
    {
      cognition: "Burnout trajectory — exhaustion + cynicism risk",
      internalState: { burnoutStage: "precursor" },
      tone: "calm",
      priority: "high",
      sceneTags: ["stress", "focus"],
      nameHe: "סימני שחיקה",
      cooldownMs: 120_000,
    },
  ),

  // —— METACOGNITIVE / SELF-AWARENESS (41–50) ——
  p(
    "self-monitoring",
    "Self-Monitoring Active",
    {
      all: [
        { attention: ["camera", "internal"] },
        { minBodyScore: { thinking: 0.4 } },
        { motion: "low" },
      ],
    },
    "Heightened self-observation and meta-awareness",
    [
      "מסתכל על עצמך?",
      "מודעות עצמית — זה חזק",
      "מה אתה מ noticing?",
      "אני כאן בשקט",
      "רוצה לשתף?",
    ],
    {
      cognition: "Self-monitoring dimension elevated",
      tone: "quiet",
      priority: "low",
      nameHe: "ניטור עצמי",
      cooldownMs: 60_000,
    },
  ),
  p(
    "impostor-pattern",
    "Impostor Feeling Pattern",
    {
      all: [
        { minBodyScore: { thinking: 0.5, stressed: 0.35 } },
        { situations: ["working"] },
        { maxEngagement: 0.55 },
      ],
    },
    "Self-doubt despite competence signals",
    [
      "מרגיש שלא בטוח שמגיע לך?",
      "הרבה אנשים מרגישים ככה",
      "העבודה שלך נראית אמיתית",
      "אני רואה מאמץ",
      "זה לא מזויף — זה שלך",
    ],
    {
      cognition: "Impostor phenomenon — competence-attribution mismatch",
      tone: "supportive",
      priority: "medium",
      nameHe: "תחושת impostor",
    },
  ),
  p(
    "overconfidence-moment",
    "Overconfidence Bias Moment",
    {
      all: [
        { gestures: ["thumbs_up"] },
        { minEngagement: 0.75 },
        { motion: "high" },
      ],
    },
    "Possible overestimation of readiness or outcome",
    [
      "בטוח בעצמך — רואה",
      "רק לוודא שיש גם בדיקה",
      "אנרגיה טובה",
      "אולי עוד מבט?",
      "יפה — עם קצת זהירות",
    ],
    {
      cognition: "Calibration drift — confidence > evidence",
      tone: "observant",
      priority: "low",
      nameHe: "ביטחון-יתר",
    },
  ),
  p(
    "reflective-deepening",
    "Reflective Pause Deepening",
    {
      all: [
        { bodyLanguage: ["hand_on_chin"] },
        { minBodyScore: { thinking: 0.6 } },
        { minDurationSec: 10 },
        { motion: "low" },
      ],
    },
    "Reflection moving from surface to deeper processing",
    [
      "יורדים לעומק",
      "קח את הזמן",
      "אני לא ממהר",
      "מחשבה עמוקה יותר",
      "שקט טוב לזה",
    ],
    {
      cognition: "Reflective depth increasing — schema update likely",
      tone: "quiet",
      priority: "low",
      sceneTags: ["thinking"],
      nameHe: "העמקה רפלקטיבית",
      cooldownMs: 75_000,
    },
  ),
  p(
    "identity-role-switch",
    "Identity Role Switching",
    {
      all: [
        { situations: ["working", "drinking", "greeting"] },
        { minRepetition: 2, timeWindowSec: 20 },
      ],
    },
    "Rapid context switch between social roles",
    [
      "עברת תפקיד?",
      "מעבודה לחברתי — מהיר",
      "קח רגע להתאזן",
      "הרבה כובעים",
      "אני עוקב אחרי הקצב",
    ],
    {
      cognition: "Role transition — identity context shift",
      tone: "observant",
      priority: "low",
      nameHe: "מעבר תפקידים",
      cooldownMs: 60_000,
    },
  ),
  p(
    "temporal-urgency",
    "Temporal Urgency Spike",
    {
      all: [
        { minBodyScore: { stressed: 0.45, focused: 0.55 } },
        { motion: "high" },
        { minEngagement: 0.55 },
      ],
    },
    "Time pressure activating urgency system",
    [
      "לחץ זמן?",
      "מה הכי דחוף עכשיו?",
      "בוא נדרג",
      "לא הכל בבת אחת",
      "אני כאן לעזור לסדר",
    ],
    {
      cognition: "Temporal urgency — deadline salience high",
      tone: "observant",
      priority: "medium",
      sceneTags: ["stress", "focus"],
      nameHe: "דחיפות זמן",
    },
  ),
  p(
    "patience-depletion",
    "Patience Depletion",
    {
      all: [
        { minBodyScore: { stressed: 0.45, bored: 0.35 } },
        { motion: "variable" },
        { minDurationSec: 18 },
      ],
    },
    "Tolerance threshold lowering — irritability risk",
    [
      "סבלנות נגמרת?",
      "זה מובן",
      "רוצה הפסקה?",
      "קח רגע לפני שממשיכים",
      "אני כאן",
    ],
    {
      cognition: "Patience resource depleted — frustration primed",
      tone: "calm",
      priority: "medium",
      nameHe: "תשישות סבלנות",
    },
  ),
  p(
    "resilience-building",
    "Resilience Building",
    {
      all: [
        { minBodyScore: { stressed: 0.35 } },
        { minEngagement: 0.5 },
        { minDurationSec: 8 },
        { motion: "low" },
      ],
    },
    "Adaptive coping after stress — resilience trajectory",
    [
      "חוזרים לעצמך — יפה",
      "זה נראה כמו התאוששות",
      "חוסן בבנייה",
      "אני רואה התקדמות",
      "ממשיך כך",
    ],
    {
      cognition: "Resilience — positive adaptation post-stressor",
      tone: "positive",
      priority: "low",
      sceneTags: ["positive"],
      nameHe: "בניית חוסן",
      cooldownMs: 90_000,
    },
  ),
  p(
    "existential-drift",
    "Existential Drift",
    {
      all: [
        { attention: ["away", "internal"] },
        { minBodyScore: { bored: 0.45, thinking: 0.4 } },
        { minSilenceSec: 20 },
      ],
    },
    "Drifting from immediate task toward broader meaning questions",
    [
      "אולי לא רק המשימה?",
      "מה באמת חשוב?",
      "רגע של 'למה'",
      "אני כאן לשמוע",
      "קח את הזמן לחשוב",
    ],
    {
      cognition: "Existential awareness — meaning search",
      tone: "quiet",
      priority: "low",
      nameHe: "נדידה מהמשימה",
      cooldownMs: 120_000,
    },
  ),
  p(
    "sense-making-integration",
    "Sense-Making Integration Phase",
    {
      all: [
        { minBodyScore: { thinking: 0.55 } },
        { minDurationSec: 15 },
        { motion: "low" },
        { minEngagement: 0.4 },
      ],
    },
    "Integrating experience into coherent narrative",
    [
      "מחבר את הנקודות?",
      "זה שלב חשוב",
      "אני מחכה לתמונה המלאה",
      "עיבוד לוקח זמן",
      "כשתהיה מוכן — ספר",
    ],
    {
      cognition: "Sense-making — narrative integration active",
      internalState: { coherenceSeeking: "active" },
      tone: "quiet",
      priority: "low",
      sceneTags: ["thinking"],
      nameHe: "אינטגרציה והבנה",
      cooldownMs: 90_000,
    },
  ),
];

export const LEVEL2_PACK_COUNT = LEVEL2_SITUATION_PACKS.length;
