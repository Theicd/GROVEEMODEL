import {
  filterEventsForCharacter,
  isCuriousSubject,
  isExcitedEvent,
  normalizeTopic,
  pickPrimaryEvent,
  topicKeyFromEvent,
} from "./eventDetector";
import { findSituationUtterance, loadSituationRegistry } from "./situationRegistry";
import type { SemanticEvent, WorldMemory } from "./worldMemory";

export type CharacterMood = "observing" | "curious" | "bored" | "excited";

export type CharacterDecision = {
  mood: CharacterMood;
  message: string;
  topic: string;
  reason: string;
};

export const CHARACTER_CONFIG = {
  /** Gentle first question about baseline scene (user silent). */
  baselineCuriousAfterMs: 30_000,
  curiousAfterMs: 90_000,
  boredStableMs: 300_000,
  boredSpeakIntervalMs: 240_000,
  excitedCooldownMs: 25_000,
  /** Wave / held object / stood with drink — respond faster. */
  urgentCooldownMs: 12_000,
  generalCooldownMs: 60_000,
  topicMentionTtlMs: 900_000,
  maxRecentQuestions: 12,
  /** Proactive when user left frame for this long. */
  absentSpeakAfterMs: 300_000,
  /** Staring at screen — low motion + person present + user silent. */
  screenStareAfterMs: 180_000,
} as const;

export class CharacterBrain {
  mood: CharacterMood = "observing";
  curiosity = 0.35;
  boredom = 0.1;
  lastUserInteractionAt = Date.now();
  lastProactiveAt = 0;
  baselineSceneAt = 0;
  baselineIntroDone = false;
  topicsMentioned = new Map<string, number>();
  recentQuestions: { text: string; ts: number }[] = [];

  reset(): void {
    this.mood = "observing";
    this.curiosity = 0.35;
    this.boredom = 0.1;
    this.lastUserInteractionAt = Date.now();
    this.lastProactiveAt = 0;
    this.baselineSceneAt = 0;
    this.baselineIntroDone = false;
    this.topicsMentioned.clear();
    this.recentQuestions = [];
  }

  noteBaselineScene(): void {
    this.baselineSceneAt = Date.now();
    this.mood = "observing";
  }

  recordUserInteraction(): void {
    this.lastUserInteractionAt = Date.now();
    this.boredom = Math.max(0, this.boredom - 0.25);
    this.curiosity = Math.min(1, this.curiosity + 0.1);
    if (this.mood === "bored") this.mood = "observing";
  }

  msSinceUserInteraction(): number {
    return Date.now() - this.lastUserInteractionAt;
  }

  wasTopicMentionedRecently(topic: string): boolean {
    const key = normalizeTopic(topic);
    const ts = this.topicsMentioned.get(key);
    if (!ts) return false;
    return Date.now() - ts < CHARACTER_CONFIG.topicMentionTtlMs;
  }

  markSpoke(decision: CharacterDecision): void {
    this.lastProactiveAt = Date.now();
    this.topicsMentioned.set(normalizeTopic(decision.topic), Date.now());
    if (decision.reason === "curious:baseline") this.baselineIntroDone = true;
    this.recentQuestions.unshift({ text: decision.message, ts: Date.now() });
    this.recentQuestions = this.recentQuestions.slice(0, CHARACTER_CONFIG.maxRecentQuestions);
    this.mood = decision.mood;
  }

  evaluate(world: WorldMemory, freshEvents: SemanticEvent[]): CharacterDecision | null {
    const now = Date.now();
    const msIdle = this.msSinceUserInteraction();
    const msSinceProactive = this.lastProactiveAt ? now - this.lastProactiveAt : Number.POSITIVE_INFINITY;
    const msSinceChange = world.msSinceLastChange();

    this.boredom = msSinceChange > 120_000 ? Math.min(1, this.boredom + 0.02) : Math.max(0, this.boredom - 0.01);
    this.curiosity = msIdle > CHARACTER_CONFIG.curiousAfterMs ? Math.min(1, this.curiosity + 0.03) : this.curiosity;

    const significant = filterEventsForCharacter(freshEvents);
    const primary = pickPrimaryEvent(significant);

    if (primary?.type === "activity_change") {
      const sub = primary.subject ?? "";
      if (isSituationActivity(sub)) {
        const urgent = isUrgentSituation(sub);
        const cooldown = urgent ? CHARACTER_CONFIG.urgentCooldownMs : CHARACTER_CONFIG.excitedCooldownMs;
        if (msSinceProactive >= cooldown) {
          const topic = situationTopic(sub);
          if (!this.wasTopicMentionedRecently(topic)) {
            const message = utteranceSituation(sub, world);
            if (message) {
              const mood: CharacterMood =
                sub.startsWith("object_held:") || sub === "stood_with_drink" ? "curious" : "excited";
              return {
                mood,
                message,
                topic,
                reason: `situation:${sub}`,
              };
            }
          }
        }
      }
    }

    if (primary?.type === "person_entered") {
      if (msSinceProactive >= CHARACTER_CONFIG.excitedCooldownMs) {
        const topic = "person";
        if (!this.wasTopicMentionedRecently(topic)) {
          return {
            mood: "curious",
            message: "שמתי לב שאתה בפריים — לא ראיתי אותך קודם. מה קורה?",
            topic,
            reason: "curious:person_entered",
          };
        }
      }
    }

    if (primary && isExcitedEvent(primary) && primary.type !== "person_entered") {
      if (msSinceProactive >= CHARACTER_CONFIG.excitedCooldownMs) {
        const topic = topicKeyFromEvent(primary);
        if (!this.wasTopicMentionedRecently(topic)) {
          const message = utteranceExcited(primary, world);
          if (message) {
            return { mood: "excited", message, topic, reason: `excited:${primary.type}` };
          }
        }
      }
    }

    const msSinceBaseline = this.baselineSceneAt ? now - this.baselineSceneAt : Number.POSITIVE_INFINITY;
    if (
      !this.baselineIntroDone &&
      world.baselineEstablished &&
      this.baselineSceneAt > 0 &&
      msSinceBaseline >= CHARACTER_CONFIG.baselineCuriousAfterMs &&
      msIdle >= CHARACTER_CONFIG.baselineCuriousAfterMs &&
      msSinceProactive >= CHARACTER_CONFIG.generalCooldownMs
    ) {
      const curious = utteranceCurious(world, this);
      if (curious) {
        return { mood: "curious", message: curious.message, topic: curious.topic, reason: "curious:baseline" };
      }
    }

    if (
      msIdle >= CHARACTER_CONFIG.curiousAfterMs &&
      msSinceProactive >= CHARACTER_CONFIG.generalCooldownMs &&
      this.curiosity >= 0.35
    ) {
      const curious = utteranceCurious(world, this);
      if (curious) {
        return { mood: "curious", message: curious.message, topic: curious.topic, reason: "curious:idle" };
      }
    }

    if (
      msSinceChange >= CHARACTER_CONFIG.boredStableMs &&
      msIdle >= CHARACTER_CONFIG.boredStableMs &&
      msSinceProactive >= CHARACTER_CONFIG.boredSpeakIntervalMs &&
      this.boredom >= 0.45
    ) {
      const topic = "ambient_silence";
      if (!this.wasTopicMentionedRecently(topic)) {
        return {
          mood: "bored",
          message: utteranceBored(),
          topic,
          reason: "bored:stable",
        };
      }
    }

    if (
      !world.personPresent &&
      world.absentSince > 0 &&
      world.msSinceAbsent() >= CHARACTER_CONFIG.absentSpeakAfterMs &&
      msSinceProactive >= CHARACTER_CONFIG.generalCooldownMs
    ) {
      const topic = "user_absent";
      if (!this.wasTopicMentionedRecently(topic)) {
        return {
          mood: "curious",
          message: "נראה שהשארת אותי לבד עם החדר.",
          topic,
          reason: "presence:alone",
        };
      }
    }

    if (
      world.personPresent &&
      world.lastMotionLevel < 0.04 &&
      msIdle >= CHARACTER_CONFIG.screenStareAfterMs &&
      msSinceProactive >= CHARACTER_CONFIG.generalCooldownMs
    ) {
      const topic = "screen_stare";
      if (!this.wasTopicMentionedRecently(topic)) {
        return {
          mood: "curious",
          message: "שמתי לב שאתה כבר כמה דקות בוהה במסך — הכל בסדר?",
          topic,
          reason: "presence:staring",
        };
      }
    }

    this.mood = "observing";
    return null;
  }
}

function isSituationActivity(subject: string): boolean {
  return (
    /^(wave|arm_movement|motion_burst|focused_work|pose_change|stood_with_drink|hands_on_head|hand_on_face|gesture:thumbs_up)$/.test(
      subject,
    ) ||
    subject.startsWith("object_held:") ||
    subject.startsWith("object:") ||
    subject.startsWith("pose_change:")
  );
}

function isUrgentSituation(subject: string): boolean {
  return (
    /^(wave|arm_movement|motion_burst|stood_with_drink)$/.test(subject) ||
    subject.startsWith("object_held:")
  );
}

function situationTopic(subject: string): string {
  if (subject.startsWith("object_held:")) return normalizeTopic(subject);
  return normalizeTopic(subject);
}

function utteranceSituation(subject: string, world: WorldMemory): string | null {
  const registryLine = findSituationUtterance(loadSituationRegistry(), subject);
  if (registryLine) return registryLine;

  if (subject === "stood_with_drink") {
    const drink = world.holding.find((h) => /cup|bottle/.test(h));
    if (drink === "cup") return "נראה שקמת עם כוס — רגע של קפה?";
    if (drink === "bottle") return "קמת עם בקבוק — הפסקה קצרה?";
    return "נראה שקמת עם משהו לשתות — רגע מנוחה?";
  }
  if (subject.startsWith("object_held:")) {
    const item = subject.slice("object_held:".length);
    if (item === "cup") return "שמתי לב לכוס ביד — הפסקת קפה?";
    if (item === "bottle") return "נראה שיש בקבוק ביד — מתכנן הפסקה?";
    if (item === "book") return "עכשיו עם ספר — על מה אתה קורא?";
    if (item === "phone") return "טלפון ביד — משהו דחוף?";
    return `שמתי לב ש-${item} ביד — מעניין.`;
  }
  if (subject.startsWith("object:")) {
    const item = subject.slice("object:".length);
    if (item === "guitar") return "שמתי לב שיש גיטרה בחדר — אתה מנגן?";
    return `שמתי לב ל-${item} בסצנה — מעניין.`;
  }
  return utteranceMotion(subject);
}

function utteranceMotion(subject: string): string | null {
  switch (subject) {
    case "wave":
      return "אני די בטוח שניסית למשוך את תשומת הלב שלי עכשיו.";
    case "arm_movement":
      return "אני רואה תנועה בידיים — משהו שאתה רוצה להראות לי?";
    case "motion_burst":
      return "אני רואה הרבה תנועה פתאום. בודק שאני עדיין איתך?";
    case "focused_work":
      return "נראה שאתה שקוע במשהו מולך — לא אפריע.";
    case "pose_change":
      return "שמתי לב לשינוי קטן בתנוחה — הכל בסדר?";
    default:
      return null;
  }
}

function utteranceExcited(ev: SemanticEvent, world: WorldMemory): string | null {
  const sub = ev.subject ?? "";
  switch (ev.type) {
    case "user_returned": {
      const dur = world.lastAbsentDurationMs;
      if (dur >= 300_000) return "ברוך שובך — היה שקט כאן בלעדיך.";
      if (dur >= 60_000) return "שמח לראות אותך שוב — עדכנתי את מה שקורה סביב.";
      return "חזרת — טוב לראות אותך שוב.";
    }
    case "person_entered":
      return "שמתי לב שמישהו בפריים.";
    case "door_opened":
      return "נראה שהדלת נפתחה. הכל בסדר?";
    case "object_appeared":
      if (/phone|טלפון/i.test(sub)) return "שמתי לב לטלפון — משהו דחוף?";
      if (/television|tv/i.test(sub)) return "נראה שהטלוויזיה דולקת — אתה צופה במשהו?";
      if (/laptop|computer|מחשב/i.test(sub)) return "יש מחשב פתוח — עובדים על משהו?";
      return sub ? `שמתי לב ל-${sub} — זה חדש כאן.` : null;
    case "object_removed":
      return sub ? `שמתי לב ש${sub} כבר לא נראה.` : "משהו שהיה כאן קודם — נעלם מהפריים.";
    case "person_left":
      return "נראה שמישהו יצא מהפריים.";
    case "activity_change":
      if (/laptop|computer|מחשב/i.test(sub + world.lastSummary)) {
        return "אני רואה שאתה עובד מול המחשב.";
      }
      return null;
    default:
      return ev.text ? `שמתי לב: ${ev.text}.` : null;
  }
}

function utteranceCurious(
  world: WorldMemory,
  brain: CharacterBrain,
): { message: string; topic: string } | null {
  for (const obj of world.objects) {
    const topic = normalizeTopic(obj);
    if (brain.wasTopicMentionedRecently(topic)) continue;
    const lower = obj.toLowerCase();
    if (/door|דלת/.test(lower)) {
      return { message: "שמתי לב שהדלת פתוחה — הכל בסדר?", topic: "door" };
    }
    if (/guitar|gitar|גיטרה/.test(lower)) {
      return { message: "שמתי לב שיש גיטרה בחדר — אתה מנגן?", topic };
    }
    if (/laptop|computer|מחשב/.test(lower)) {
      return { message: "נראה שיש מחשב פתוח — על מה אתה עובד?", topic };
    }
    if (/book|ספר/.test(lower)) {
      return { message: "יש ספר על השולחן — על מה אתה קורא?", topic };
    }
    if (/clock|שעון/.test(lower)) {
      return { message: "יש שעון בפריים — אתה שם לב לזמן כשאתה עובד?", topic };
    }
  }

  if (world.objects.length >= 3 && !brain.wasTopicMentionedRecently("multi_screen")) {
    const screens = world.objects.filter((o) => /screen|monitor|מסך|laptop|computer/i.test(o));
    if (screens.length >= 2) {
      return { message: "יש כמה מסכים פתוחים. אתה עובד על פיתוח?", topic: "multi_screen" };
    }
  }

  if (world.lastSummary.trim() && !brain.wasTopicMentionedRecently("atmosphere")) {
    const sum = world.lastSummary.toLowerCase();
    if (/blue|אור כחול|mysterious|atmospheric|dim|dark/.test(sum)) {
      return {
        message: "יש פה אווירה מיוחדת — כמעט כמו סצנה מסרט. זה מכוון?",
        topic: "atmosphere",
      };
    }
  }

  if (isCuriousSubject(undefined, world.objects) && world.objects.length && !brain.wasTopicMentionedRecently("scene_general")) {
    return {
      message: "אני כאן — יש משהו שאתה רוצה לשתף על מה שקורה סביב?",
      topic: "scene_general",
    };
  }

  return null;
}

function utteranceBored(): string {
  const lines = [
    "די שקט כאן לאחרונה.",
    "אני עדיין במצב תצפית — לא הרבה השתנה בסביבה.",
    "שקט… אני כאן אם תרצה לדבר.",
  ];
  return lines[Math.floor(Math.random() * lines.length)];
}

export const moodLabelHe = (mood: CharacterMood): string => {
  switch (mood) {
    case "observing":
      return "תצפית";
    case "curious":
      return "סקרן";
    case "bored":
      return "משועמם";
    case "excited":
      return "נרגש";
  }
};

export const moodStatusLine = (mood: CharacterMood): string => {
  switch (mood) {
    case "observing":
      return "👁 Character · תצפית";
    case "curious":
      return "👁 Character · סקרן";
    case "bored":
      return "👁 Character · שקט";
    case "excited":
      return "👁 Character · מגיב";
  }
};
