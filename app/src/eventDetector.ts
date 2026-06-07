import type { SemanticEvent, SemanticEventType } from "./worldMemory";

const EXCITED_TYPES: SemanticEventType[] = [
  "person_entered",
  "person_left",
  "object_appeared",
  "object_removed",
  "door_opened",
  "user_returned",
];

export const isExcitedEvent = (ev: SemanticEvent): boolean =>
  ev.significant || EXCITED_TYPES.includes(ev.type);

export const isCuriousSubject = (subject?: string, objects: string[] = []): boolean => {
  const s = (subject ?? "").toLowerCase();
  const pool = [s, ...objects.map((o) => o.toLowerCase())].join(" ");
  return (
    /guitar|gitar|גיטרה/i.test(pool) ||
    /laptop|computer|מחשב|מסך|screen|monitor/i.test(pool) ||
    /book|ספר|phone|טלפון|keyboard/i.test(pool) ||
    objects.length >= 3
  );
};

/** Only significant events reach the character layer. */
export const filterEventsForCharacter = (events: SemanticEvent[]): SemanticEvent[] =>
  events.filter((e) => e.significant || e.type !== "generic");

export const pickPrimaryEvent = (events: SemanticEvent[]): SemanticEvent | null => {
  const filtered = filterEventsForCharacter(events);
  if (!filtered.length) return null;
  const rank = (t: SemanticEventType, ev: SemanticEvent) => {
    const sub = ev.subject ?? "";
    if (sub === "stood_with_drink" || sub.startsWith("object_held:")) return 0;
    if (/wave|arm_movement|motion_burst/.test(sub)) return 1;
    if (t === "user_returned" || t === "person_entered") return 2;
    if (t === "door_opened" || t === "object_appeared") return 3;
    if (t === "object_removed" || t === "person_left") return 4;
    if (t === "activity_change") return 5;
    return 6;
  };
  return [...filtered].sort((a, b) => rank(a.type, a) - rank(b.type, b))[0];
};

export const topicKeyFromEvent = (ev: SemanticEvent): string => {
  if (ev.subject) return normalizeTopic(ev.subject);
  return normalizeTopic(ev.type);
};

export const normalizeTopic = (s: string): string =>
  s
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9\u0590-\u05FF]+/g, "_")
    .slice(0, 48);
