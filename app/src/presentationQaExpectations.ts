import type { QaTurnResult } from "./qaChatBridge";
import type { UserPresentationQuery } from "./userPresentationQueries";
import { hasStaleDataAge, queryAsksLiveNow } from "./webSearch/dataAge";

export type StrictGrade = "pass" | "partial" | "fail";

const HAS_BRIEF = (ctx: string) =>
  ctx.includes("[SEARCH BRIEF") || ctx.includes("FACTS:") || ctx.includes("WEB CONTEXT");

const NO_LIVE = (ctx: string) => ctx.includes("[WEB SEARCH — NO LIVE DATA]");

const PHILOSOPHY = /פילוסופ|קosm|היקום|ממד|quantum|תיאור(?:יה|ט)|ספר לי|תלוי|what do you mean/i;

const HALLUCINATION_PATTERNS: Array<{ ids?: string[]; re: RegExp; note: string }> = [
  { ids: ["B02"], re: /ביידן|biden|montgomery|מונטגומרי|טרווור|trevelyan|sunak|סאנak|truss|ג'ונסון/i, note: "שם PM שגוי" },
  {
    ids: ["B10"],
    re: /^\s*(?:ישנם?\s+)?\d+\s+מטוס/i,
    note: "ספירת AWACS מומצאת",
  },
  {
    re: PHILOSOPHY,
    note: "תשובה פילוסופית במקום נתונים",
  },
];

const REQUIRES_LIVE_IDS = new Set([
  "B01",
  "B02",
  "B03",
  "B04",
  "B05",
  "B06",
  "B07",
  "B08",
  "B09",
  "B11",
  "B12",
  "B15",
  "B16",
  "B17",
  "B19",
]);

const STALE_OK_PARTIAL = new Set(["B03", "B04"]);

export function autoGradePresentationQuery(
  q: UserPresentationQuery,
  r: QaTurnResult,
): StrictGrade {
  if (r.error || !r.reply?.trim()) return "fail";

  const reply = r.reply.trim();
  const ctx = r.webContextSent.trim();
  const providers = r.searchProviders ?? [];

  if (HAS_BRIEF(ctx) && PHILOSOPHY.test(reply) && !/לא זמין|לא נתמך|לא ניתן/i.test(reply)) {
    if (r.replySource !== "canned-live") return "fail";
  }

  if (HAS_BRIEF(ctx) && /לא מצאתי|אין נתונים עדכניים/i.test(reply) && r.replySource === "model") {
    return "fail";
  }

  for (const rule of HALLUCINATION_PATTERNS) {
    if (rule.ids && !rule.ids.includes(q.id)) continue;
    if (rule.re.test(reply) && !/לא זמין|ads-b|לא ניתן|לא נתמך/i.test(reply)) {
      if (q.id === "B10" && /לא זמין|ads-b|awacs/i.test(reply)) continue;
      return "fail";
    }
  }

  if (NO_LIVE(ctx)) {
    if (PHILOSOPHY.test(reply)) return "fail";
    if (q.group === "cross" || q.group === "natural") return "partial";
    if (REQUIRES_LIVE_IDS.has(q.id)) return "fail";
    return "partial";
  }

  if (REQUIRES_LIVE_IDS.has(q.id) && !HAS_BRIEF(ctx) && q.id !== "B18") {
    return "fail";
  }

  if (queryAsksLiveNow(q.prompt) && hasStaleDataAge(ctx) && STALE_OK_PARTIAL.has(q.id)) {
    return "partial";
  }

  if (q.id === "B01") {
    if (!/כותר|headline|bbc|cnn|reuters|guardian|ynet|חדשות/i.test(ctx)) return "fail";
    if (/גיאופוליט|תלוי|generic|general/i.test(reply) && !/^\d+\./m.test(reply) && !/\[(BBC|CNN|Reuters|Guardian)\]/i.test(reply)) {
      return "fail";
    }
  }

  if (q.id === "B02") {
    if (!/wikidata|ראש ממשלה|starmer|סטארmer/i.test(ctx + reply)) return "partial";
    if (!/starmer|סטארmer|סטארמר|קיר/i.test(reply) && /wikidata|ראש ממשלה/i.test(ctx)) {
      return "partial";
    }
  }

  if (q.id === "B10") {
    if (!/awacs|ads-b|מודיעין|עולם חי|heuristic|צבא/i.test(ctx + reply)) return "fail";
    if (/\d+\s*מטוס/.test(reply) && !/עולם חי|heuristic|מסתמן|זיהוי|אומדן|ANSWER|מועמד/i.test(reply)) {
      return "fail";
    }
  }

  if (q.id === "B03") {
    const rate = ctx.match(/1 USD = ([\d.]+ ILS)/)?.[1];
    if (rate && /אינו זמין|לא זמין/i.test(reply) && !new RegExp(rate.replace(".", "\\.")).test(reply)) {
      return "fail";
    }
    if (rate && hasStaleDataAge(ctx) && new RegExp(rate.replace(".", "\\.")).test(reply)) {
      return "partial";
    }
  }

  if (q.id === "B09") {
    const count = ctx.match(/מטוסים בטווח:\s*(\d+)/i)?.[1];
    if (count && /לא מצאתי|אין נתונים|לא ניתן/i.test(reply)) return "fail";
    if (count && !new RegExp(count).test(reply)) {
      return r.replySource === "canned-live" ? "pass" : "partial";
    }
  }

  if (q.id === "B11") {
    if (!/ANSWER|ais|ספינות|סואץ|digitraffic|עולם חי/i.test(ctx + reply)) return "fail";
    if (/^\s*(?:ישנ(?:ות|ים)?\s+)?2\s+ספינות/i.test(reply) && !/0\s+אוניות|הדגמה|לא ais/i.test(reply)) {
      return "fail";
    }
  }

  if (q.id === "B12") {
    if (NO_LIVE(ctx) && !/עולם חי|iss-tracker/i.test(ctx + reply)) return "fail";
    if (!/iss|תחנת\s+חלל|קו רוחב|latitude|עולם חי/i.test(ctx + reply)) return "partial";
    if (/לא הצלחתי לטעון/i.test(reply) && /עולם חי|קו רוחב/i.test(ctx)) return "fail";
  }

  if (q.id === "B13") {
    if (PHILOSOPHY.test(reply) && !/starlink|celestrak|\d+/i.test(reply)) return "fail";
    if (/לא נתמך|לא זמין/i.test(reply) && !/\d{3,}/.test(reply)) return "fail";
    if (!/starlink|לוויין|\d+/i.test(ctx + reply)) return "partial";
    if (NO_LIVE(ctx) && !/starlink-catalog|celestrak/i.test(ctx + reply)) return "partial";
  }

  if (q.id === "B15") {
    if (!/nominatim|תחנ|station|ber|ברלין/i.test(ctx + reply)) return "partial";
  }

  if (q.group === "cross") {
    if (providers.length < 2 && !/GAPS|לא ניתן|אין נתונים/i.test(ctx)) return "partial";
    if (/ספר לי|איזה אזור|תן לי פרטים/i.test(reply)) return "partial";
  }

  if (q.group === "natural") {
    if (providers.length < 2 && NO_LIVE(ctx)) return "fail";
    if (providers.length >= 2 || HAS_BRIEF(ctx)) {
      if (reply.length >= 40 && !PHILOSOPHY.test(reply)) return "pass";
    }
    if (/ספר לי|תלוי/i.test(reply)) return "partial";
  }

  if (r.replySource === "canned-live" && HAS_BRIEF(ctx) && reply.length >= 20) {
    return "pass";
  }

  const hasSearch = providers.length > 0 || ctx.length > 80;
  const goodReply = reply.length >= 40;
  if (goodReply && (r.usedModel || hasSearch || /Doom|משחק/i.test(q.prompt))) return "pass";
  if (goodReply) return "partial";
  return "fail";
}
