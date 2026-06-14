import { sanitizeSearchQuery } from "../webSearch/queryExtract";
import type { StartupContext } from "./types";

const DAY_HE = ["ראשון", "שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת"];

/** Questions answerable from StartupContext alone — no web search. */
export const isLocalContextTimeQuery = (text: string): boolean => {
  const q = sanitizeSearchQuery(text);
  if (
    /(?:מה\s+)?(?:ה)?שע(?:ה|ת)\s+(?:ב|ב־|in|at|for)\s+/i.test(q) ||
    /(?:what\s+)?time\s+(?:in|at|for)\s+/i.test(q) ||
    /(?:כמה\s+)?(?:שעות?\s+)?(?:ה)?(?:פרש|הפרש)\s+/i.test(q) ||
    /time\s+(?:zone\s+)?(?:difference|offset)/i.test(q)
  ) {
    return false;
  }
  return (
    /(?:מה\s+)?(?:ה)?שע(?:ה|ת)\s*(?:עכשיו|כרגע|now)?\s*[?!.]?$/i.test(q) ||
    /^what(?:'s|\s+is)?\s+(?:the\s+)?time(?:\s+is\s+it|\s+now)?\s*[?!.]?$/i.test(q) ||
    /^what\s+time\s+is\s+it\s*[?!.]?$/i.test(q) ||
    /(?:מה\s+)?(?:ה)?תאריך\s*(?:היום|עכשיו|today|now)?\s*[?!.]?$/i.test(q) ||
    /^what(?:'s|\s+is)?\s+(?:the\s+)?date(?:\s+today)?\s*[?!.]?$/i.test(q) ||
    /איזה\s+יום(?:\s+היום)?\s*[?!.]?$/i.test(q) ||
    /^what\s+day\s+(?:is\s+it|today)?\s*[?!.]?$/i.test(q) ||
    /(?:מה\s+)?(?:מספר\s+)?(?:ה)?שבוע\s+(?:בשנה|number)?\s*[?!.]?$/i.test(q) ||
    /^week\s+number\s*[?!.]?$/i.test(q) ||
    /(?:האם\s+)?(?:עכשיו|כרגע)\s+(?:לילה|יום|בוקר|ערב)/i.test(q) ||
    /(?:is\s+it)\s+(?:day|night|morning|evening)\s+(?:now)?/i.test(q)
  );
};

const formatInTz = (ctx: StartupContext, opts: Intl.DateTimeFormatOptions): string => {
  try {
    const d = ctx.datetime ? new Date(ctx.datetime) : new Date();
    return new Intl.DateTimeFormat("he-IL", { timeZone: ctx.timezone, ...opts }).format(d);
  } catch {
    return new Date().toLocaleString("he-IL", opts);
  }
};

export const buildLocalTimeAnswer = (ctx: StartupContext, query: string): string => {
  const q = sanitizeSearchQuery(query);
  const place = ctx.cityName
    ? `${ctx.cityName}, ${ctx.countryName}`
    : ctx.countryName;
  const timeStr = formatInTz(ctx, {
    weekday: "long",
    year: "numeric",
    month: "long",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
  const hour = formatInTz(ctx, { hour: "2-digit", hour12: false });
  const h = parseInt(hour, 10);
  const dayPart =
    h >= 5 && h < 12 ? "בוקר" : h >= 12 && h < 17 ? "צהריים" : h >= 17 && h < 21 ? "ערב" : "לילה";

  const lines = [
    `[LOCAL CONTEXT — no web fetch]`,
    `מיקום (IP): ${place}`,
    `אזור זמן: ${ctx.timezone} (${ctx.utcOffset}${ctx.dst ? ", DST" : ""})`,
    `שעה מקומית: ${timeStr}`,
    `יום בשבוע: ${DAY_HE[ctx.dayOfWeek] ?? ctx.dayOfWeek}`,
  ];
  if (ctx.weekNumber != null) lines.push(`שבוע בשנה: ${ctx.weekNumber}`);
  if (/(?:לילה|יום|בוקר|ערב|day|night|morning|evening)/i.test(q)) {
    lines.push(`כרגע ${dayPart} באזורך.`);
  }
  lines.push(`[/LOCAL CONTEXT]`);
  return lines.join("\n");
};

export const buildStartupPromptBlock = (ctx: StartupContext): string => {
  const place = ctx.cityName
    ? `${ctx.cityName}, ${ctx.countryName} (${ctx.countryCode})`
    : `${ctx.countryName} (${ctx.countryCode})`;
  const timeShort = formatInTz(ctx, {
    weekday: "short",
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
  return (
    `User locale context (from IP/time API, approximate): ${place}; ` +
    `timezone ${ctx.timezone} ${ctx.utcOffset}; local time ${timeShort}. ` +
    `Use for "what time here", local weather, and "near me" searches unless user names another place.`
  );
};
