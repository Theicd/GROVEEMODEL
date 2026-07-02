/**
 * Pre-LLM routing: meta capabilities, reset, and game vs live-data disambiguation.
 * Regex/heuristics only — no LLM calls.
 */

import type { ChatTopic } from "./chatIntents";
import { isSimpleGreeting } from "./chatIntents";
import { buildSimpleGreetingReply } from "./chatInlineContent";
import { isInlineTextTaskRequest } from "./chatComposition";
import { isGameSearchRequest, isTriviaOrSocialGame } from "./gameSearch/gameIntents";
import { shouldOpenGamePanel } from "./gameSearch";
import {
  isAviationQuery,
  isDisasterQuery,
  isEarthquakeQuery,
  isNewsQuery,
  isShipsQuery,
  needsWebSearch,
} from "./webSearch/intents";
import { isLocalContextTimeQuery } from "./startupContext";
import type { QaReplySource } from "./qaChatBridge";

export type UiLang = "he" | "en";

const META_CAPABILITIES_RE =
  /^(?:מי\s+אתה|מה\s+אתה\s+(?:יודע|יודעת|יכול|יכולה|יודע\s+לעשות|יכול\s+לעשות)|ספר\s+לי\s+ע(?:ל\s+)?(?:עצמך|ך)|הצג\s+(?:א(?:ת\s+)?)?עצמך|איזה\s+פונקציות|מה\s+ה(?:יכולות|פונקציות)|what\s+can\s+you|who\s+are\s+you|tell\s+me\s+about\s+yourself|present\s+yourself|your\s+capabilities|what\s+are\s+you\s+capable)/i;

const RESET_COMMAND_RE =
  /^\/reset\b|^איפוס(?:\s+שיחה|\s+הקשר)?(?:[.!?]*)$|^reset\s+conversation(?:[.!?]*)$/i;

const BORED_HINT_RE = /משעמם|bored|kill\s+time/i;

const EXPLICIT_GAME_RE =
  /(?:^|\s)(?:משחק(?:ים)?|נשחק|רוצה\s+לשחק|play\s+(?:a\s+)?game|let'?s\s+play|\bplay\b|פוקימון|pokemon|ארקייד|arcade)(?:\s|$|[?!.])/i;

const LIVE_DATA_HINT_RE =
  /רעיד|earthquake|tsunami|צונאמ|מטוס|aircraft|adsb|תעופ|flight|מזג|weather|סופה|hurricane|אסון|disaster|gdacs|usgs|חדשות|headline|כותר|breaking|מטוסים|ships|ספינ|ais\b|רעידות|seismic/i;

export type DisambiguationResult =
  | { kind: "clarify"; replyHe: string; replyEn: string }
  | { kind: "route"; preferGames: boolean; preferLiveSearch: boolean };

export type EarlyTurnRouting =
  | { action: "canned"; reply: string; replySource: QaReplySource; resetSession?: boolean }
  | {
      action: "continue";
      wantsGameSearch: boolean;
      shouldRunWebSearch: boolean;
      chatTopic: ChatTopic;
    };

export function isMetaCapabilitiesQuery(text: string): boolean {
  const t = text.trim();
  if (!t || isInlineTextTaskRequest(t)) return false;
  if (t.length > 120) return false;
  return META_CAPABILITIES_RE.test(t);
}

export function buildGrooveCapabilitiesReply(uiLang: UiLang): string {
  if (uiLang === "he") {
    return [
      "אני **GROVEE** (גרובי) — עוזר AI בדפדפן, בלי ענן.",
      "",
      "**מה אפשר לעשות כאן:**",
      "• חיפוש חי — רעידות אדמה, מטוסים, ספינות, מזג אוויר, שערי מט\"ח, חדשות",
      "• מפת REALITY LIVE (Cesium) — הצגת מדינות וערים",
      "• משחקי ארקייד מהארכיון — כשמבקשים במפורש «משחק» או «play»",
      "• רדיו וטלוויזיה חיים, התרעות בזמן אמת",
      "• יצירת תמונות — בחר מודל תמונה בבורר למעלה",
      "",
      "נסה: «רעידות אדמה אחרונות», «כמה מטוסים מעל ישראל», «משחקי ארקייד משנות ה80».",
    ].join("\n");
  }
  return [
    "I'm **GROVEE** — a browser AI assistant (no cloud chat).",
    "",
    "**What you can do here:**",
    "• Live search — earthquakes, flights, ships, weather, FX, news",
    "• REALITY LIVE map (Cesium) — countries and cities",
    "• Retro arcade games — when you explicitly ask to play or search games",
    "• Live radio/TV and real-time alerts",
    "• Image generation — pick an image model in the rack",
    "",
    "Try: “recent earthquakes”, “flights over Israel”, “80s arcade games”.",
  ].join("\n");
}

export function handleMetaIntent(text: string, uiLang: UiLang): string | null {
  if (!isMetaCapabilitiesQuery(text)) return null;
  return buildGrooveCapabilitiesReply(uiLang);
}

export function handleResetCommand(text: string, uiLang: UiLang): string | null {
  const t = text.trim();
  if (!RESET_COMMAND_RE.test(t)) return null;
  return uiLang === "he"
    ? "אופס! איפסתי את השיחה. אפשר להתחיל מחדש."
    : "Done — conversation reset. You can start fresh.";
}

function hasExplicitLiveDataIntent(text: string): boolean {
  const t = text.trim();
  if (!t) return false;
  if (
    isEarthquakeQuery(t) ||
    isAviationQuery(t) ||
    isShipsQuery(t) ||
    isDisasterQuery(t) ||
    isNewsQuery(t)
  ) {
    return true;
  }
  return LIVE_DATA_HINT_RE.test(t);
}

function hasExplicitGameIntent(text: string): boolean {
  const t = text.trim();
  if (!t) return false;
  if (isTriviaOrSocialGame(t)) return false;
  if (isGameSearchRequest(t)) return true;
  return EXPLICIT_GAME_RE.test(t);
}

function hasBoredGameLean(text: string): boolean {
  return BORED_HINT_RE.test(text.trim());
}

/** Tie-breaker when both play/bored and live-data signals appear. */
export function disambiguateIntent(text: string, _chatTopic: ChatTopic): DisambiguationResult {
  const t = text.trim();
  if (!t) return { kind: "route", preferGames: false, preferLiveSearch: false };

  const live = hasExplicitLiveDataIntent(t);
  const game = hasExplicitGameIntent(t);
  const bored = hasBoredGameLean(t);

  if (live && !game && !bored) {
    return { kind: "route", preferGames: false, preferLiveSearch: true };
  }
  if (game && !live) {
    return { kind: "route", preferGames: true, preferLiveSearch: false };
  }
  if (live && (game || bored)) {
    return {
      kind: "clarify",
      replyHe:
        "יש כאן שני כיוונים אפשריים: **מידע חי** (רעידות אדמה, חדשות, מטוסים…) או **משחקים**.\n\nהאם התכוונת לרעידות אדמה / מידע עדכני, או למשחקים?",
      replyEn:
        "This could mean **live data** (earthquakes, news, flights…) or **games**.\n\nDid you mean live earthquake/info, or games?",
    };
  }

  return { kind: "route", preferGames: false, preferLiveSearch: false };
}

export function resolveEarlyTurnRouting(input: {
  text: string;
  effectivePrompt: string;
  chatTopic: ChatTopic;
  uiLang: UiLang;
  startupContext: StartupContext | null;
  blockGames: boolean;
  blockSearch?: boolean;
}): EarlyTurnRouting {
  const query = (input.text || input.effectivePrompt).trim();

  const resetReply = handleResetCommand(query, input.uiLang);
  if (resetReply) {
    return {
      action: "canned",
      reply: resetReply,
      replySource: "reset",
      resetSession: true,
    };
  }

  if (isSimpleGreeting(query)) {
    return {
      action: "canned",
      reply: buildSimpleGreetingReply(input.uiLang),
      replySource: "greeting",
    };
  }

  const metaReply = handleMetaIntent(query, input.uiLang);
  if (metaReply) {
    return { action: "canned", reply: metaReply, replySource: "meta-capabilities" };
  }

  const dis = disambiguateIntent(query, input.chatTopic);
  if (dis.kind === "clarify") {
    return {
      action: "canned",
      reply: input.uiLang === "he" ? dis.replyHe : dis.replyEn,
      replySource: "disambiguation",
    };
  }

  let chatTopic = input.chatTopic;
  let wantsGameSearch = false;

  if (dis.kind === "route" && dis.preferLiveSearch) {
    wantsGameSearch = false;
  } else if (dis.kind === "route" && dis.preferGames && !input.blockGames) {
    wantsGameSearch = true;
  } else if (!input.blockGames) {
    wantsGameSearch = shouldOpenGamePanel(query, chatTopic) && !isTriviaOrSocialGame(query);
  }

  if (dis.kind === "route" && dis.preferLiveSearch) {
    wantsGameSearch = false;
  }

  const localTimeOnly =
    !wantsGameSearch &&
    !input.blockSearch &&
    !!input.startupContext &&
    isLocalContextTimeQuery(query);

  const shouldRunWebSearch =
    !input.blockSearch &&
    !wantsGameSearch &&
    !localTimeOnly &&
    !isSimpleGreeting(query) &&
    needsWebSearch(query);

  return {
    action: "continue",
    wantsGameSearch,
    shouldRunWebSearch,
    chatTopic,
  };
}
