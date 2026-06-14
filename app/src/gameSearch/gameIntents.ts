import type { GameCategoryId, ResolvedGameSearch } from "./types";
import { resolveGameSearch, extractDecadeRange, buildGamePanelTitle } from "./gameAliases";
import { detectCategoryFromText, isCategoryOnlyText } from "./categoryKeywords";

const GAME_SEARCH_RE =
  /(?:^|\s)(?:שחק(?:י)?|play)\s+[^\s?!.]{2,}|(?:^|\s)(?:פש\s*)?משחק(?:ים)?|נשחק|רוצה\s+לשחק|want\s+to\s+play|\bbored\b|play\s+(?:a\s+)?game|let'?s\s+play|משעמם|kill\s+time|און\s*ליין|online\s+game|game\s+search|חפש.*משחק|מצא.*משחק|search\s+(?:for\s+)?(?:the\s+)?games?|משחקים?\s*מ+?ו?לצ|מ+?ו?לצים|המלצ(?:ה|ות).*משחק|recommended\s*games?|האם\s+יש.*משחק|אילו\s+משחקים|(?:שנות|משנות).*?(?:80|90|70|שמונ|תשע)|(?:80|90)s\b|קווסט|quest|הרפתק(?:ה|ות)|תציג.*משחק|הראה.*משחק|הצג.*משחק|show\s+games|קטגור(?:יה|יית).*משחק|משחקי\s+(?:מירוצ|ארקייד|קרב|ירי|אקשן|חיד|ספורט|רטרו)/i;

const stripGameSearchPrefix = (text: string): string => {
  let t = text.trim();
  t = t.replace(/^(?:משעמם(?:\s+לי)?|i'?m\s+bored|bored)[,\s]+/i, "");
  t = t.replace(/^(?:שחק(?:י)?|play)\s+/i, "");
  t = t.replace(
    /^(?:חפש|מצא|תן|תביא|הראה|הצג|show|find|search\s+for|look\s+for|search|האם\s+יש|אילו)\s+(?:לי\s+)?(?:את\s+)?(?:ה)?(?:משחק(?:ים)?|games?)\s*(?:ש(?:ל)?|ה)?/i,
    "",
  );
  t = t.replace(/^(?:את\s+)?(?:ה)?(?:משחק(?:ים)?|games?)\s*(?:ש(?:ל)?|ה)?/i, "");
  t = t.replace(/^(?:פש\s*)?משחק(?:ים)?\s*/i, "");
  return t.trim();
};

const stripGameSearchSuffix = (text: string): string => {
  let t = text.trim();
  t = t.replace(/\s+(?:משעמם(?:\s+לי)?|i'?m\s+bored|bored)(?:\s+לי)?(?:[,\s]+(?:אני\s+רוצה\s+לשחק|i\s+want\s+to\s+play))?.*$/i, "");
  t = t.replace(/\s*[,،]?\s*אני\s+רוצה\s+לשחק\s*$/i, "");
  t = t.replace(/(?:און\s*ליין|online|בדפדפן|browser)\s*/gi, "");
  t = t.replace(
    /(?:של\s+)?(?:פלייסטיישן\s*2?|playstation\s*2?|\bps[12x]\b|פס\s*[12]|סוני|sony)\s*/gi,
    "",
  );
  t = t.replace(/\s*[,،]?\s*מ+?ו?לצים?\s*$/i, "");
  t = t.replace(/\s*(?:ו)?(?:תציג|תראה|הראה|הצג)(?:\s+(?:אות(?:ם|ן|ו)|לי))?\s*$/i, "");
  t = t.replace(/\s+יש\s*$/i, "");
  t = t.replace(/[?!.]+$/g, "").trim();
  return t;
};

const isRecommendedBrowse = (text: string): boolean =>
  /מ+?ו?לצ|recommended|האם\s+יש.*משחק/i.test(text) ||
  (/אילו\s+משחקים/i.test(text) && /מ+?ו?לצ|המלצ/i.test(text));

const NOT_ARCHIVE_GAME_RE =
  /משחק\s*(?:חשיבה|שחמט|כדורגל|מונופול|קלפים)|thinking\s+game|logic\s+game|riddle|חיד(?:ה|ות)|שחמט|chess\b|board\s+game(?!\s+search)/i;

export const isGameSearchRequest = (text: string): boolean => {
  const t = text.trim();
  if (
    NOT_ARCHIVE_GAME_RE.test(t) &&
    !/(?:חפש|מצא|ארכיון|archive|internet\s+archive|און\s*ליין|online\s+game|משחקים?\s*מ+?ו?לצ)/i.test(t)
  ) {
    return false;
  }
  return GAME_SEARCH_RE.test(t);
};

export const detectGameCategory = (text: string): GameCategoryId | null => {
  const fromLabels = detectCategoryFromText(text);
  if (fromLabels) return fromLabels;
  if (isRecommendedBrowse(text)) return "featured";
  return resolveGameSearch(text, null).category;
};

/** Full parse of a user message → Archive search params. */
export const parseGameUserRequest = (text: string): ResolvedGameSearch => {
  const category = detectGameCategory(text);
  const stripped = stripGameSearchSuffix(stripGameSearchPrefix(text));
  const resolved = resolveGameSearch(stripped || text, category);

  const decade = extractDecadeRange(text);
  if (decade) {
    resolved.yearFrom = decade.yearFrom;
    resolved.yearTo = decade.yearTo;
  }

  if (isRecommendedBrowse(text)) {
    resolved.category = "featured";
    resolved.query = "";
    resolved.browseMode = true;
  }

  if (category && isCategoryOnlyText(stripped || text, category)) {
    resolved.category = category;
    resolved.query = "";
    resolved.browseMode = true;
  }

  if (
    category &&
    /(?:משחקי|קטגור(?:יה|יית)|חפש|מצא|הראה|הצג|תציג|show|search|אילו\s+משחקים)/i.test(text)
  ) {
    if (!resolved.query || isCategoryOnlyText(resolved.query, category)) {
      resolved.category = category;
      resolved.query = "";
      resolved.browseMode = true;
    }
  }

  if (
    resolved.yearFrom != null &&
    (!resolved.query || /^(?:ארקייד|arcade|רטרו|retro|ישנ(?:ים|ים)?)$/i.test(resolved.query))
  ) {
    resolved.query = "";
    resolved.browseMode = true;
  }

  resolved.panelTitle = buildGamePanelTitle(resolved);
  return resolved;
};

/** @deprecated use parseGameUserRequest */
export const extractGameQuery = (text: string): string => parseGameUserRequest(text).query;

export const shouldOpenGamePanel = (text: string, chatTopic: string): boolean =>
  isGameSearchRequest(text) || chatTopic === "bored_play";

export const buildGameSearchPanelTitle = (resolved: ResolvedGameSearch): string =>
  resolved.panelTitle;

export { detectCategoryFromText, categoryLabelHe, formatCategoryListForPrompt } from "./categoryKeywords";
