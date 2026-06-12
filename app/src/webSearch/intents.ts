import type { SearchIntent } from "./types";

export { extractLocationPhrase } from "./queryExtract";
export { extractCountryPhrase, extractTimeZonePair, extractCurrencyPair, extractPlacePair, extractPoiNearQuery, extractNewsSite } from "./queryExtract";

/** Remove leading search verb so Wikipedia gets a clean topic. */
export const stripSearchVerb = (query: string): string =>
  query
    .trim()
    .replace(
      /^(?:חפש|חיפוש|תחפש|search\s+for|look\s+up)\s+(?:מידע\s+)?(?:על|about|info\s+on)?\s*/i,
      "",
    )
    .trim() || query.trim();

/** User explicitly asks to search (works even without Search toggle). */
export const userRequestsSearch = (text: string): boolean =>
  /(?:^|\s)(?:חפש|חיפוש|תחפש|תעשה\s+חיפוש|search\s+for|look\s+up|find\s+(?:info|information))(?:\s|$|[?!.])/i.test(
    text.trim(),
  );

const CASUAL_CHAT_RE =
  /^(?:היי|הי+|שלום|מה\s+נשמע|מה\s+שלומ(?:ך|כם|ן|ה)|how\s+are\s+you|hello|hi|hey|thanks|thank\s+you|תודה|בוקר\s+טוב|ערב\s+טוב|לילה\s+טוב|מה\s+קור(?:ה|ה)|what'?s\s+up|good\s+(?:morning|evening|night))(?:[\s!?.،,]*)*$/i;

/** Small talk / greetings — no web search. */
export const isCasualConversation = (text: string): boolean => {
  const q = text.trim();
  if (!q) return true;
  if (CASUAL_CHAT_RE.test(q)) return true;
  if (/^(?:היי|הי|שלום|hey|hi|hello)\s+מה\s+שלומ/i.test(q)) return true;
  if (/^מה\s+שלומ(?:ך|כם|ן|ה)?(?:[\s!?.،,]*)*$/i.test(q)) return true;
  if (q.length <= 24 && !/\?/.test(q) && !userRequestsSearch(q)) {
    if (/^(?:אוקיי|ok(?:ay)?|cool|nice|יפה|מעולה|בסדר|סבבה|got\s+it)\b/i.test(q)) return true;
  }
  return false;
};

/** Factual / live-data questions that benefit from external search. */
export const isFactualKnowledgeQuery = (text: string): boolean => {
  const q = text.trim();
  if (!q || isCasualConversation(q)) return false;

  if (
    /(?:מה\s+שלומ|מה\s+נשמע|how\s+are\s+you|what'?s\s+up)/i.test(q) ||
    /^(?:ספר\s+לי\s+)?(?:בדיחה|סיפור|שיר)\b/i.test(q)
  ) {
    return false;
  }

  if (
    /(?:^|\s)(?:מה|מי|איפה|מתי|למה|מדוע|כמה|איזה|אילו|what|who|where|when|why|how\s+(?:many|much|tall|high|long|big|old)|which)(?:\s|$|[?!.])/i.test(
      q,
    )
  ) {
    return true;
  }

  if (/(?:היום|עכשיו|כרגע|currently|today|now|latest|recent|אחרונ(?:ות|ים)|עדכני|מחיר|price|stock|score)/i.test(q)) {
    return true;
  }

  if (/(?:מה\s+(?:זה|הוא|היא)|מהו|מהי|explain|tell\s+me\s+about|מידע\s+על|ספר\s+לי\s+על|define|definition)/i.test(q)) {
    return true;
  }

  return false;
};

/** Live market / stock / commodity — no browser provider yet; avoid Wikipedia fallback. */
export const isMarketPriceQuery = (text: string): boolean =>
  /(?:מחיר|price|quote|שער|נסחר).*(?:מנ(?:י(?:ה|ית|ות)?|stock|זהב|gold|silver|crypto|ביטקוין|bitcoin)|NVIDIA|AAPL|TSLA|אפל\b)/i.test(
    text,
  ) || /(?:מחיר|price)\s+(?:של|of)?\s*(?:ה)?(?:זהב|gold|מנ(?:י(?:ה|ית))?)/i.test(text);

export const isRedditQuery = (text: string): boolean =>
  /(?:^|[\s\-/])r\/\w+|reddit|רדיט|סאב(?:רeddit)?/i.test(text);

export const isHackerNewsQuery = (text: string): boolean =>
  /(?:hacker\s*news|\bhn\b|ycombinator|חדשות\s*ט(?:ech|ק))/i.test(text);

export const isArxivQuery = (text: string): boolean =>
  /(?:arxiv|arXiv|מאמר(?:ים)?\s+(?:מ(?:דע|חקר)|ai|ml)|\bpapers?\b|\bresearch\b)/i.test(text);

export const isGeneralWebSearchQuery = (text: string): boolean =>
  userRequestsSearch(text) ||
  /(?:חפש\s+(?:ב)?(?:אינטרנט|web|online)|google|גוגל|search\s+the\s+web)/i.test(text);

/** Decide if this turn should hit live web providers (no manual toggle). */
export const needsWebSearch = (text: string): boolean => {
  const q = text.trim();
  if (!q || isCasualConversation(q)) return false;
  if (isMarketPriceQuery(q) || isRedditQuery(q)) return true;
  if (isHackerNewsQuery(q) || isArxivQuery(q)) return true;
  if (isGeneralWebSearchQuery(q)) return true;
  if (userRequestsSearch(q)) return true;
  if (isWeatherQuery(q) || isMarineQuery(q) || isEarthquakeQuery(q)) return true;
  if (isWorldTimeQuery(q) || isCountryQuery(q) || isHolidayQuery(q)) return true;
  if (isGovernmentQuery(q) || isCurrencyQuery(q)) return true;
  if (isDistanceQuery(q) || isPlacesQuery(q) || isNewsQuery(q)) return true;
  if (isAviationQuery(q) || isSatelliteQuery(q) || isSpaceWeatherQuery(q)) return true;
  if (isIsraelAlertsQuery(q) || isDisasterQuery(q)) return true;
  if (isGitHubQuery(q) || isHuggingFaceQuery(q)) return true;
  if (isTechQuery(q) && /(?:חפש|find|repo|github|מודל|api|dataset)/i.test(q)) return true;
  return isFactualKnowledgeQuery(q);
};

export const isWorldTimeQuery = (text: string): boolean =>
  /(?:מה\s+)?(?:ה)?שע(?:ה|ת|ות)|what\s+time|time\s+in|time\s+zone|timezone|UTC|GMT|פרש\s+שע|הפרש\s+שע|כמה\s+שעות\s+.*(?:פרש|הפרש)|שעון\s+עולמי|day\s+change|מתי\s+מתחלף\s+יום/i.test(
    text,
  ) && !isWeatherQuery(text);

export const isCountryQuery = (text: string): boolean =>
  /(?:בירה|capital|population|אוכלוסיה|מטבע|currency|דגל|flag|שפ(?:ה|ות)|language|יבשת|continent|קידומת\s+טלפון|phone\s+code|מדינ(?:ה|ות)|country|countries|restcountries)/i.test(
    text,
  );

export const isHolidayQuery = (text: string): boolean =>
  /(?:חג|חגים|holiday|holidays|public\s+holiday|bank\s+holiday|חג\s+ציבורי|האם\s+היום\s+חג)/i.test(text);

export const isGovernmentQuery = (text: string): boolean =>
  /(?:ראש\s+(?:הממשלה|ממשלה|מדינה)|נשיא|prime\s+minister|president|head\s+of\s+(?:state|government)|מפלגה\s+בשלטון|cabinet|parliament|כנסת|ממשלה\s+נוכחית)/i.test(
    text,
  );

export const isCurrencyQuery = (text: string): boolean => {
  // "מה המטבע של ברזיל" → country info, not FX rate
  if (/מטבע\s+(?:של|of)\s+/i.test(text) && !/(?:שער|יחס|המר|exchange|rate|convert|קונה|buy|\bUSD\b|\bBRL\b|\bEUR\b)/i.test(text)) {
    return false;
  }
  if (/(?:שער|exchange\s+rate|currency\s+rate|convert|המר|יחס|קונה|buy\s+with|\bBRL\b|\bJPY\b|\bCNY\b|\bCHF\b)/i.test(text)) {
    return true;
  }
  return (
    /(?:\bUSD\b|\bEUR\b|\bILS\b|\bGBP\b|\bBRL\b|דולר|יורו|שקל|ברזילאי|real)/i.test(text) &&
    /(?:ל|to|→|->|שער|rate|exchange|יחס|המר|קונה|buy|דולר|dollar)/i.test(text)
  );
};

export const isDistanceQuery = (text: string): boolean =>
  /(?:מרחק|distance|כמה\s+(?:ק"?מ|km|kilometers?)|קילומטר)/i.test(text) &&
  /(?:בין|between|מ|from|ל|to)/i.test(text);

export const isPlacesQuery = (text: string): boolean =>
  ((/(?:מצא|find|where\s+is|איפה|locate|search\s+for\s+(?:a|an|the)?)/i.test(text) &&
    /(?:ליד|near|around|by|קרוב)/i.test(text)) ||
    /(?:אילו|what|which)\s+(?:תחנ(?:ות|ת)|train\s+stations?|hospitals?|בתי\s+חולים|רכבת)/i.test(text) ||
    (/(?:תחנ(?:ות|ת)\s+רכבת|train\s+stations?|בית\s+חולים|בתי\s+חולים|תחנ(?:ת|ות)\s+דלק)/i.test(text) &&
      /(?:ליד|near|by|at|ב|של|שדה\s+תעופה|airport|הית'?רו|heathrow)/i.test(text)));

export const isNewsQuery = (text: string): boolean =>
  !isRedditQuery(text) &&
  (/(?:חדשות|news|headline|כותרת\s+ראשית|main\s+headline|breaking)/i.test(text) ||
    /(?:bbc|cnn|ynet)/i.test(text));

export const isAviationQuery = (text: string): boolean =>
  /(?:מטוס|מטוסים|aircraft|airplane|plane|adsb|opensky|תעבורה\s+אווירית|טיסות\s+מעל)/i.test(text) ||
  /(?:כמה\s+)?(?:מהם|מאלה)\s*(?:הם\s+)?(?:צבאיים|military|מסחריים)?/i.test(text);

export const isSatelliteQuery = (text: string): boolean =>
  /(?:iss|תחנת\s+חלל|לוויין|לווינים|satellite|חלל\s+בינלאומ)/i.test(text) &&
  !isAviationQuery(text);

export const isSpaceWeatherQuery = (text: string): boolean =>
  /(?:מזג\s+אוויר\s+חללי|space\s+weather|kp\s+index|רוח\s+סולארית|סערה\s+גיאומגנטית|aurora)/i.test(text);

export const isIsraelAlertsQuery = (text: string): boolean =>
  /(?:צבע\s+אדום|התרע(?:ה|ות)|פיקוד\s+העורף|oref|tzeva\s+adom|אזעק)/i.test(text);

export const isDisasterQuery = (text: string): boolean =>
  /(?:אסון|אסונות|gdacs|הוריקן|hurricane|tsunami|צונאמי|שריפ(?:ה|ות)|wildfire|סופ(?:ה|ות)\s+טרופי)/i.test(text);

export const isWeatherQuery = (text: string): boolean =>
  /מזג\s*האוויר|מז"?\s*א|טמפרטור|temperature|\btemp\b|weather|גשם|שלג|מעונן|לחות|חם\s+ב|קר\s+ב|ממש\s+ח(?:ם)?/i.test(
    text,
  );

export const isMarineQuery = (text: string): boolean =>
  /גלים|wave|סערה|הurricane|typhoon|גובה\s*גל|סופה|ים\b|ocean|marine|שיא\s*גלים/i.test(text);

export const isEarthquakeQuery = (text: string): boolean =>
  /רעיד(?:ת|ות)?\s*אדמה|רעש\s*אדמה|earthquake|seismic|tsunami|רichter|סוללת\s*רעיד/i.test(text);

export const isHuggingFaceQuery = (text: string): boolean =>
  /hugging\s*face|huggingface|hf\.co|transformers\.js/i.test(text) ||
  /(?:מודל(?:י)?|datasets?)\s*(?:ai|שפה|nlp|llm)?/i.test(text) ||
  /מודל\s+ל(?:עברית|שפה)/i.test(text);

export const isGitHubQuery = (text: string): boolean =>
  /github|גיטהב|repository|repositories|\brepo\b|open\s*source|קוד\s*פתוח/i.test(text) ||
  /פרויקט\s*(?:קוד|open)/i.test(text) ||
  /(?:פופולרי|popular|trending).*(?:github|גיטהב)/i.test(text);

export const isTechQuery = (text: string): boolean =>
  isGitHubQuery(text) ||
  isHuggingFaceQuery(text) ||
  /api|javascript|python|react|llm|machine\s*learning|neural|onnx|wasm|webgpu/i.test(text) ||
  /מצלמ[ות]?|אבטחה|dashboard|דשבורד|monitoring|ניטור/i.test(text);

export const classifySearchIntents = (query: string): SearchIntent[] => {
  const intents: SearchIntent[] = [];

  if (isMarketPriceQuery(query)) intents.push("market");
  if (isRedditQuery(query)) intents.push("reddit");
  if (isHackerNewsQuery(query)) intents.push("hackernews");
  if (isArxivQuery(query)) intents.push("arxiv");
  if (isGeneralWebSearchQuery(query)) intents.push("searx");

  if (intents.some((i) => ["market", "reddit", "hackernews", "arxiv"].includes(i))) {
    if (isGeneralWebSearchQuery(query)) intents.push("searx");
    return [...new Set(intents)];
  }

  if (isWorldTimeQuery(query)) intents.push("worldtime");
  if (isWeatherQuery(query)) intents.push("weather");
  if (isMarineQuery(query)) intents.push("marine");
  if (isEarthquakeQuery(query)) intents.push("earthquake");
  if (isCurrencyQuery(query)) intents.push("currency");
  if (isDistanceQuery(query)) intents.push("distance");
  if (isPlacesQuery(query)) intents.push("places");
  if (isNewsQuery(query)) intents.push("news");
  if (isAviationQuery(query)) intents.push("aviation");
  if (isSatelliteQuery(query)) intents.push("satellite");
  if (isSpaceWeatherQuery(query)) intents.push("spaceweather");
  if (isIsraelAlertsQuery(query)) intents.push("alerts");
  if (isDisasterQuery(query)) intents.push("disaster");
  if (isHolidayQuery(query)) intents.push("holiday");
  if (isGovernmentQuery(query)) intents.push("government");
  if (isCountryQuery(query)) intents.push("country");
  if (isGitHubQuery(query) || isTechQuery(query)) intents.push("github");
  if (isHuggingFaceQuery(query)) intents.push("huggingface");

  const structuredData = intents.some((i) =>
    [
      "worldtime", "weather", "marine", "earthquake", "currency", "holiday", "government",
      "country", "distance", "places", "news", "aviation", "satellite", "spaceweather",
      "alerts", "disaster", "market", "reddit", "hackernews", "arxiv",
    ].includes(i),
  );

  const needsWiki =
    !structuredData &&
    !intents.includes("wikipedia") &&
    !isMarketPriceQuery(query) &&
    !isRedditQuery(query) &&
    !isHackerNewsQuery(query) &&
    !isArxivQuery(query) &&
    (userRequestsSearch(query) || isFactualKnowledgeQuery(query) || needsWebSearch(query));

  if (needsWiki) {
    intents.push("wikipedia");
    if (userRequestsSearch(query) || isGeneralWebSearchQuery(query)) {
      intents.push("searx");
    }
  }

  if (!intents.length && needsWebSearch(query) && !isMarketPriceQuery(query) && !isRedditQuery(query)) {
    intents.push("wikipedia");
  }

  return [...new Set(intents)];
};

export const buildGitHubSearchQuery = (query: string): string => {
  const raw = query.trim();
  if (!raw) return "";
  const latinTokens = raw.match(/[a-zA-Z][a-zA-Z0-9_.-]{1,}/g);
  const latin = latinTokens ? latinTokens.join(" ") : "";
  if (latin.length >= 3) return latin.slice(0, 256);

  const parts: string[] = [];
  if (/מצלמ[ות]?|אבטחה|surveillance/i.test(raw)) parts.push("security camera surveillance");
  if (/ממשק|דשבורד|dashboard/i.test(raw)) parts.push("dashboard ui");
  if (/ניטור|monitoring/i.test(raw)) parts.push("monitoring");
  if (/קוד\s*פתוח|open\s*source/i.test(raw)) parts.push("open source");
  if (/מודל|llm|ai/i.test(raw)) parts.push("llm language model");
  if (/github|גיטהב/i.test(raw) && /(?:פופולרי|popular|trending|השבוע|this\s+week)/i.test(raw)) {
    return "stars:>100 pushed:>2024-01-01";
  }
  if (/github|גיטהב/i.test(raw) && parts.length) return parts.join(" ").slice(0, 256);
  if (parts.join(" ").length >= 6) return parts.join(" ").slice(0, 256);
  return "";
};

export const buildHuggingFaceSearchQuery = (query: string): string => {
  const raw = query
    .trim()
    .replace(/hugging\s*face|huggingface|hf\.co/gi, " ")
    .replace(/\bdatasets?\b/gi, " ")
    .replace(/\bmodels?\b/gi, " ")
    .trim();
  const latin = raw.match(/[a-zA-Z][a-zA-Z0-9_.-]{1,}/g)?.join(" ") ?? "";
  if (latin.length >= 2) return latin.slice(0, 128);
  if (/עברית|hebrew/i.test(query)) return "hebrew";
  if (/מודל/i.test(query)) return "text-generation";
  return raw.slice(0, 64) || "text-generation";
};
