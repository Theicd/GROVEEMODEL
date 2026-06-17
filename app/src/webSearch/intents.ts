import type { SearchIntent } from "./types";
import { expandCrossSourceIntents, isCrossSourceQuery } from "./crossSourceIntents";
import { isLocalContextTimeQuery } from "../startupContext/localTime";
import { hasUrlInQuery, isGitHubRepoUrlInQuery } from "./urlExtract";
import {
  isGeneralNewsDigestQuery,
  isIsraelNewsQuery,
  isWorldHeadlineQuery,
} from "./queryExtract";

export { extractLocationPhrase, sanitizeSearchQuery } from "./queryExtract";
export { expandCrossSourceIntents, isCrossSourceQuery } from "./crossSourceIntents";
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
  /(?:^|\s)(?:חפש|חיפוש|תחפש|תעשה\s+חיפוש|מצא|מצאי|search\s+for|look\s+up|find\s+(?:info|information))(?:\s|$|[?!.])/i.test(
    text.trim(),
  );

/**
 * Topic-specific "what's happening" — not headline news RSS.
 * e.g. "מה קורה בעולם הרובוטיקה?" → web fallback, NOT BBC RSS.
 */
export const isGeneralWebTopicQuery = (text: string): boolean => {
  const q = text.trim();
  if (!q) return false;
  if (/מה\s+קור(?:ה|ה).*(?:רובוט|robotics)/i.test(q)) return true;
  if (/what'?s\s+happening\s+(?:in\s+)?(?:the\s+)?(?:world\s+of\s+)?robotics/i.test(q)) return true;
  if (
    /מה\s+קור(?:ה|ה)\s+(?:ב)?עולם\s+\S+/i.test(q) &&
    !/^מה\s+קור(?:ה|ה)\s+(?:ב)?עולם\s*[?!.]?\s*$/i.test(q)
  ) {
    return true;
  }
  return false;
};

/** Overview / status questions without the exact «מה קורה» phrase. */
export const isTopicalOverviewQuery = (text: string): boolean => {
  const q = text.trim();
  if (!q || isBareWorldNewsQuery(q)) return false;
  if (isGeneralWebTopicQuery(q)) return true;
  if (/מה\s+(?:ה)?מצב\s+(?:ב)?תחום/i.test(q)) return true;
  if (/מה\s+(?:ה)?מצב\s+(?:ב)?(?:מערכת|שוק|תעשי)/i.test(q)) return true;
  if (/מה\s+(?:ה)?מצב\s+(?:ב)?עולם\s+\S+/i.test(q)) return true;
  if (/מה\s+חדש\s+(?:ב|בעולם|עם|לגבי)/i.test(q)) return true;
  if (/עדכונ(?:ים|י)\s+(?:אחרונ|על|ב)/i.test(q)) return true;
  if (/what'?s\s+new\s+(?:in|with|on)/i.test(q)) return true;
  if (/latest\s+(?:news|developments|trends|updates)\s+(?:in|on|about|with)/i.test(q)) return true;
  if (/what'?s\s+happening\s+(?:in|with|on)/i.test(q) && !/בעולם\s*[?!.]?\s*$/i.test(q)) return true;
  return false;
};

/** Wording that implies fresh / live info — no magic «מה קורה». */
export const hasTimelyInfoSignal = (text: string): boolean =>
  hasLiveTemporalSignal(text) ||
  /(?:מגמ(?:ה|ות)|trends?|עדכונ(?:ים|י)|developments?|שינוי(?:י)?ם|מצב\s+(?:ה)?(?:שוק|תעשי|תחום|תחומ)|status\s+of|מה\s+חדש|what'?s\s+new|breaking|אחרונ(?:ות|ים)|כרגע|עכשיו|היום|currently|today|now|latest|recent)/i.test(
    text,
  );

/**
 * Factual question that likely needs external fetch — auto search without «חפש» / «מה קורה».
 * Skips when a structured API intent already handles it (weather, ships, …).
 */
export const isTimelyOverviewQuery = (text: string): boolean => {
  const q = text.trim();
  if (!q || isCasualConversation(q) || isStaticFactualQuery(q)) return false;
  if (isTopicalOverviewQuery(q) || isBareWorldNewsQuery(q)) return true;
  const intents = classifySearchIntents(q);
  const structured = intents.filter(
    (i) => !["news", "wikipedia", "hackernews"].includes(i),
  );
  if (structured.length) return false;
  return hasTimelyInfoSignal(q) && isFactualKnowledgeQuery(q);
};

/** «מה קורה בעולם?» / «מה המצב בעולם?» — headline RSS (no extra topic word). */
export const isBareWorldNewsQuery = (text: string): boolean => {
  const q = text.trim();
  if (isGeneralWebTopicQuery(q)) return false;
  return (
    /^מה\s+חדש\s+בעולם\s*[?!.]?\s*$/i.test(q) ||
    /^מה\s+קור(?:ה|ה)\s+(?:ב)?(?:עולם|ישראל)\s*[?!.]?\s*$/i.test(q) ||
    /^מה\s+קור(?:ה|ה)\s+עכשיו\s+(?:ב)?(?:עולם|ישראל)\s*[?!.]?\s*$/i.test(q) ||
    /^מה\s+(?:ה)?מצב\s+(?:ב)?(?:עולם|ישראל)\s*[?!.]?\s*$/i.test(q) ||
    /^מה\s+(?:ה)?מצב\s+עכשיו\s+(?:ב)?(?:עולם|ישראל)\s*[?!.]?\s*$/i.test(q) ||
    (/^מה\s+קור(?:ה|ה)\s+(?:ב)?(?:עולם|ישראל)/i.test(q) &&
      /(?:current\s+events|אירועים\s+אחרונים)/i.test(q))
  );
};

/** World / topical overview — enrich with RSS headlines + SearXNG web snippets in parallel. */
export const isOverviewBlendQuery = (text: string): boolean => isBareWorldNewsQuery(text);

const CASUAL_CHAT_RE =
  /^(?:היי|הי+|שלום|מה\s+נשמע|מה\s+שלומ(?:ך|כם|ן|ה)|how\s+are\s+you|hello|hi|hey|thanks|thank\s+you|תודה|בוקר\s+טוב|ערב\s+טוב|לילה\s+טוב|מה\s+קור(?:ה|ה)|what'?s\s+up|good\s+(?:morning|evening|night))(?:[\s!?.،,]*)*$/i;

/** Small talk / greetings — no web search. */
export const isCasualConversation = (text: string): boolean => {
  const q = text.trim();
  if (!q) return true;
  if (CASUAL_CHAT_RE.test(q)) return true;
  if (/^(?:היי|הי|שלום|hey|hi|hello)\s+מה\s+שלומ/i.test(q)) return true;
  if (/^מה\s+שלומ(?:ך|כם|ן|ה)?(?:[\s!?.،,]*)*$/i.test(q)) return true;
  if (/בינה\s+(?:ה)?מלאכותית|artificial\s+intelligence|\bai\b|machine\s+learning/i.test(q)) {
    return false;
  }
  if (q.length <= 24 && !/\?/.test(q) && !userRequestsSearch(q)) {
    if (/^(?:אוקיי|ok(?:ay)?|cool|nice|יפה|מעולה|בסדר|סבבה|got\s+it)\b/i.test(q)) return true;
  }
  return false;
};

/** Time-sensitive wording — auto web search is for live data, not encyclopedic facts. */
export const hasLiveTemporalSignal = (text: string): boolean =>
  /(?:עכשיו|כרגע|היום|השבוע|ב-?24|אחרונ(?:ות|ים)|עדכני|currently|today|now|latest|recent|live|real.?time|מחיר|price|שער|exchange|headline|כותרת|צפוי|forecast|מעל|above|מעל\s+ישראל)/i.test(
    text,
  );

/** Static geography / country facts the model already knows — no auto search. */
export const isStaticFactualQuery = (text: string): boolean => {
  const q = text.trim();
  if (!q) return false;
  if (isCountryQuery(q) && !hasLiveTemporalSignal(q)) return true;
  if (isDistanceQuery(q) && !/(?:מצא|find|חפש|search)/i.test(q)) return true;
  if (/^(?:מה\s+(?:ה)?(?:ביר(?:ה|ת)|מטבע)|what\s+(?:is\s+)?(?:the\s+)?(?:capital|currency|population))/i.test(q)) {
    return true;
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

/** Live commodity prices (oil, gold, metals). */
export const isCommodityPriceQuery = (text: string): boolean =>
  /(?:מחיר|price|quote|שער).*(?:נפט|oil|brent|wti|crude|petroleum|חבית|barrel|זהב|gold|silver|xau)/i.test(text) ||
  /(?:חבית|barrel|אונקי|ounce).*(?:נפט|brent|oil|זהב|gold)/i.test(text) ||
  /(?:brent|wti|crude\s+oil|gold\s+price|xauusd|xau)/i.test(text);

/** Live stock / equity / index prices. */
export const isMarketPriceQuery = (text: string): boolean => {
  if (isCommodityPriceQuery(text)) return false;
  return (
    /(?:מחיר|price|quote|שער|נסחר|מצב|איך\s+עומד).*(?:מנ(?:י(?:ה|ית|ות)?|stock)|NVIDIA|AAPL|TSLA|NVDA|אפל\b)/i.test(text) ||
    /(?:מחיר|price)\s+(?:של|of)?\s*(?:ה)?(?:מנ(?:י(?:ה|ית))?)/i.test(text) ||
    /(?:מדד|index|s&p|sp\s*500|sp500|nasdaq|dow\s*jones|ta-?\d+|dax|ftse|nikkei)/i.test(text) &&
      /(?:מחיר|price|מצב|quote|שער|נסחר|עולה|יורד|כמה|עכשיו|כרגע|היום|ערך)/i.test(text) ||
    /(?:מה\s+)?(?:מצב|ערך)\s+(?:של\s+)?(?:מדד\s+)?(?:s&p|sp\s*500|nasdaq|dow)/i.test(text)
  );
};

export const isHuggingFaceImageQuery = (text: string): boolean =>
  /(?:מודel(?:י|ים)?|מודל(?:י|ים)?).{0,12}תמונה|תמונה.{0,12}(?:מודel|מודל)|image\s+model|text-?to-?image|stable\s*diffusion|sdxl|flux|diffusion|יצירת\s+תמונה|דיפוז|תמונה\s+פופולר/i.test(
    text,
  );

export const isYouTubeQuery = (text: string): boolean =>
  /(?:youtube|יוטיוב)/i.test(text) &&
  /(?:פופולר|popular|trending|best|top|סרטון|video|videos)/i.test(text);

export const isRedditQuery = (text: string): boolean =>
  /(?:^|[\s\-/])r\/\w+|reddit|רדיט|סאב(?:רeddit)?/i.test(text);

export const isCryptoQuery = (text: string): boolean =>
  /(?:ביטקוין|bitcoin|btc|ethereum|eth|קריפטו|crypto|מטבע(?:ות)?\s+דיגיטל)/i.test(text) &&
  /(?:מחיר|price|על(?:ה|ו)|24\s*שע)/i.test(text);

export const isHackerNewsQuery = (text: string): boolean =>
  /(?:hacker\s*news|hn\b|ycombinator)/i.test(text) ||
  (/פופולרי|popular|top|best|כותרת|headline|פוסט/i.test(text) &&
    /(?:hacker\s*news|hn\b|ycombinator)/i.test(text)) ||
  (/openai|trending|artificial\s+intelligence|\bai\b|machine\s+learning|בינה\s+(?:ה)?מלאכותית/i.test(
    text,
  ) &&
    !/github|גיטהב|repository|\brepo\b|פרויקט/i.test(text) &&
    /(?:חדשות|news|headline|קורה|עדכנ|latest|כרגע|היום)/i.test(text));

export const isTechNewsQuery = (text: string): boolean =>
  /(?:מה\s+קור(?:ה|ה)|what'?s\s+happening|latest|עדכנ|כרגע|היום|now)/i.test(text) &&
  /(?:בינה\s+(?:ה)?מלאכותית|artificial\s+intelligence|\bai\b|machine\s+learning|טכנולוג|tech|openai|llm)/i.test(
    text,
  );

export const isFlightStatusQuery = (text: string): boolean =>
  /(?:flight\s+status|סטטוס\s+טיס|departures?|arrivals?|delays?|עיכוב)/i.test(text) ||
  (/(?:נמל\s+התעופה|airport|JFK|LAX|Heathrow|CDG|TLV|Ben\s+Gurion)/i.test(text) &&
    /(?:flight|טיס|departure|arrival|status|מצב|עיכוב)/i.test(text));

/** Static marine infrastructure — buoys, lighthouses, harbours (Overpass OSM). */
export const isMarineInfraQuery = (text: string): boolean => {
  const q = text.trim();
  if (!q) return false;
  const infra =
    /(?:מצופ|מגדלור|מגדל\s+אור|buoy|lighthouse|seamark|רציף|pier|wharf|breakwater|harbour|harbor)/i.test(q) ||
    (/(?:כמה|אילו|what|which|how\s+many)/i.test(q) &&
      /(?:מצופ|מגדלור|buoy|lighthouse|harbour|harbor|נמל(?:ים)?)/i.test(q) &&
      !/(?:ספינ|אונi|ship|vessel|שייט|ais)/i.test(q));
  return infra;
};

export const isShipsQuery = (text: string): boolean => {
  if (isMarineInfraQuery(text) && !/(?:ספינ|אונi|ship|vessel|שייט|ais|כלי\s+שייט)/i.test(text)) {
    return false;
  }
  return (
    /(?:ספינ|אוני(?:יות|ה)|אוניות|כלי\s+שייט|כלי\s+ימי|vessel|ships?\b|ais\b|תעלת\s+סואץ|suez\s+canal|מכלית|tanker|מפרץ\s+הפרס|persian\s+gulf)/i.test(
      text,
    ) ||
    (/(?:אילו|what|which|כמה).*(?:ספינ|אונi|ship|vessel|שייט|מכלית|tanker)/i.test(text) &&
      /(?:סואץ|suez|ים|sea|port|נמל|canal|תעלה|חיפה|haifa|מפרץ|רוטרדם|rotterdam|יוון|greece|פרס|persian)/i.test(text)) ||
    /(?:אוניות|ספינות|כלי\s+שייט|ships?|vessels?)\s+(?:ליד|ב|סביב|near|around)\s+/i.test(text)
  );
};

export const isSpaceXQuery = (text: string): boolean =>
  /spacex|space\s*x|סpace\s*x|ספייס\s*אקס|ספייס-?אקס/i.test(text) ||
  (/(?:שיגור|launch|rocket|טיל)/i.test(text) && /spacex|space\s*x|ספייס/i.test(text));

export const isIssQuery = (text: string): boolean =>
  /\biss\b|תחנת\s+(?:ה)?חלל|space\s+station|החלל\s+הבינלאומ|מסלול\s+(?:ה)?-?iss/i.test(text) ||
  /(?:תעבור|יעבור|pass(?:es)?).*(?:מעל|above)\s+(?:ישראל|israel)/i.test(text) ||
  /(?:מעל|above)\s+(?:ישראל|israel)/i.test(text) && /(?:iss|חלל|תחנת|לוויין|space\s+station)/i.test(text);

export const isSatelliteCatalogQuery = (text: string): boolean =>
  (/(?:כמה|how\s+many|מספר|number\s+of).*(?:לוויין|satellite)/i.test(text) ||
    /(?:לוויינים|satellites)\s+(?:פעיל|active|במסלול|in\s+orbit)/i.test(text)) &&
  !isIssQuery(text) &&
  !isStarlinkCountQuery(text);

/** Global Starlink catalog count (CelesTrak GROUP=starlink) — not regional tracking. */
export const isStarlinkCountQuery = (text: string): boolean =>
  /starlink/i.test(text) &&
  (/(?:כמה|how\s+many|מספר|number\s+of)/i.test(text) ||
    /(?:לוויינ|satellite).*(?:פעיל|active|במסלול|in\s+orbit)/i.test(text) ||
    /starlink.*(?:פעיל|active|כמה|how\s+many|במסלול)/i.test(text));

/** Starlink above a region / list by area — not supported in-browser. */
export const isStarlinkRegionalQuery = (text: string): boolean =>
  /starlink/i.test(text) &&
  !isStarlinkCountQuery(text) &&
  (/(?:מעל|above|over|באזור|ליד|near)/i.test(text) ||
    /(?:אילו|which|list)\s/i.test(text));

/**
 * Decide if this turn should hit live web providers (no manual toggle).
 * Static facts (capital, currency name, distance) → model knowledge only.
 * Auto search = live / time-sensitive data or explicit lookup (POI, "חפש").
 */
export const needsWebSearch = (text: string): boolean => {
  const q = text.trim();
  if (!q || isCasualConversation(q)) return false;
  if (hasUrlInQuery(q)) return true;
  if (userRequestsSearch(q)) return true;
  if (isWorldOverviewQuery(q)) return true;
  if (isCrossSourceQuery(q)) return true;
  if (/\bawacs\b/i.test(q)) return true;

  if (isGovernmentQuery(q)) return true;
  if (isCommodityPriceQuery(q) || isMarketPriceQuery(q) || isRedditQuery(q)) return true;
  if (isYouTubeQuery(q)) return true;
  if (isStaticFactualQuery(q)) return false;
  if (isGeneralWebTopicQuery(q)) return true;
  if (isTopicalOverviewQuery(q)) return true;
  if (isTimelyOverviewQuery(q)) return true;
  if (isCryptoQuery(q)) return true;
  if (isIsraelAlertsQuery(q) || isDisasterQuery(q)) return true;
  if (isNewsQuery(q)) return true;
  if (isWorldTimeQuery(q)) {
    if (isLocalContextTimeQuery(q)) return false;
    return true;
  }
  if (isWeatherQuery(q) || isMarineQuery(q) || isAirQualityQuery(q)) return true;
  if (isArxivQuery(q)) return true;
  if (isCurrencyQuery(q)) return true;
  if (isFlightStatusQuery(q)) return true;
  if (isShipsQuery(q) || isMarineInfraQuery(q)) return true;
  if (isSpaceXQuery(q)) return true;
  if (isPlacesQuery(q)) return true;
  if (isSatelliteQuery(q) || isSpaceWeatherQuery(q)) return true;
  if (isHackerNewsQuery(q) || isTechNewsQuery(q)) return true;
  if (isEarthquakeQuery(q)) return true;
  if (isAviationQuery(q)) return true;
  if (isHolidayQuery(q)) {
    return /(?:היום|האם\s+היום|today|now)/i.test(q);
  }
  if (isGitHubQuery(q) || isHuggingFaceQuery(q)) {
    return /(?:פופולרי|popular|trending|השבוע|release|גרסה|חפש|search|find|מצא|היום|מודל|פרויקט)/i.test(q);
  }
  if (isTechQuery(q) && /(?:חפש|find|repo|github|מודל|api|dataset)/i.test(q)) return true;

  return false;
};

export const isWorldTimeQuery = (text: string): boolean => {
  if (isEarthquakeQuery(text) || isWeatherQuery(text)) return false;
  const asksClock =
    /(?:מה\s+)?(?:ה)?שע(?:ה|ת)(?=\s|$|[?!.,])|what\s+time|time\s+in|time\s+zone|timezone|UTC|GMT|פרש\s+שע|הפרש\s+שע|כמה\s+שעות\s+.*(?:פרש|הפרש)|שעון\s+עולמי|day\s+change|מתי\s+מתחלף\s+יום|(?:מה\s+)?(?:ה)?תאריך\s+(?:ב|ב־|in)|what(?:'s|\s+is)?\s+the\s+date/i.test(
      text,
    );
  if (/\d+\s*(?:שעות|hours)(?=\s|$|[?!.,])/i.test(text) && !asksClock) {
    return false;
  }
  return asksClock && !isWeatherQuery(text);
};

export const isCountryQuery = (text: string): boolean =>
  /(?:ביר(?:ה|ת)|capital|population|אוכלוסי(?:ה|יה)|תושבים|מטבע|currency|דגל|flag|שפ(?:ה|ות)|language|יבשת|continent|קידומת\s+טלפון|phone\s+code|מדינ(?:ה|ות)|country|countries|restcountries)/i.test(
    text,
  );

export const isHolidayQuery = (text: string): boolean =>
  /(?:חג|חגים|holiday|holidays|public\s+holiday|bank\s+holiday|חג\s+ציבורי|האם\s+היום\s+חג)/i.test(text);

export const isGovernmentQuery = (text: string): boolean =>
  /(?:ראש\s+(?:ה)?(?:ממשלה|ממשלת|מדינה)|נשיא|prime\s+minister|president|head\s+of\s+(?:state|government)|מפלגה\s+בשלטון|cabinet|parliament|כנסת|ממשלה\s+נוכחית)/i.test(
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
    (/\b(?:hotel|restaurant|hospital|station|pharmacy|מלון|מסעדה|בית\s+חולים|תחנ(?:ת|ות)\s+רכבת)\b/i.test(
      text,
    ) &&
      /\b(?:near|around|by|ליד|קרוב)\b/i.test(text)) ||
    (/\b(?:nearest|closest|הכי\s+קרוב(?:ה|ים)?|הקרוב(?:ה|ים)?\s+ביותר)\b/i.test(text) &&
      /(?:station|רכבת|train|תחנ|airport|שדה\s+תעופה|tower|מגדל|louvre|לובר|eiffel|אייפ|ברלין|berlin|BER)/i.test(
        text,
      )) ||
    /(?:מה|what|איזו|which)\s+.*(?:תחנ(?:ות|ת)\s+רכבת|train\s+station).*(?:הקרוב|nearest|closest)/i.test(
      text,
    ) ||
    /(?:אילו|what|which)\s+(?:תחנ(?:ות|ת)|train\s+stations?|hospitals?|בתי\s+חולים|רכבת)/i.test(text) ||
    (/(?:תחנ(?:ות|ת)\s+רכבת|train\s+stations?|בית\s+חולים|בתי\s+חולים|תחנ(?:ת|ות)\s+דלק)/i.test(text) &&
      /(?:ליד|near|by|at|ב|של|שדה\s+תעופה|airport|הית'?רו|heathrow|eiffel|אייפ|louvre|לובר)/i.test(text)));

export const isNewsQuery = (text: string): boolean => {
  if (isRedditQuery(text)) return false;
  if (isIsraelNewsQuery(text) || isWorldHeadlineQuery(text) || isGeneralNewsDigestQuery(text)) {
    return true;
  }
  if (
    /(?:חדשות|headline|(?:ה)?כותר(?:ת|ות)(?:\s+(?:ה)?ראשי(?:ת|ות|יה))?|main\s+headline|breaking|כתבות?|דיווחים?|מובילות|עיקר)/i.test(
      text,
    )
  ) {
    return true;
  }
  if (/\bnews\b/i.test(text) && !/hacker\s*news/i.test(text)) return true;
  if (isHackerNewsQuery(text)) return false;
  if (isGeneralWebTopicQuery(text)) return false;
  return (
    /(?:כותרות(?:\s+ה)?\s*חשובות|headlines)/i.test(text) ||
    /(?:חפש|מצא|תביא|עוד|search|find).*(?:כתבות?|articles?|reports?)/i.test(text) ||
    /(?:מה\s+קור(?:ה|ה)|what'?s\s+happening).*(?:ישראל|israel|עולם|world)/i.test(text) ||
    /(?:מצב\s+(?:ב)?(?:עולם|ישראל)|current\s+events|אירועים\s+אחרונים)/i.test(text) ||
    /(?:bbc|cnn|ynet)/i.test(text)
  );
};

export const isAviationQuery = (text: string): boolean =>
  !isFlightStatusQuery(text) &&
  (/(?:מטוס|מטוסים|aircraft|airplane|plane|adsb|opensky|תעבורה\s+(?:ה)?אווירית|תנועה\s+(?:ה)?אווירית|טיסות\s+מעל)/i.test(text) ||
    /\bawacs\b/i.test(text) ||
    /(?:עומס\s+(?:ב)?(?:שמי|האוויר)|air\s+traffic|traffic\s+above|שמי\s+ישראל|israeli\s+airspace)/i.test(text) ||
    (/(?:בעולם|worldwide|global|ברחבי\s+העולם|around\s+the\s+world|in\s+the\s+air)/i.test(text) &&
      /(?:מטוס|aircraft|plane|adsb|תעופה|air|טיס|עומס|traffic)/i.test(text)) ||
    (/(?:מעל|above|over)\s+(?:ישראל|israel|לונדון|london)/i.test(text) &&
      /(?:מטוס|aircraft|plane|adsb|תעופה|עומס|traffic|כמה)/i.test(text)) ||
    /(?:כמה\s+)?(?:מהם|מאלה)\s*(?:הם\s+)?(?:צבאיים|military|מסחריים)(?:\s|$|[?!.])/i.test(text));

export const isSatelliteQuery = (text: string): boolean =>
  (isIssQuery(text) || isSatelliteCatalogQuery(text) || isStarlinkCountQuery(text)) &&
  !isAviationQuery(text);

export const isSpaceWeatherQuery = (text: string): boolean =>
  /(?:מזג\s+אוויר\s+חללי|space\s+weather|kp\s+index|רוח\s+סולארית|סערה\s+גיאומגנטית|aurora)/i.test(text);

export const isWorldOverviewQuery = (text: string): boolean =>
  /(?:תמונת\s+מצב|סקיר(?:ה|ת)\s+(?:של\s+)?(?:מצב\s+)?(?:ה)?עולם|מצב\s+העולם|מה\s+הדברים\s+המעניינים|משהו\s+חריג|אירועים\s+חשובים|20\s+האירועים|סכם.*התראות|מקומות\s+הפעילים|מה\s+קור(?:ה|ה)\s+עכשיו\s+ב(?:חלל|אוקיינוס|ים))/i.test(
    text,
  ) ||
  /(?:תן\s+לי\s+)?סקירה\s+של\s+מצב\s+העולם/i.test(text) ||
  /(?:ה)?24\s+שעות\s+האחרונות/i.test(text) ||
  /מה\s+קור(?:ה|ה)\s+עכשיו\s+ב(?:חלל|אוקיינוס|ים|שמי)/i.test(text) ||
  /מה\s+קור(?:ה|ה)\s+כרגע\s+ב(?:אזור\s+)?(?:הכי\s+)?עמוס/i.test(text) ||
  /איזה\s+אזור\s+בעולם\s+נראה\s+הכי\s+פעיל/i.test(text) ||
  /(?:כמה\s+)?אירועים\s+משמעותיים\s+במקביל/i.test(text);

export const isUnsupportedLiveQuery = (text: string): boolean =>
  /(?:שדה\s+התעופה|airport).*(?:העמוס|busiest)/i.test(text) ||
  /(?:נמל\s+העמוס|busiest\s+port)/i.test(text) ||
  isStarlinkRegionalQuery(text) ||
  /(?:קווי\s+רכבet|train\s+lines?\s+to)/i.test(text);

export const isIsraelAlertsQuery = (text: string): boolean =>
  /(?:צבע\s+אדום|התרע(?:ה|ות)|פיקוד\s+העורף|oref|tzeva\s+adom|אזעק)/i.test(text);

export const isDisasterQuery = (text: string): boolean =>
  /(?:אסון|אסונות|gdacs|הוריקן|hurricane|tsunami|צונאמי|שריפ(?:ה|ות)|wildfire|סופ(?:ה|ות)\s+טרופי)/i.test(text);

export const isWeatherQuery = (text: string): boolean =>
  /מזג\s*האוויר|מסג\s*האוויר|מז"?\s*א|טמפרטור|temperatur|weather|temperature|גשם|שלג|מעונן|לחות|מזג|מהירות\s+(?:ה)?רוח|wind\s+speed/i.test(text);

/** Live air pollution — Open-Meteo Air Quality API. */
export const isAirQualityQuery = (text: string): boolean =>
  /(?:איכות\s+(?:ה)?אוויר|air\s+quality|זיהום\s+אוויר|pm2\.?5|pm10|\bus_aqi\b|\baqi\b)/i.test(text);

/** Academic preprints — arXiv Atom API. */
export const isArxivQuery = (text: string): boolean =>
  /\barxiv\b|ארxiv/i.test(text) ||
  (/(?:מאמר(?:י|ים)?|paper|papers|preprint|publication)/i.test(text) &&
    /(?:חפש|find|search|על|about|בנושא|on)/i.test(text)) ||
  /(?:מאמר(?:י|ים)?\s+(?:על|about|בנושא)|scientific\s+paper)/i.test(text);

export const isMarineQuery = (text: string): boolean =>
  !isShipsQuery(text) &&
  /גלים|wave|סערה|הurricane|typhoon|גובה\s*גל|סופה|שיא\s*גלים|marine\s+weather|ocean\s+wave/i.test(text);

export const isEarthquakeQuery = (text: string): boolean =>
  /רעיד(?:ת|ות)?\s*(?:ה)?אדמה|רעש\s*אדמה|earthquake|seismic|tsunami|רichter|סוללת\s*רעיד/i.test(text);

export const isHuggingFaceQuery = (text: string): boolean =>
  /hugging\s*face|huggingface|hf\.co|transformers\.js/i.test(text) ||
  /(?:מודל(?:י)?|datasets?)\s*(?:ai|שפה|nlp|llm)?/i.test(text) ||
  /מודל\s+ל(?:עברית|שפה|ocr|OCR|תמלול)/i.test(text) ||
  /(?:חפש|find|search).*(?:מודל|model)/i.test(text) ||
  /\bocr\b|זיהוי\s+טקסט|text\s+recognition|tesseract|paddleocr/i.test(text);

export const isGitHubQuery = (text: string): boolean =>
  /github|גיטהב|repository|repositories|\brepo\b|open\s*source|קוד\s*פתוח/i.test(text) ||
  /פרויקט\s*(?:קוד|open)/i.test(text) ||
  /(?:פופולרי|popular|trending).*(?:github|גיטהב|פרויקט)/i.test(text) ||
  /(?:פרויקט(?:ים)?).*(?:פופולרי|popular|trending|היום|השבוע)/i.test(text) ||
  (/^מצא\s+/i.test(text) &&
    /(?:פרויקט|webgpu|three|ollama|ai\b|github|גיטהב|משחק)/i.test(text));

export const isTechQuery = (text: string): boolean =>
  isGitHubQuery(text) ||
  isHuggingFaceQuery(text) ||
  /api|javascript|python|react|llm|machine\s*learning|neural|onnx|wasm|webgpu/i.test(text) ||
  /מצלמ[ות]?|אבטחה|dashboard|דשבורד|monitoring|ניטור/i.test(text);

export const classifySearchIntents = (query: string): SearchIntent[] => {
  if (hasUrlInQuery(query)) {
    return ["link"];
  }
  if (isGeneralWebTopicQuery(query)) {
    return [];
  }
  if (isBareWorldNewsQuery(query)) {
    return ["news"];
  }
  if (isTopicalOverviewQuery(query)) {
    if (isNewsQuery(query)) return ["news"];
    return [];
  }
  const intents: SearchIntent[] = [];
  if (isWorldOverviewQuery(query)) {
    return [
      ...new Set([
        "disaster",
        "earthquake",
        "aviation",
        "ships",
        "news",
        "hackernews",
        "weather",
        "marine",
        "satellite",
        "alerts",
      ] as SearchIntent[]),
    ];
  }
  if (isRedditQuery(query)) return intents;
  if (isCommodityPriceQuery(query)) {
    intents.push("commodity");
    return intents;
  }
  if (isMarketPriceQuery(query)) {
    intents.push("market");
    return intents;
  }
  if (isCryptoQuery(query)) intents.push("crypto");
  if (isCommodityPriceQuery(query)) intents.push("commodity");
  const explicitNews = /(?:חדשות|headline|כותרת|כותרות|breaking|כתבות?|דיווח)/i.test(query);
  const pureNewsQuery =
    isNewsQuery(query) && !isTopicalOverviewQuery(query) && !isGeneralWebTopicQuery(query);
  if (!pureNewsQuery) {
    if (isTechNewsQuery(query) && !explicitNews) intents.push("hackernews");
    else if (isHackerNewsQuery(query)) intents.push("hackernews");
  }
  if (isYouTubeQuery(query)) intents.push("youtube");
  if (isWorldTimeQuery(query)) intents.push("worldtime");
  if (isWeatherQuery(query)) intents.push("weather");
  if (isAirQualityQuery(query)) intents.push("airquality");
  if (isArxivQuery(query)) intents.push("arxiv");
  if (isMarineQuery(query)) intents.push("marine");
  if (isMarineInfraQuery(query)) intents.push("marine-infra");
  if (isShipsQuery(query)) intents.push("ships");
  if (isEarthquakeQuery(query)) intents.push("earthquake");
  if (isCurrencyQuery(query)) intents.push("currency");
  if (isDistanceQuery(query)) intents.push("distance");
  if (isPlacesQuery(query)) intents.push("places");
  if (isNewsQuery(query) && (!isTechNewsQuery(query) || explicitNews)) intents.push("news");
  if (isAviationQuery(query)) intents.push("aviation");
  if (isStarlinkCountQuery(query)) intents.push("satellite");
  else if (isSatelliteCatalogQuery(query)) intents.push("satellite");
  else if (isIssQuery(query)) intents.push("satellite");
  if (isSpaceXQuery(query)) intents.push("spacex");
  if (isSpaceWeatherQuery(query)) intents.push("spaceweather");
  if (isIsraelAlertsQuery(query)) intents.push("alerts");
  if (isDisasterQuery(query)) intents.push("disaster");
  if (isHolidayQuery(query)) intents.push("holiday");
  if (isGovernmentQuery(query)) intents.push("government");
  if (isCountryQuery(query)) intents.push("country");
  const hf = isHuggingFaceQuery(query);
  const explicitSearch = userRequestsSearch(query);
  if (
    !pureNewsQuery &&
    !isGitHubRepoUrlInQuery(query) &&
    (isGitHubQuery(query) || (isTechQuery(query) && !hf) || (explicitSearch && hf))
  ) {
    intents.push("github");
  }
  if (hf) intents.push("huggingface");

  // Only explicit GitHub/HF questions skip Wikipedia; generic tech questions
  // (e.g. "מה קורה עם React") still benefit from encyclopedic context.
  const techExclusive = isGitHubQuery(query) || hf;
  const structured = intents.some((i) =>
    [
      "worldtime", "weather", "airquality", "arxiv", "marine", "earthquake", "currency", "holiday", "government",
      "country", "distance", "places", "news", "aviation", "satellite", "spaceweather",
      "alerts", "disaster", "crypto", "commodity", "market", "hackernews", "ships", "spacex", "link",
    ].includes(i),
  );

  // Wikipedia only when the user explicitly asks to search — never as silent fallback.
  if (
    userRequestsSearch(query) &&
    !structured &&
    !techExclusive &&
    !isNewsQuery(query) &&
    !isMarketPriceQuery(query) &&
    !isRedditQuery(query) &&
    !isStaticFactualQuery(query)
  ) {
    intents.push("wikipedia");
  }

  if (isCrossSourceQuery(query)) {
    if (/^כמה\s+/i.test(query) && (isAviationQuery(query) || isShipsQuery(query))) {
      return [...new Set(intents)];
    }
    return expandCrossSourceIntents(query, intents);
  }

  return [...new Set(intents)];
};

export const isGitHubPopularQuery = (text: string): boolean =>
  /(?:פרויקט(?:ים)?|project|repo).*(?:פופולרי|popular|trending|היום|today|השבוע|this\s+week|כרגע|now)/i.test(
    text,
  ) ||
  /(?:פופולרי|popular|trending|היום|today|השבוע|this\s+week|כרגע|now).*(?:פרויקט|project|repo|github|גיטהב)/i.test(
    text,
  ) ||
  /(?:מהו|what\s+is).*(?:פופולרי|popular|trending).*(?:github|גיטהב|פרויקט)/i.test(text);

/** GitHub Search API query for trending / popular repo questions. */
export const buildGitHubPopularSearchQuery = (text: string): string => {
  const daysBack = /היום|today|כרגע|now/i.test(text) ? 7 : 30;
  const since = new Date(Date.now() - daysBack * 86_400_000).toISOString().slice(0, 10);
  return `stars:>500 pushed:>${since} archived:false fork:false`;
};

export const buildGitHubSearchQuery = (query: string): string => {
  const raw = query.trim();
  if (!raw) return "";

  if (isGitHubPopularQuery(raw)) {
    return buildGitHubPopularSearchQuery(raw);
  }

  if (/ocr|זיהוי\s+טקסט|text\s+recognition|tesseract|paddleocr/i.test(raw)) {
    return "ocr text recognition tesseract paddleocr";
  }
  const latinTokens = raw.match(/[a-zA-Z][a-zA-Z0-9_.-]{1,}/g);
  const latin = latinTokens ? latinTokens.join(" ") : "";
  const latinNoiseOnly = /^(?:github|gitlab|repo|repos|repository|repositories)$/i.test(latin.trim());
  if (latin.length >= 3 && !latinNoiseOnly) return latin.slice(0, 256);

  if (/(?:פרויקט(?:ים)?).*(?:פופולרי|popular|trending|היום)/i.test(raw)) {
    return buildGitHubPopularSearchQuery(raw);
  }
  if (/ollama|חלופ/i.test(raw)) return "ollama alternative local llm";
  if (/three\.?js|threejs/i.test(raw)) return "three.js game";
  if (/webgpu/i.test(raw)) return "webgpu";
  if (/(?:AI|בינה\s+מלאכותית).*(?:שבוע|week)/i.test(raw)) return "AI stars:>50 pushed:>2025-01-01";
  if (/^מצא\s+/i.test(raw) && /פרויקט/i.test(raw)) {
    const topic = raw.replace(/^מצא\s+(?:פרויקט(?:ים)?\s*)?(?:בנושא\s+|ש(?:ל|ה)?\s*)?/i, "").trim();
    if (topic.length >= 2) return topic.slice(0, 80);
  }
  const parts: string[] = [];
  if (/גיימינג|gaming|video\s*game|esports|משחק/i.test(raw)) parts.push("gaming video game");
  if (/רובוט|robotics/i.test(raw)) parts.push("robotics");
  if (/מצלמ[ות]?|אבטחה|surveillance/i.test(raw)) parts.push("security camera surveillance");
  if (/ניטור|monitoring/i.test(raw)) parts.push("monitoring");
  if (/קוד\s*פתוח|open\s*source/i.test(raw)) parts.push("open source");
  if (/מודל|llm|ai/i.test(raw)) parts.push("llm language model");
  if (/github|גיטהב/i.test(raw) && /(?:פופולרי|popular|trending|השבוע|this\s+week)/i.test(raw)) {
    return buildGitHubPopularSearchQuery(raw);
  }
  if (/github|גיטהב/i.test(raw) && parts.length) return parts.join(" ").slice(0, 256);
  if (parts.join(" ").length >= 6) return parts.join(" ").slice(0, 256);
  return "";
};

export const buildHuggingFaceSearchQuery = (query: string): string => {
  if (/\bocr\b|זיהוי\s+טקסט|text\s+recognition|tesseract|paddleocr/i.test(query)) {
    return "ocr text recognition";
  }
  if (/vlm|vision.?language|ראייה\s*\+\s*שפה/i.test(query)) return "vision-language";
  if (/זיהוי\s+אובייקט|object\s+detection|yolo|detr/i.test(query)) return "object-detection";
  if (/תנוח(?:ה|ות)\s+גוף|pose\s+estimation|mediapipe/i.test(query)) return "pose-estimation";
  if (/webgpu/i.test(query)) return "webgpu";
  if (/(?:דפדפן|browser|onnx|wasm)/i.test(query)) return "transformers.js onnx";
  if (isHuggingFaceImageQuery(query)) {
    return "stable-diffusion";
  }
  const raw = query
    .trim()
    .replace(/hugging\s*face|huggingface|hf\.co/gi, " ")
    .replace(/\bdatasets?\b/gi, " ")
    .replace(/\bmodels?\b/gi, " ")
    .replace(/(?:חפש|חיפוש|search\s+for|find)\s+/gi, " ")
    .trim();
  const latin = raw.match(/[a-zA-Z][a-zA-Z0-9_.-]{1,}/g)?.join(" ") ?? "";
  if (latin.length >= 2) return latin.slice(0, 128);
  if (/עברית|hebrew/i.test(query)) return "hebrew";
  if (/מודל/i.test(query)) return "text-generation";
  return raw.slice(0, 64) || "text-generation";
};
