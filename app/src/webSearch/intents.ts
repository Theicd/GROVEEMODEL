import type { SearchIntent } from "./types";

export { extractLocationPhrase, sanitizeSearchQuery } from "./queryExtract";
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
  (/openai|trending|פופולרי|artificial\s+intelligence|\bai\b|machine\s+learning|בינה\s+(?:ה)?מלאכותית/i.test(
    text,
  ) &&
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

export const isShipsQuery = (text: string): boolean =>
  /(?:ספינ|אוני(?:יות|ה)|אוניות|vessel|ships?\b|ais\b|תעלת\s+סואץ|suez\s+canal|נמל\s+.*(?:ספינ|אוני|ship))/i.test(
    text,
  ) ||
  (/(?:אילו|what|which|כמה).*(?:ספינ|אוני|ship|vessel)/i.test(text) &&
    /(?:סואץ|suez|ים|sea|port|נמל|canal|תעלה)/i.test(text));

export const isSpaceXQuery = (text: string): boolean =>
  /spacex|space\s*x|סpace\s*x|ספייס\s*אקס|ספייס-?אקס/i.test(text) ||
  (/(?:שיגור|launch|rocket|טיל)/i.test(text) && /spacex|space\s*x|ספייס/i.test(text));

export const isIssQuery = (text: string): boolean =>
  /\biss\b|תחנת\s+(?:ה)?חלל\s+(?:ה)?בינלאומ|space\s+station|החלל\s+הבינלאומ/i.test(text);

export const isSatelliteCatalogQuery = (text: string): boolean =>
  (/(?:כמה|how\s+many|מספר|number\s+of).*(?:לוויין|satellite)/i.test(text) ||
    /(?:לוויינים|satellites)\s+(?:פעיל|active|במסלול|in\s+orbit)/i.test(text)) &&
  !isIssQuery(text);

/**
 * Decide if this turn should hit live web providers (no manual toggle).
 * Static facts (capital, currency name, distance) → model knowledge only.
 * Auto search = live / time-sensitive data or explicit lookup (POI, "חפש").
 */
export const needsWebSearch = (text: string): boolean => {
  const q = text.trim();
  if (!q || isCasualConversation(q)) return false;
  if (userRequestsSearch(q)) return true;

  if (isGovernmentQuery(q)) return true;
  if (isCommodityPriceQuery(q) || isMarketPriceQuery(q) || isRedditQuery(q)) return true;
  if (isYouTubeQuery(q)) return true;
  if (isStaticFactualQuery(q)) return false;
  if (isCryptoQuery(q)) return true;
  if (isIsraelAlertsQuery(q) || isDisasterQuery(q)) return true;
  if (isNewsQuery(q)) return true;
  if (isWorldTimeQuery(q)) return true;
  if (isWeatherQuery(q) || isMarineQuery(q)) return true;
  if (isCurrencyQuery(q)) return true;
  if (isFlightStatusQuery(q)) return true;
  if (isShipsQuery(q)) return true;
  if (isSpaceXQuery(q)) return true;
  if (isPlacesQuery(q)) return true;
  if (isSatelliteQuery(q) || isSpaceWeatherQuery(q)) return true;
  if (isHackerNewsQuery(q) || isTechNewsQuery(q)) return true;

  if (isEarthquakeQuery(q)) {
    return hasLiveTemporalSignal(q) || /(?:24|אחרונ|recent|latest)/i.test(q);
  }
  if (isAviationQuery(q)) {
    return hasLiveTemporalSignal(q) || /(?:מעל|above|near|over|מטוס)/i.test(q);
  }
  if (isHolidayQuery(q)) {
    return /(?:היום|האם\s+היום|today|now)/i.test(q);
  }
  if (isGitHubQuery(q) || isHuggingFaceQuery(q)) {
    return /(?:פופולרי|popular|trending|השבוע|release|גרסה|חפש|search|find)/i.test(q);
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
    (/\b(?:hotel|restaurant|hospital|station|pharmacy|מלון|מסעדה|בית\s+חולים|תחנ(?:ת|ות)\s+רכבת)\b/i.test(
      text,
    ) &&
      /\b(?:near|around|by|ליד|קרוב)\b/i.test(text)) ||
    (/\b(?:nearest|closest|הכי\s+קרוב(?:ה|ים)?)\b/i.test(text) &&
      /(?:station|רכבת|hospital|hotel|מלון|תחנ|airport|שדה\s+תעופה|tower|מגדל|louvre|לובר|eiffel|אייפ)/i.test(
        text,
      )) ||
    /(?:אילו|what|which)\s+(?:תחנ(?:ות|ת)|train\s+stations?|hospitals?|בתי\s+חולים|רכבת)/i.test(text) ||
    (/(?:תחנ(?:ות|ת)\s+רכבת|train\s+stations?|בית\s+חולים|בתי\s+חולים|תחנ(?:ת|ות)\s+דלק)/i.test(text) &&
      /(?:ליד|near|by|at|ב|של|שדה\s+תעופה|airport|הית'?רו|heathrow|eiffel|אייפ|louvre|לובר)/i.test(text)));

export const isNewsQuery = (text: string): boolean =>
  !isRedditQuery(text) &&
  !isHackerNewsQuery(text) &&
  (/(?:חדשות|news|headline|כותרת\s+ראשית|main\s+headline|breaking)/i.test(text) ||
    /(?:bbc|cnn|ynet)/i.test(text));

export const isAviationQuery = (text: string): boolean =>
  !isFlightStatusQuery(text) &&
  (/(?:מטוס|מטוסים|aircraft|airplane|plane|adsb|opensky|תעבורה\s+אווירית|טיסות\s+מעל)/i.test(text) ||
    (/(?:בעולם|worldwide|global|ברחבי\s+העולם|around\s+the\s+world|in\s+the\s+air)/i.test(text) &&
      /(?:מטוס|aircraft|plane|adsb|תעופה|air|טיס)/i.test(text)) ||
    /(?:כמה\s+)?(?:מהם|מאלה)\s*(?:הם\s+)?(?:צבאיים|military|מסחריים)(?:\s|$|[?!.])/i.test(text));

export const isSatelliteQuery = (text: string): boolean =>
  (isIssQuery(text) || isSatelliteCatalogQuery(text)) && !isAviationQuery(text);

export const isSpaceWeatherQuery = (text: string): boolean =>
  /(?:מזג\s+אוויר\s+חללי|space\s+weather|kp\s+index|רוח\s+סולארית|סערה\s+גיאומגנטית|aurora)/i.test(text);

export const isIsraelAlertsQuery = (text: string): boolean =>
  /(?:צבע\s+אדום|התרע(?:ה|ות)|פיקוד\s+העורף|oref|tzeva\s+adom|אזעק)/i.test(text);

export const isDisasterQuery = (text: string): boolean =>
  /(?:אסון|אסונות|gdacs|הוריקן|hurricane|tsunami|צונאמי|שריפ(?:ה|ות)|wildfire|סופ(?:ה|ות)\s+טרופי)/i.test(text);

export const isWeatherQuery = (text: string): boolean =>
  /מזג\s*האוויר|מז"?\s*א|טמפרטור|temperatur|weather|temperature|גשם|שלג|מעונן|לחות|מזג|מהירות\s+(?:ה)?רוח|wind\s+speed/i.test(text);

export const isMarineQuery = (text: string): boolean =>
  !isShipsQuery(text) &&
  /גלים|wave|סערה|הurricane|typhoon|גובה\s*גל|סופה|שיא\s*גלים|marine\s+weather|ocean\s+wave/i.test(text);

export const isEarthquakeQuery = (text: string): boolean =>
  /רעיד(?:ת|ות)?\s*אדמה|רעש\s*אדמה|earthquake|seismic|tsunami|רichter|סוללת\s*רעיד/i.test(text);

export const isHuggingFaceQuery = (text: string): boolean =>
  /hugging\s*face|huggingface|hf\.co|transformers\.js/i.test(text) ||
  /(?:מודל(?:י)?|datasets?)\s*(?:ai|שפה|nlp|llm)?/i.test(text) ||
  /מודל\s+ל(?:עברית|שפה|ocr|OCR|תמלול)/i.test(text) ||
  /(?:חפש|find|search).*(?:מודל|model)/i.test(text) ||
  /\bocr\b|זיהוי\s+טקסט|text\s+recognition|tesseract|paddleocr/i.test(text);

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
  if (isTechNewsQuery(query)) intents.push("hackernews");
  else if (isHackerNewsQuery(query)) intents.push("hackernews");
  if (isYouTubeQuery(query)) intents.push("youtube");
  if (isWorldTimeQuery(query)) intents.push("worldtime");
  if (isWeatherQuery(query)) intents.push("weather");
  if (isMarineQuery(query)) intents.push("marine");
  if (isShipsQuery(query)) intents.push("ships");
  if (isEarthquakeQuery(query)) intents.push("earthquake");
  if (isCurrencyQuery(query)) intents.push("currency");
  if (isDistanceQuery(query)) intents.push("distance");
  if (isPlacesQuery(query)) intents.push("places");
  if (isNewsQuery(query) && !isTechNewsQuery(query)) intents.push("news");
  if (isAviationQuery(query)) intents.push("aviation");
  if (isSatelliteCatalogQuery(query)) intents.push("satellite");
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
  if (isGitHubQuery(query) || (isTechQuery(query) && !hf) || (explicitSearch && hf)) {
    intents.push("github");
  }
  if (hf) intents.push("huggingface");

  // Only explicit GitHub/HF questions skip Wikipedia; generic tech questions
  // (e.g. "מה קורה עם React") still benefit from encyclopedic context.
  const techExclusive = isGitHubQuery(query) || hf;
  const structured = intents.some((i) =>
    [
      "worldtime", "weather", "marine", "earthquake", "currency", "holiday", "government",
      "country", "distance", "places", "news", "aviation", "satellite", "spaceweather",
      "alerts", "disaster", "crypto", "commodity", "market", "hackernews", "ships", "spacex",
    ].includes(i),
  );

  // Wikipedia only when the user explicitly asks to search — never as silent fallback.
  if (
    userRequestsSearch(query) &&
    !structured &&
    !techExclusive &&
    !isMarketPriceQuery(query) &&
    !isRedditQuery(query) &&
    !isStaticFactualQuery(query)
  ) {
    intents.push("wikipedia");
  }

  return [...new Set(intents)];
};

export const buildGitHubSearchQuery = (query: string): string => {
  const raw = query.trim();
  if (!raw) return "";
  if (/ocr|זיהוי\s+טקסט|text\s+recognition|tesseract|paddleocr/i.test(raw)) {
    return "ocr text recognition tesseract paddleocr";
  }
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
  if (/\bocr\b|זיהוי\s+טקסט|text\s+recognition|tesseract|paddleocr/i.test(query)) {
    return "ocr text recognition";
  }
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
