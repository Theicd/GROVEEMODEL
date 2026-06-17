/** Full Hebrew topic phrases → English engine query (longest match first). */
const HEBREW_PHRASE_TO_EN: Record<string, string> = {
  "בינה מלאכותית יישומית": "artificial intelligence",
  "בינה מלאכותית": "artificial intelligence",
  "סייבר ואבטחת מידע": "cybersecurity",
  "פוליטיקה בישראל": "israel politics",
  "כלכלה ושוק ההון": "stock market",
  "טכנולוגיה וסטארטאפים": "technology startups",
  "מדע וחלל": "space science",
  "מזג אוויר קיצוני": "extreme weather",
  "מלחמות וסכסוכים": "war conflict",
  "אנרגיה וחשמל ירוק": "renewable energy",
  "תחבורה ורכב חשמלי": "electric vehicle",
  "קריפטו וביטקוין": "bitcoin crypto",
  "חברות טכנולוגיה גדולות": "big tech",
  "רשתות חברתיות ואינטרנט": "social media",
  "בריאות ורפואה": "health medicine",
  "חינוך וטכנולוגיות למידה": "education technology",
  "חוק ורגולציה טכנולוגית": "tech regulation",
  "צבא וביטחון": "military defense",
  "חלל וחקר היקום": "space",
  "גיימינג ותעשיית המשחקים": "gaming",
  "תרבות ובידור": "entertainment culture",
  "ספורט עולמי": "sports",
  "אירועים חריגים": "breaking news disaster",
  "חדשנות ובינה מלאכותית": "artificial intelligence innovation",
  "סטארטאפים ישראליים": "israel startups",
  "חקלאות וטכנולוגיות מזון": "agriculture food tech",
  "אבטחת מידע": "cybersecurity",
  "שוק ההון": "stock market",
  "רכב חשמלי": "electric vehicle",
  "רגולציה טכנולוגית": "tech regulation",
  "תעשיית המשחקים": "gaming industry",
};

/** Map Hebrew news topics to English engine search terms (RSS index is mostly English). */
const HEBREW_TOPIC_TO_EN: Record<string, string> = {
  חלל: "space",
  נאסא: "nasa",
  לוויין: "satellite",
  איראן: "iran",
  ישראל: "israel",
  ישראליים: "israel",
  עזה: "gaza",
  חמאס: "hamas",
  אוקראינה: "ukraine",
  רוסיה: "russia",
  סין: "china",
  טורקיה: "turkey",
  הודו: "india",
  ברזיל: "brazil",
  אירופה: "europe",
  ארהב: "united states",
  אמריקה: "united states",
  מלחמה: "war",
  מלחמות: "war",
  סכסוכים: "conflict",
  כלכלה: "economy",
  שוק: "market",
  בורסה: "market",
  נפט: "oil",
  אנרגיה: "energy",
  סייבר: "cyber",
  האקרים: "cyber",
  בינה: "ai",
  טכנולוגיה: "technology",
  רובוטיקה: "robotics",
  מדע: "science",
  בריאות: "health",
  רפואה: "medicine",
  אקלים: "climate",
  מזג: "weather",
  אוויר: "weather",
  קיצוני: "extreme",
  ספורט: "sport",
  כדורגל: "football",
  מוזיקה: "music",
  קולנוע: "film",
  משחקים: "gaming",
  גיימינג: "gaming",
  רכב: "car",
  רכבים: "car",
  תחבורה: "transport",
  תעופה: "aviation",
  מטוס: "aviation",
  ספינות: "maritime",
  הים: "marine",
  פלסטינים: "palestine",
  לבנון: "lebanon",
  סוריה: "syria",
  ערבים: "arab",
  נשק: "nuclear",
  גרעין: "nuclear",
  ביטקוין: "bitcoin",
  קריפטו: "crypto",
  מחאות: "protest",
  בחירות: "election",
  פוליטיקה: "politics",
  סטארטאפים: "startups",
  סטארטאפ: "startup",
  חדשנות: "innovation",
  חינוך: "education",
  למידה: "learning",
  חוק: "regulation",
  רגולציה: "regulation",
  צבא: "military",
  ביטחון: "defense",
  תרבות: "culture",
  בידור: "entertainment",
  חריגים: "disaster",
  אירועים: "breaking",
  חקלאות: "agriculture",
  מזון: "food",
  אינטרנט: "internet",
  רשתות: "social",
  חברתיות: "social",
  חברות: "companies",
  ירוק: "renewable",
  חשמל: "electric",
  חשמלי: "electric",
  טראמפ: "trump",
  ביידן: "biden",
  נתניהו: "netanyahu",
  לונדון: "london",
  פריז: "paris",
  ברלין: "berlin",
  מוסקבה: "moscow",
  מדריד: "madrid",
  רומא: "rome",
  אתונה: "athens",
  וינה: "vienna",
  בייגינג: "beijing",
  שאנגחאי: "shanghai",
  טוקיו: "tokyo",
  סיאול: "seoul",
  דובאי: "dubai",
  קהיר: "cairo",
  ירושלים: "jerusalem",
  "תל אביב": "tel aviv",
  "תל-אביב": "tel aviv",
  חיפה: "haifa",
  "ניו יורק": "new york",
  "ניו-יורק": "new york",
  וושינגטון: "washington",
  "לוס אנג'לס": "los angeles",
  "לוס אנג׳לס": "los angeles",
  שיקגו: "chicago",
  מיאמי: "miami",
  בוסטון: "boston",
  "סן פרנסיסקו": "san francisco",
  "מנצ'סטר": "manchester",
  ליברפול: "liverpool",
  אמסטרדם: "amsterdam",
  בוקרשט: "bucharest",
  ורשה: "warsaw",
  קייב: "kyiv",
  בגדד: "baghdad",
  ריאד: "riyadh",
  סידני: "sydney",
  מלבורן: "melbourne",
  מקסיקו: "mexico city",
  "בואנוס איירס": "buenos aires",
};

const HEBREW_STOP_RE =
  /(?:חפש|מצא|תביא|עוד|חדשות|כותרות|כתבות|דיווחים|אחרונות|על|בנושא|מה|ה|של|את|עם|לי|תן|ספר|בבקשה|אפשר|יש|כרגע|עכשיו|היום|בעולם|בישראל|latest|news|headlines?|about|search|find|ו|גדולות|יישומית|היקום|המשחקים|החברתיות|המזון|ההון|המידע|הלמידה|הטכנולוגית|המחשב|העולמי)/gi;

const TOPIC_PHRASE_RE =
  /(?:חפש|מצא|תביא)?\s*(?:חדשות|כותרות|כתבות|דיווחים)?\s*(?:על|בנושא|לגבי|ב|מ|מתוך)\s+(.+?)(?:[?.!]|$)/i;

const BROAD_OVERVIEW_RE =
  /^(?:world news|world|global news|international news|headlines|news today)$/i;

const sortedHebrewPhrases = () =>
  Object.keys(HEBREW_PHRASE_TO_EN).sort((a, b) => b.length - a.length);

const sortedHebrewKeys = () =>
  Object.keys(HEBREW_TOPIC_TO_EN).sort((a, b) => b.length - a.length);

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Avoid matching short Hebrew keys inside longer words (e.g. ים inside סטארטאפים). */
function containsHebrewKey(text: string, key: string): boolean {
  if (!text || !key) return false;
  if (key.length <= 2) {
    return new RegExp(`(^|\\s)${escapeRegExp(key)}(\\s|$)`).test(text);
  }
  return text.includes(key);
}

function stripScopeSuffix(phrase: string): string {
  return phrase
    .replace(/\s+בעולם\s*$/i, "")
    .replace(/\s+באזור\s*$/i, "")
    .trim();
}

function mapPhraseToEnglish(phrase: string): string | null {
  const cleaned = stripScopeSuffix(phrase.trim());
  if (!cleaned) return null;

  for (const he of sortedHebrewPhrases()) {
    if (cleaned.includes(he)) return HEBREW_PHRASE_TO_EN[he];
  }

  if (HEBREW_PHRASE_TO_EN[cleaned]) return HEBREW_PHRASE_TO_EN[cleaned];
  return null;
}

/** Extract explicit topic phrase from Hebrew news-style queries. */
export function extractNewsTopicPhrase(query: string): string {
  const raw = query.trim();
  if (!raw) return "";
  const phrase = raw.match(TOPIC_PHRASE_RE)?.[1]?.trim() ?? "";
  return stripScopeSuffix(phrase);
}

/** Explicit «חדשות בנושא X» / «חדשות על X» — always topic search, not world digest. */
export function isExplicitNewsTopicSearch(query: string): boolean {
  return /(?:חפש|מצא|תביא)?\s*חדשות\s+(?:בנושא|על|לגבי)\s+\S+/i.test(query.trim());
}

/** True when the user named a concrete topic (city, country, entity) — not a world overview. */
export function isSpecificNewsTopicQuery(query: string): boolean {
  const raw = query.trim();
  if (!raw) return false;
  if (isExplicitNewsTopicSearch(raw)) return true;
  const phrase = extractNewsTopicPhrase(raw);
  if (phrase && /[\u0590-\u05FF]/.test(phrase)) return true;
  const normalized = normalizeNewsEngineQuery(raw);
  if (!normalized || BROAD_OVERVIEW_RE.test(normalized)) return false;
  if (/(?:בעולם|עולם|מה קורה|מה חדש|כותרות מובילות)/i.test(raw) && !phrase) return false;
  return normalized.split(/\s+/).some((t) => t.length > 2 && !BROAD_OVERVIEW_RE.test(t));
}

function mapHebrewTextToEnglish(text: string): string[] {
  const mapped: string[] = [];
  let work = stripScopeSuffix(text.trim());
  if (!work) return mapped;

  const phraseHit = mapPhraseToEnglish(work);
  if (phraseHit) return [phraseHit];

  for (const he of sortedHebrewKeys()) {
    if (containsHebrewKey(work, he)) {
      mapped.push(HEBREW_TOPIC_TO_EN[he]);
      work = work.replaceAll(he, " ");
    }
  }

  const leftover = work.replace(/\s+/g, " ").trim();
  if (leftover && HEBREW_TOPIC_TO_EN[leftover]) {
    mapped.push(HEBREW_TOPIC_TO_EN[leftover]);
  }

  return mapped;
}

/** Normalize user chat query → English terms the NEWS engine can match in RSS headlines. */
export function normalizeNewsEngineQuery(query: string): string {
  const raw = query.trim();
  if (!raw) return "";

  const phrase = extractNewsTopicPhrase(raw);
  let work = phrase || raw;

  const phraseEnglish = mapPhraseToEnglish(work);
  if (phraseEnglish) return phraseEnglish;

  const mapped = mapHebrewTextToEnglish(work);

  const english = work
    .replace(HEBREW_STOP_RE, " ")
    .replace(/[^\w\s'-]/g, " ")
    .split(/\s+/)
    .map((w) => w.trim().toLowerCase())
    .filter((w) => w.length > 1);

  const terms = [...new Set([...mapped, ...english])];
  if (terms.length) return terms.join(" ");

  if (phrase && /[\u0590-\u05FF]/.test(phrase)) {
    const direct = HEBREW_TOPIC_TO_EN[phrase.trim()] ?? HEBREW_PHRASE_TO_EN[phrase.trim()];
    if (direct) return direct;
    return "";
  }

  if (/[\u0590-\u05FF]/.test(raw)) {
    if (/(?:בישראל|ישראל)/i.test(raw)) return "israel";
    if (/(?:בעולם|עולם)/i.test(raw) && !phrase) return "world news";
    return "";
  }

  return raw;
}

export function extractNewsTopicTerms(query: string): string[] {
  const normalized = normalizeNewsEngineQuery(query);
  return normalized
    .toLowerCase()
    .split(/\s+/)
    .filter((t) => t.length > 1 && !BROAD_OVERVIEW_RE.test(t));
}

export function isBroadNewsOverviewQuery(engineQuery: string): boolean {
  const q = engineQuery.trim().toLowerCase();
  return !q || BROAD_OVERVIEW_RE.test(q);
}
