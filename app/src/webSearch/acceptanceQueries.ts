import type { SearchIntent, SearchProviderId } from "./types";

/** Acceptance query — used by live tests and manual QA checklist. */
export type AcceptanceQuery = {
  id: string;
  category: string;
  query: string;
  /** At least one of these intents must appear. */
  expectIntents: SearchIntent[];
  /** At least one of these providers must return ok:true. */
  expectProvidersOk: SearchProviderId[];
  /** Substrings that must appear in combined successful source text. */
  expectTextIncludes?: string[];
  notesHe?: string;
};

export const ACCEPTANCE_QUERIES: AcceptanceQuery[] = [
  {
    id: "W01",
    category: "מזג אוויר",
    query: "what is the weather in New York",
    expectIntents: ["weather"],
    expectProvidersOk: ["open-meteo"],
    expectTextIncludes: ["°C", "New York"],
    notesHe: "אנגלית — Open-Meteo geocoding + forecast",
  },
  {
    id: "W02",
    category: "מזג אוויר",
    query: "מה מזג האוויר בניו יורק?",
    expectIntents: ["weather"],
    expectProvidersOk: ["open-meteo"],
    expectTextIncludes: ["°C", "מיקום"],
    notesHe: "עברית — הבעיה המקורית שתוקנה",
  },
  {
    id: "W03",
    category: "מזג אוויר",
    query: "מה מזג האוויר בתל אביב",
    expectIntents: ["weather"],
    expectProvidersOk: ["open-meteo"],
    expectTextIncludes: ["°C"],
    notesHe: "עיר ישראלית בעברית",
  },
  {
    id: "W04",
    category: "מזג אוויר",
    query: "weather in London",
    expectIntents: ["weather"],
    expectProvidersOk: ["open-meteo"],
    expectTextIncludes: ["°C", "London"],
    notesHe: "כולל נתוני רוח",
  },
  {
    id: "M01",
    category: "ים וגלים",
    query: "wave height Tel Aviv",
    expectIntents: ["marine"],
    expectProvidersOk: ["open-meteo-marine"],
    expectTextIncludes: ["גל"],
    notesHe: "Open-Meteo Marine API",
  },
  {
    id: "M02",
    category: "ים וגלים",
    query: "wave height in Miami",
    expectIntents: ["marine"],
    expectProvidersOk: ["open-meteo-marine"],
    expectTextIncludes: ["גל"],
    notesHe: "אנגלית — marine (weather intent may co-trigger)",
  },
  {
    id: "E01",
    category: "רעידות אדמה",
    query: "רעידות אדמה אחרונות",
    expectIntents: ["earthquake"],
    expectProvidersOk: ["usgs-earthquake"],
    expectTextIncludes: ["M"],
    notesHe: "USGS feed יומי",
  },
  {
    id: "E02",
    category: "רעידות אדמה",
    query: "recent earthquakes Japan",
    expectIntents: ["earthquake"],
    expectProvidersOk: ["usgs-earthquake"],
    expectTextIncludes: ["M"],
    notesHe: "סינון לפי אזור (Japan) אם קיים בפיד",
  },
  {
    id: "K01",
    category: "Wikipedia",
    query: "who was Albert Einstein",
    expectIntents: ["wikipedia"],
    expectProvidersOk: ["wikipedia-en"],
    expectTextIncludes: ["Einstein"],
    notesHe: "extract מלא מאנגלית",
  },
  {
    id: "K02",
    category: "Wikipedia",
    query: "חפש מידע על פירמידות",
    expectIntents: ["wikipedia"],
    expectProvidersOk: ["wikipedia-he", "wikipedia-en"],
    expectTextIncludes: [],
    notesHe: "עברית + מילת חיפוש מפורשת",
  },
  {
    id: "G01",
    category: "GitHub",
    query: "github open source llm chat",
    expectIntents: ["github"],
    expectProvidersOk: ["github"],
    expectTextIncludes: ["★", "http"],
    notesHe: "מאגרים עם כוכבים",
  },
  {
    id: "G02",
    category: "GitHub",
    query: "פרויקט github למצלמות אבטחה",
    expectIntents: ["github"],
    expectProvidersOk: ["github"],
    expectTextIncludes: [],
    notesHe: "תרגום hints מעברית לאנגלית",
  },
  {
    id: "H01",
    category: "Hugging Face",
    query: "gemma models huggingface",
    expectIntents: ["huggingface"],
    expectProvidersOk: ["huggingface-models", "github"],
    expectTextIncludes: ["gemma"],
    notesHe: "חיפוש מודלים",
  },
  {
    id: "H02",
    category: "Hugging Face",
    query: "hebrew dataset",
    expectIntents: ["huggingface"],
    expectProvidersOk: ["huggingface-datasets", "huggingface-models", "github"],
    expectTextIncludes: ["hebrew"],
    notesHe: "חיפוש datasets",
  },
  {
    id: "T01",
    category: "שעון עולמי",
    query: "what time in Tokyo",
    expectIntents: ["worldtime"],
    expectProvidersOk: ["world-time"],
    expectTextIncludes: ["Tokyo"],
    notesHe: "WorldTimeAPI + geocoding timezone",
  },
  {
    id: "T02",
    category: "שעון עולמי",
    query: "מה השעה בניו יורק",
    expectIntents: ["worldtime"],
    expectProvidersOk: ["world-time"],
    expectTextIncludes: ["UTC"],
    notesHe: "עברית — שעה מקומית",
  },
  {
    id: "C01",
    category: "מדינות",
    query: "מה הבירה של גרמניה",
    expectIntents: ["country"],
    expectProvidersOk: ["rest-countries"],
    expectTextIncludes: ["Berlin"],
    notesHe: "REST Countries",
  },
  {
    id: "C02",
    category: "מדינות",
    query: "population of Israel",
    expectIntents: ["country"],
    expectProvidersOk: ["rest-countries"],
    expectTextIncludes: ["Israel"],
    notesHe: "אוכלוסיה + metadata",
  },
  {
    id: "H03",
    category: "חגים",
    query: "האם היום חג בגרמניה",
    expectIntents: ["holiday"],
    expectProvidersOk: ["nager-holidays"],
    expectTextIncludes: ["Germany"],
    notesHe: "Nager.Date public holidays",
  },
  {
    id: "G03",
    category: "ממשל",
    query: "מי ראש הממשלה של ישראל",
    expectIntents: ["government"],
    expectProvidersOk: ["wikidata-gov"],
    expectTextIncludes: [],
    notesHe: "Wikidata SPARQL — may vary by live data",
  },
  {
    id: "F01",
    category: "מטבעות",
    query: "USD to ILS exchange rate",
    expectIntents: ["currency"],
    expectProvidersOk: ["frankfurter-fx"],
    expectTextIncludes: ["USD"],
    notesHe: "Frankfurter ECB rates",
  },
  {
    id: "X01",
    category: "Router",
    query: "מה קורה עם React hooks",
    expectIntents: ["github", "wikipedia"],
    expectProvidersOk: ["wikipedia-en", "github"],
    expectTextIncludes: [],
    notesHe: "שאלה טכנית — wiki + github fallback",
  },
  // --- QA מלא (2026-06-13) ---
  { id: "QA-B01", category: "בסיסיות", query: "מה המטבע של ברזיל?", expectIntents: ["country"], expectProvidersOk: ["rest-countries"], expectTextIncludes: ["BRL"], notesHe: "REST Countries" },
  { id: "QA-B02", category: "בסיסיות", query: "מה בירת יפן?", expectIntents: ["country"], expectProvidersOk: ["rest-countries"], expectTextIncludes: ["Tokyo"], notesHe: "" },
  { id: "QA-B03", category: "בסיסיות", query: "מי ראש הממשלה של בריטניה?", expectIntents: ["government"], expectProvidersOk: ["wikidata-gov"], expectTextIncludes: [], notesHe: "Wikidata live" },
  { id: "QA-B04", category: "בסיסיות", query: "כמה תושבים יש בקנדה?", expectIntents: ["country"], expectProvidersOk: ["rest-countries"], expectTextIncludes: ["Canada"], notesHe: "" },
  { id: "QA-T01", category: "זמן", query: "מה השעה כרגע בטוקיו?", expectIntents: ["worldtime"], expectProvidersOk: ["world-time"], expectTextIncludes: ["Tokyo"], notesHe: "" },
  { id: "QA-T02", category: "זמן", query: "מה התאריך בניו יורק?", expectIntents: ["worldtime"], expectProvidersOk: ["world-time"], expectTextIncludes: [], notesHe: "" },
  { id: "QA-T03", category: "זמן", query: "כמה שעות הפרש בין ישראל ללונדון?", expectIntents: ["worldtime"], expectProvidersOk: ["world-time"], expectTextIncludes: ["הפרש"], notesHe: "" },
  { id: "QA-T04", category: "זמן", query: "בוקר טוב מר גרובי, איזה יום ומה השעה בישראל", expectIntents: ["worldtime"], expectProvidersOk: ["world-time"], expectTextIncludes: ["ישראל"], notesHe: "sanitize greeting" },
  { id: "QA-W01", category: "מזג אוויר", query: "מה הטמפרטורה כרגע בתל אביב?", expectIntents: ["weather"], expectProvidersOk: ["open-meteo"], expectTextIncludes: ["°C"], notesHe: "Globe לא נפתח" },
  { id: "QA-W02", category: "מזג אוויר", query: "האם צפוי גשם בלונדון היום?", expectIntents: ["weather"], expectProvidersOk: ["open-meteo"], expectTextIncludes: [], notesHe: "" },
  { id: "QA-W03", category: "מזג אוויר", query: "מה מהירות הרוח בפריז?", expectIntents: ["weather"], expectProvidersOk: ["open-meteo"], expectTextIncludes: [], notesHe: "" },
  { id: "QA-F01", category: "מטבעות", query: "כמה שקלים שווים 100 דולר?", expectIntents: ["currency"], expectProvidersOk: ["frankfurter-fx"], expectTextIncludes: [], notesHe: "" },
  { id: "QA-F02", category: "מטבעות", query: "כמה BRL מקבלים עבור 1 דולר?", expectIntents: ["currency"], expectProvidersOk: ["frankfurter-fx"], expectTextIncludes: [], notesHe: "" },
  { id: "QA-F03", category: "מטבעות", query: "כמה יורו שווים 1000 שקלים?", expectIntents: ["currency"], expectProvidersOk: ["frankfurter-fx"], expectTextIncludes: [], notesHe: "" },
  { id: "QA-P01", category: "מפות", query: "מצא בית חולים ליד מגדל אייפel", expectIntents: ["places"], expectProvidersOk: ["nominatim-places"], expectTextIncludes: [], notesHe: "Nominatim/OSM" },
  { id: "QA-P02", category: "מפות", query: "מצא תחנת דלק ליד נמל התעופה הית'רo", expectIntents: ["places"], expectProvidersOk: ["nominatim-places"], expectTextIncludes: [], notesHe: "" },
  { id: "QA-P03", category: "מפות", query: "אילו תחנות רכבת יש ליד שדה התעופה הית'רo?", expectIntents: ["places"], expectProvidersOk: ["nominatim-places"], expectTextIncludes: [], notesHe: "" },
  { id: "QA-P04", category: "מפות", query: 'כמה ק"מ בין ירושלים לחיפה?', expectIntents: ["distance"], expectProvidersOk: ["osrm-distance"], expectTextIncludes: ["km"], notesHe: "" },
  { id: "QA-N01", category: "חדשות", query: "מה החדשות האחרונות על OpenAI?", expectIntents: ["news"], expectProvidersOk: ["news-rss"], expectTextIncludes: [], notesHe: "RSS CORS may vary" },
  { id: "QA-N02", category: "חדשות", query: "מה הכותרת הראשית באתר BBC עכשיו?", expectIntents: ["news"], expectProvidersOk: ["news-rss"], expectTextIncludes: [], notesHe: "" },
  { id: "QA-GH1", category: "GitHub", query: "מהם הפרויקטים הפופולריים ביותר השבוע ב-GitHub?", expectIntents: ["github"], expectProvidersOk: ["github"], expectTextIncludes: ["★"], notesHe: "ללא Wikipedia" },
  { id: "QA-GH2", category: "GitHub", query: "חפש פרויקטים בנושא WebGPU", expectIntents: ["github"], expectProvidersOk: ["github"], expectTextIncludes: [], notesHe: "" },
  { id: "QA-HF1", category: "Hugging Face", query: "מהם מודלי התמונה הפופולריים ביותר השבוע?", expectIntents: ["huggingface"], expectProvidersOk: ["huggingface-models"], expectTextIncludes: [], notesHe: "" },
  { id: "QA-CR1", category: "קריפטו", query: "מה מחיר הביטקוין עכשיו?", expectIntents: ["crypto"], expectProvidersOk: ["coingecko"], expectTextIncludes: ["USD"], notesHe: "CoinGecko free" },
  { id: "QA-EQ1", category: "רעידות", query: "אילו רעידות אדמה התרחשו ב-24 השעות האחרונות?", expectIntents: ["earthquake"], expectProvidersOk: ["usgs-earthquake"], expectTextIncludes: ["M"], notesHe: "" },
  { id: "QA-EQ2", category: "רעידות", query: "יש לך מידע על רעידות אדמה? למשל בישראל?", expectIntents: ["earthquake"], expectProvidersOk: ["usgs-earthquake"], expectTextIncludes: [], notesHe: "needsWebSearch always for earthquake" },
  { id: "QA-EQ3", category: "רעידות", query: "איפה הייתה רעידת האדמה החזקה בעולם ב-24 השעות האחרונות?", expectIntents: ["earthquake"], expectProvidersOk: ["usgs-earthquake"], expectTextIncludes: ["M"], notesHe: "" },
  { id: "QA-AV1", category: "טיסות", query: "כמה מטוסים נמצאים כרגע מעל ישראל?", expectIntents: ["aviation"], expectProvidersOk: ["adsb-aviation"], expectTextIncludes: [], notesHe: "OpenSky/ADSB" },
  { id: "QA-ISS1", category: "חלל", query: "מתי תחנת החלל הבינלאומית תעבור מעל ישראל?", expectIntents: ["satellite"], expectProvidersOk: ["iss-tracker"], expectTextIncludes: [], notesHe: "" },
  { id: "QA-ISS2", category: "חלל", query: "איפה תחנת החלל הבינלאומית עכשיו?", expectIntents: ["satellite"], expectProvidersOk: ["iss-tracker"], expectTextIncludes: ["קו"], notesHe: "" },
  { id: "QA-SH2", category: "ספינות", query: "כמה כלי שייט או אוניות יש במפרץ חיפה?", expectIntents: ["ships"], expectProvidersOk: ["ais-ships"], expectTextIncludes: ["ספינות בטווח"], notesHe: "Haifa bay + medPorts route markers" },
  { id: "QA-SH3", category: "תשתיות ימיות", query: "כמה מצופים יש במפרץ חיפה?", expectIntents: ["marine-infra"], expectProvidersOk: ["osm-overpass-marine"], expectTextIncludes: ["תשתיות"], notesHe: "Overpass OSM static infra" },
];

/** שאלות שדורשות API key / proxy — צפוי הודעת unsupported ברורה */
export const QA_UNSUPPORTED_QUERIES: Array<{ id: string; query: string; reasonHe: string }> = [
  { id: "QA-R01", query: "על מה מדברים עכשיו ב-r/LocalLLaMA?", reasonHe: "Reddit OAuth" },
  { id: "QA-S01", query: "מה מחיר מניית NVIDIA עכשיו?", reasonHe: "Finnhub/Alpha Vantage API key" },
  { id: "QA-S02", query: "מה מצב מדד S&P 500?", reasonHe: "Finnhub API key" },
  { id: "QA-G01", query: "מה מחיר הזהב כרגע?", reasonHe: "Finnhub/commodities API key" },
  { id: "QA-SH1", query: "אילו אוניות נמצאות כרגע בנמל חיפה?", reasonHe: "Marine AIS — לא מחובר" },
];

/** מקורות מידע — יעד ארכיטקטורה */
export const DATA_SOURCE_REGISTRY: Array<{ name: string; status: "live" | "partial" | "planned" | "needs-key"; notesHe: string }> = [
  { name: "Wikipedia / Wikidata", status: "live", notesHe: "wiki + SPARQL ממשל" },
  { name: "Open-Meteo", status: "live", notesHe: "מזג אוויר + marine" },
  { name: "OpenStreetMap / Nominatim", status: "live", notesHe: "POI, מרחקים OSRM, Overpass תשתיות ימיות" },
  { name: "TimeAPI.io", status: "live", notesHe: "שעון עולמי" },
  { name: "Frankfurter", status: "live", notesHe: "מטבעות FX" },
  { name: "REST Countries", status: "live", notesHe: "מדינות" },
  { name: "GitHub API", status: "live", notesHe: "repos" },
  { name: "Hugging Face API", status: "live", notesHe: "models/datasets" },
  { name: "USGS / GDACS / ISS / ADSB", status: "partial", notesHe: "realityData providers" },
  { name: "CoinGecko", status: "live", notesHe: "קריפטו free tier" },
  { name: "Hacker News", status: "live", notesHe: "Firebase API" },
  { name: "News RSS", status: "partial", notesHe: "CORS על חלק מהפידים" },
  { name: "Reddit", status: "needs-key", notesHe: "OAuth" },
  { name: "Finnhub / Alpha Vantage", status: "needs-key", notesHe: "מניות/סחורות" },
  { name: "SearXNG / Brave Search", status: "planned", notesHe: "דורש self-host או API key" },
  { name: "Overpass API", status: "planned", notesHe: "POI מתקדם" },
  { name: "arXiv", status: "planned", notesHe: "מאמרים" },
];

/** Quick manual checklist for UI testing (Search toggle ON unless noted). */
export const MANUAL_UI_CHECKS: Array<{
  step: number;
  actionHe: string;
  query: string;
  expectHe: string;
}> = ACCEPTANCE_QUERIES.slice(0, 10).map((q, i) => ({
  step: i + 1,
  actionHe: `סמן Search → שלח: «${q.query}»`,
  query: q.query,
  expectHe: `בלוק מקורות עם ${q.expectProvidersOk.join(" / ")} + תשובת AI`,
}));
