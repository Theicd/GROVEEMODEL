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
