import type { SearchIntent } from "../webSearch/types";

export type NewsPanelMode = "topics" | "search";

/** Manual + automated QA for GROVEE NEWS chat → panel → local index. */
export type NewsAcceptanceQuery = {
  id: string;
  /** Hebrew label for checklist */
  labelHe: string;
  query: string;
  /** Expected routing intent (must include news unless noted). */
  expectIntents: SearchIntent[];
  /** Topics digest vs keyword search in engine. */
  expectPanelMode: NewsPanelMode;
  /** After normalizeNewsEngineQuery — only for search mode. */
  expectEngineQuery?: string;
  /** At least one card title should match one keyword (live / manual). */
  expectTitleKeywords?: string[];
  /** grovee-news must return ok with text. */
  expectProviderOk: boolean;
  /** What the human tester should confirm in UI. */
  verifyHe: string;
};

/**
 * 11 questions — different phrasings, flows, and expected outcomes.
 * Run routing: npm test -- app/src/groveeNews/newsAcceptance.test.ts
 * Run smoke:   npm run qa:news-chat
 */
export const NEWS_ACCEPTANCE_QUERIES: NewsAcceptanceQuery[] = [
  {
    id: "NEWS-Q01",
    labelHe: "סקירת עולם (קצר)",
    query: "מה קורה בעולם?",
    expectIntents: ["news"],
    expectPanelMode: "topics",
    expectProviderOk: true,
    verifyHe:
      "פאנל ימין נפתח · מצב Topics · כרטיסיות מכמה נושאים (AI, מלחמה, שוק…) · grovee-news ירוק בחיפוש",
  },
  {
    id: "NEWS-Q02",
    labelHe: "מה חדש — וריאנט ניסוח",
    query: "מה חדש בעולם?",
    expectIntents: ["news"],
    expectPanelMode: "topics",
    expectProviderOk: true,
    verifyHe: "כמו Q01 — Topics, לא חיפוש מילת מפתח בודדת · תשובת Gemma מסכמת כותרות",
  },
  {
    id: "NEWS-Q03",
    labelHe: "חיפוש מפורש + נושא עברית→אנגלית",
    query: "חפש חדשות על חלל",
    expectIntents: ["news"],
    expectPanelMode: "search",
    expectEngineQuery: "space",
    expectTitleKeywords: ["space", "nasa", "rocket", "orbit", "moon", "mars", "חלל"],
    expectProviderOk: true,
    verifyHe: "פאנל Search · כרטיסיות עם חלל/NASA · לא מסך ריק",
  },
  {
    id: "NEWS-Q04",
    labelHe: "נושא גיאופוליטי בעברית",
    query: "חדשות על איראן",
    expectIntents: ["news"],
    expectPanelMode: "search",
    expectEngineQuery: "iran",
    expectTitleKeywords: ["iran", "tehran", "irgc", "איראן"],
    expectProviderOk: true,
    verifyHe: "כרטיסיות רלוונטיות לאיראן · Gemma מצטט מקורות מהמאגר",
  },
  {
    id: "NEWS-Q05",
    labelHe: "ישות באנגלית — חיפוש ממוקד",
    query: "חדשות על OpenAI",
    expectIntents: ["news"],
    expectPanelMode: "search",
    expectEngineQuery: "openai",
    expectTitleKeywords: ["openai", "chatgpt", "gpt", "sam altman"],
    expectProviderOk: true,
    verifyHe: "Search (לא Topics) · כרטיסיות OpenAI/AI · grovee-news ok",
  },
  {
    id: "NEWS-Q06",
    labelHe: "ישראל — סקירה כללית (Topics)",
    query: "מה קורה בישראל?",
    expectIntents: ["news"],
    expectPanelMode: "topics",
    expectTitleKeywords: ["israel", "gaza", "idf", "netanyahu", "ישראל", "עזה"],
    expectProviderOk: true,
    verifyHe: "Topics עם נושא ישראל · לא רק כותרות עולם · פאנל נפתח",
  },
  {
    id: "NEWS-Q07",
    labelHe: "דיגסט יומי — ניסוח דיבורי",
    query: "ספר לי חדשות היום",
    expectIntents: ["news"],
    expectPanelMode: "topics",
    expectProviderOk: true,
    verifyHe: "Topics או כותרות מעורבות · grovee-news ok · לא תשובה ריקה",
  },
  {
    id: "NEWS-Q08",
    labelHe: "כותרות מובילות בעולם",
    query: "מה הכותרות המובילות בעולם",
    expectIntents: ["news"],
    expectPanelMode: "topics",
    expectProviderOk: true,
    verifyHe: "Topics · מגוון מקורות RSS · פאנל עם תמונות/מקורות שונים",
  },
  {
    id: "NEWS-Q09",
    labelHe: "שאלה באנגלית מלאה",
    query: "latest news about Ukraine",
    expectIntents: ["news"],
    expectPanelMode: "search",
    expectEngineQuery: "ukraine",
    expectTitleKeywords: ["ukraine", "kyiv", "zelensky", "russia"],
    expectProviderOk: true,
    verifyHe: "Search באנגלית · כרטיסיות אוקראינה/מלחמה · ממשק עדיין בעברית",
  },
  {
    id: "NEWS-Q10",
    labelHe: "עולם + SearXNG blend (regex planner)",
    query: "מה המצב בעולם?",
    expectIntents: ["news"],
    expectPanelMode: "topics",
    expectProviderOk: true,
    verifyHe:
      "Topics בפאנל · בחיפוש גם searxng אופציונלי (blend) · grovee-news חייב ok · לא ליפול ל-Hacker News בלבד",
  },
  {
    id: "NEWS-Q11",
    labelHe: "עיר בעברית — לונדון → london",
    query: "חפש חדשות על לונדון",
    expectIntents: ["news"],
    expectPanelMode: "search",
    expectEngineQuery: "london",
    expectTitleKeywords: ["london", "uk", "britain", "british", "westminster", "לונדון"],
    expectProviderOk: true,
    verifyHe: "Search עם london · לא מונדיאל/עולם · כרטיסיות על לונדון/בריטניה",
  },
];
