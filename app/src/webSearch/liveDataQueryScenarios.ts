import type { LiveWorldLayer } from "../liveWorld/types";
import type { SearchIntent, SearchProviderId } from "./types";

/**
 * Live-data scenarios: chat query → providers → עולם חי → SEARCH BRIEF for Gemma.
 * Run: npm test -- app/src/webSearch/liveDataHandoffQa.test.ts
 */
export type LiveDataScenario = {
  id: string;
  category: string;
  userQuery: string;
  expectIntents: SearchIntent[];
  expectProviders: SearchProviderId[];
  expectLiveWorldLayers: LiveWorldLayer[];
  /** RSS engine query after normalizeNewsEngineQuery */
  expectRssTerms: string[];
  expectMinMagnitude?: number | null;
  /** Substrings expected in USGS-style brief text */
  expectSensorBriefIncludes?: string[];
  verifyHe: string;
};

export const LIVE_DATA_SCENARIOS: LiveDataScenario[] = [
  {
    id: "LD-EQ01",
    category: "רעידות אדמה",
    userQuery: "רעידות אדמה אחרונות",
    expectIntents: ["earthquake", "news"],
    expectProviders: ["usgs-earthquake", "grovee-news"],
    expectLiveWorldLayers: ["earthquake"],
    expectRssTerms: ["earthquake"],
    expectSensorBriefIncludes: ["M", "USGS"],
    verifyHe: "USGS + RSS · עולם חי שומר היסטוריית 24ש",
  },
  {
    id: "LD-EQ02",
    category: "רעידות אדמה",
    userQuery:
      "האם היו רעידות אדמה ב-24 השעות האחרונות מעל 5 בסולם ריכטר?",
    expectIntents: ["earthquake", "news", "disaster"],
    expectProviders: ["usgs-earthquake", "grovee-news", "gdacs-disasters"],
    expectLiveWorldLayers: ["earthquake"],
    expectRssTerms: ["earthquake"],
    expectMinMagnitude: 5,
    expectSensorBriefIncludes: ["מעל M5", "M"],
    verifyHe: "סינון M≥5 · מיקום+שעה · כותרות RSS על אסונות",
  },
  {
    id: "LD-EQ03",
    category: "רעידות אדמה",
    userQuery: "איפה הייתה רעידת האדמה החזקה בעולם ב-24 השעות האחרונות?",
    expectIntents: ["earthquake", "news"],
    expectProviders: ["usgs-earthquake", "grovee-news"],
    expectLiveWorldLayers: ["earthquake"],
    expectRssTerms: ["earthquake"],
    expectMinMagnitude: 5,
    expectSensorBriefIncludes: ["הרעידה החזקה", "M"],
    verifyHe: "החזקה בעולם · לא סינון אזור שגוי",
  },
  {
    id: "LD-EQ04",
    category: "רעידות אדמה",
    userQuery: "האם הייתה רעידת אדמה בישראל השבוע?",
    expectIntents: ["earthquake", "news"],
    expectProviders: ["usgs-earthquake", "grovee-news"],
    expectLiveWorldLayers: ["earthquake"],
    expectRssTerms: ["earthquake"],
    verifyHe: "סינון ישראל/Dead Sea · עולם חי מסנן Israel",
  },
  {
    id: "LD-EQ05",
    category: "רעידות אדמה",
    userQuery: "מה קורה באזור רעידת אדמה? יש גם חדשות?",
    expectIntents: ["earthquake", "news"],
    expectProviders: ["usgs-earthquake", "grovee-news"],
    expectLiveWorldLayers: ["earthquake"],
    expectRssTerms: ["earthquake"],
    verifyHe: "חיישן + RSS · Gemma מקבל SENSOR+RSS ב-brief",
  },
  {
    id: "LD-EQ06",
    category: "רעידות אדמה",
    userQuery: "היו רעידות אדמה חזקות הלילה?",
    expectIntents: ["earthquake", "news", "disaster"],
    expectProviders: ["usgs-earthquake", "grovee-news", "gdacs-disasters"],
    expectLiveWorldLayers: ["earthquake"],
    expectRssTerms: ["earthquake"],
    expectMinMagnitude: 5,
    verifyHe: "«חזקות» → M≥5 · GDACS אופציונלי",
  },
  {
    id: "LD-EQ07",
    category: "רעידות אדמה",
    userQuery: "recent earthquakes Japan above 4.5",
    expectIntents: ["earthquake", "news"],
    expectProviders: ["usgs-earthquake", "grovee-news"],
    expectLiveWorldLayers: ["earthquake"],
    expectRssTerms: ["earthquake"],
    expectMinMagnitude: 4.5,
    verifyHe: "אנגלית · סינון יפן ב-USGS place",
  },
  {
    id: "LD-AV01",
    category: "תעופה",
    userQuery: "כמה מטוסים נמצאים כרגע מעל ישראל?",
    expectIntents: ["aviation"],
    expectProviders: ["adsb-aviation"],
    expectLiveWorldLayers: ["aviation"],
    expectRssTerms: [],
    verifyHe: "עולם חי ADS-B · snapshot fallback",
  },
  {
    id: "LD-SH01",
    category: "ספינות",
    userQuery: "כמה ספינות יש במפרץ חיפה?",
    expectIntents: ["ships"],
    expectProviders: ["ais-ships"],
    expectLiveWorldLayers: ["ships"],
    expectRssTerms: [],
    verifyHe: "AIS + bbox חיפה",
  },
  {
    id: "LD-ISS01",
    category: "חלל",
    userQuery: "איפה תחנת החלל הבינלאומית עכשיו?",
    expectIntents: ["satellite"],
    expectProviders: ["iss-tracker"],
    expectLiveWorldLayers: ["iss"],
    expectRssTerms: [],
    verifyHe: "ISS בעולם חי · קו רוחב/אורך",
  },
  {
    id: "LD-X01",
    category: "הצלבה",
    userQuery: "האם יש סופה פעילה באירופה?",
    expectIntents: ["disaster", "weather"],
    expectProviders: ["gdacs-disasters", "open-meteo"],
    expectLiveWorldLayers: [],
    expectRssTerms: [],
    verifyHe: "GDACS + מזג · לא רעידות",
  },
];
