/** Free live-data providers wired into chat search (no Google, no paid keys). */

export type SearchProviderInfo = {
  id: string;
  icon: string;
  labelHe: string;
  examplesHe: string[];
};

export const CONNECTED_SEARCH_PROVIDERS: SearchProviderInfo[] = [
  { id: "open-meteo", icon: "🌤", labelHe: "Open-Meteo — מזג אוויר", examplesHe: ["מה הטמפרטורה בפריז?", "תחזית למחר במדריד"] },
  { id: "world-time", icon: "🕐", labelHe: "TimeAPI — שעון עולמי", examplesHe: ["מה השעה בטוקיו?", "הפרש שעות ישראל–לונדון"] },
  { id: "usgs-earthquake", icon: "🌋", labelHe: "USGS — רעידות אדמה", examplesHe: ["רעידות ב-24 שעות"] },
  { id: "adsb-aviation", icon: "✈", labelHe: "ADS-B — מטוסים חיים", examplesHe: ["כמה מטוסים מעל ישראל?", "מטוסים באוויר בעולם"] },
  { id: "ais-ships", icon: "⛴", labelHe: "AIS Digitraffic + עולם חי — ספינות", examplesHe: ["כמה כלי שייט במפרץ חיפה", "ספינות בתעלת סואץ"] },
  { id: "osm-overpass-marine", icon: "⚓", labelHe: "OpenStreetMap Overpass — תשתיות ימיות", examplesHe: ["כמה מצופים במפרץ חיפה", "מגדלורים ליד חיפה"] },
  { id: "celestrak", icon: "🛰", labelHe: "CelesTrak — קטalog לוויינים", examplesHe: ["כמה לוויינים פעילים?"] },
  { id: "iss-tracker", icon: "🛰", labelHe: "WhereTheISS — תחנת החלל", examplesHe: ["איפה ה-ISS עכשיו?"] },
  { id: "spacex-launches", icon: "🚀", labelHe: "Launch Library — שיגורי SpaceX", examplesHe: ["מתי השיגור הבא של SpaceX?"] },
  { id: "hacker-news", icon: "📰", labelHe: "Hacker News — כותרות טech", examplesHe: ["פוסט פופולרי ב-HN", "חדשות AI"] },
  { id: "github", icon: "🐙", labelHe: "GitHub — מאגרים", examplesHe: ["חפש מודלים ל-OCR", "פרויקטים WebGPU"] },
  { id: "huggingface", icon: "🤗", labelHe: "Hugging Face — מודלים", examplesHe: ["מודלים לעברית", "datasets OCR"] },
  { id: "nominatim-places", icon: "📍", labelHe: "OpenStreetMap — מקומות", examplesHe: ["מלון ליד הלובר", "תחנת רכבת ליד אייפel"] },
  { id: "rest-countries", icon: "🌍", labelHe: "REST Countries", examplesHe: ["מה המטבע של ברזיל?", "אוכלוסיית קנדה"] },
  { id: "frankfurter-fx", icon: "💱", labelHe: "Frankfurter — שערי מט\"ח", examplesHe: ["כמה יורו ב-1000 שקל?"] },
  { id: "coingecko", icon: "₿", labelHe: "CoinGecko — קריפטו", examplesHe: ["מחיר ביטקוין עכשיו"] },
  { id: "yahoo-finance", icon: "📈", labelHe: "Yahoo Finance — מניות / מדדים / זהב / נפט", examplesHe: ["מצב S&P 500", "מחיר זהב", "Brent"] },
  { id: "wikidata-gov", icon: "🏛", labelHe: "Wikidata — ממשל וראשי מדינה", examplesHe: ["מי ראש הממשלה של בריטניה?"] },
  { id: "news-rss", icon: "📰", labelHe: "BBC · CNN · Reuters · Guardian — RSS", examplesHe: ["כותרות עולם"] },
];

export const LIVE_WORLD_LAYERS_HE =
  "עולם חי (🌐): מטוסים · ספינות · מצופים (OpenSeaMap) · לוויינים · רעידות · מזג אוויר";
