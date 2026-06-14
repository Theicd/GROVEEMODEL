import type { SearchIntent } from "./webSearch/types";

export type CapabilityTier =
  | "search-live"
  | "search-partial"
  | "search-weak"
  | "ui-globe"
  | "ui-game"
  | "llm-synthesis"
  | "unsupported";

export type CapabilityProbe = {
  id: string;
  category: string;
  query: string;
  tier: CapabilityTier;
  expectIntents?: SearchIntent[];
  notesHe?: string;
};

/** Full deduplicated probe suite from user capability list + landing chips. */
export const CAPABILITY_PROBE_QUERIES: CapabilityProbe[] = [
  // חדשות ואקטואליה
  { id: "N01", category: "חדשות", query: "מה הכותרות החשובות בעולם כרגע?", tier: "search-partial", expectIntents: ["news"] },
  { id: "N02", category: "חדשות", query: "מה קורה בישראל היום?", tier: "search-partial", expectIntents: ["news"] },
  { id: "N03", category: "חדשות", query: "האם יש אסונות טבע פעילים כרגע?", tier: "search-weak", notesHe: "GDACS/EONET — partial via reality" },
  { id: "N04", category: "חדשות", query: "אילו מדינות נמצאות תחת התרעות מזג אוויר?", tier: "search-weak" },
  { id: "N05", category: "חדשות", query: "האם התרחשו רעידות אדמה משמעותיות ב-24 השעות האחרונות?", tier: "search-live", expectIntents: ["earthquake"] },
  { id: "N06", category: "חדשות", query: "מהי השריפה הגדולה ביותר הפעילה כרגע בעולם?", tier: "search-weak", notesHe: "FIRMS — no ranked API" },
  { id: "N07", category: "חדשות", query: "אילו סופות טרופיות פעילות כרגע?", tier: "search-weak" },

  // רעידות
  { id: "E01", category: "רעידות", query: "הצג את 20 רעידות האדמה האחרונות בעולם", tier: "search-live", expectIntents: ["earthquake"] },
  { id: "E02", category: "רעידות", query: "מה הייתה רעידת האדמה החזקה ביותר השבוע?", tier: "search-live", expectIntents: ["earthquake"] },
  { id: "E03", category: "רעידות", query: "הצג אותן על הגלובוס", tier: "ui-globe", notesHe: "תלוי הקשר — globe command" },
  { id: "E04", category: "רעידות", query: "האם הייתה רעידת אדמה ליד יפן ב-48 השעות האחרונות?", tier: "search-live", expectIntents: ["earthquake"] },
  { id: "E05", category: "רעידות", query: "כמה רעידות מעל 5.0 התרחשו החודש?", tier: "search-weak", notesHe: "USGS feed — לא סינון חודש מדויק" },

  // מזג + ים
  { id: "W01", category: "ים", query: "מה גובה הגלים מול חופי תל אביב?", tier: "search-live", expectIntents: ["marine"] },
  { id: "W02", category: "ים", query: "איפה נמצאים הגלים הגבוהים ביותר כרגע בעולם?", tier: "search-weak" },
  { id: "W03", category: "מזג", query: "מה הטמפרטורה באיסלנד עכשיו?", tier: "search-live", expectIntents: ["weather"] },
  { id: "W04", category: "מזג", query: "איזה אזור בעולם חווה את הרוחות החזקות ביותר כרגע?", tier: "search-weak" },
  { id: "W05", category: "מזג", query: "הצג מפה של מזג האוויר באירופה", tier: "ui-globe" },

  // תעופה
  { id: "A01", category: "תעופה", query: "כמה מטוסים נמצאים כרגע מעל ישראל?", tier: "search-live", expectIntents: ["aviation"] },
  { id: "A02", category: "תעופה", query: "כמה מטוסים מעל לונדון?", tier: "search-weak", notesHe: "bbox לונדון — coverage ADSB" },
  { id: "A03", category: "תעופה", query: "אילו טיסות מתקרבות לנחיתה בנתב\"ג?", tier: "search-weak" },
  { id: "A04", category: "תעופה", query: "מהו שדה התעופה העמוס ביותר כרגע?", tier: "unsupported", notesHe: "אין API עמוסות real-time" },
  { id: "A05", category: "תעופה", query: "הצג את כל המטוסים מעל הים התיכון", tier: "ui-globe" },

  // צבא
  { id: "M01", category: "צבא", query: "כמה מטוסים צבאיים מזוהים כרגע באזור הים התיכון?", tier: "search-weak", expectIntents: ["aviation"] },
  { id: "M02", category: "צבא", query: "האם יש מטוסי תדלוק אווירי מעל אירופה?", tier: "search-weak" },
  { id: "M03", category: "צבא", query: "אילו מטוסי AWACS פעילים כרגע?", tier: "unsupported" },
  { id: "M04", category: "צבא", query: "הצג את מיקומם על המפה", tier: "ui-globe" },

  // אוניות
  { id: "S01", category: "אוניות", query: "כמה אוניות נמצאות במפרץ סואץ?", tier: "search-weak", expectIntents: ["ships"] },
  { id: "S02", category: "אוניות", query: "כמה מכליות נפט נמצאות במפרץ הפרסי?", tier: "search-weak", expectIntents: ["ships"] },
  { id: "S03", category: "אוניות", query: "אילו אוניות נמצאות ליד חופי ישראל?", tier: "search-live", expectIntents: ["ships"] },
  { id: "S04", category: "אוניות", query: "הצג אוניות מכולה באזור רוטרדם", tier: "search-live", expectIntents: ["ships"] },
  { id: "S05", category: "אוניות", query: "מהו הנמל העמוס ביותר כרגע?", tier: "unsupported" },
  { id: "S06", category: "אוניות", query: "כמה כלי שייט או אוניות יש במפרץ חיפה?", tier: "search-live", expectIntents: ["ships"] },
  { id: "S07", category: "תשתיות ים", query: "כמה מצופים יש במפרץ חיפה?", tier: "search-live", expectIntents: ["marine-infra"] },

  // חלל
  { id: "SP01", category: "חלל", query: "איפה נמצאת תחנת החלל כרגע?", tier: "search-live", expectIntents: ["satellite"] },
  { id: "SP02", category: "חלל", query: "מתי היא תעבור מעל ישראל?", tier: "search-live", expectIntents: ["satellite"] },
  { id: "SP03", category: "חלל", query: "אילו לווייני Starlink נמצאים מעל אירופה?", tier: "unsupported", notesHe: "אין Starlink feed" },
  { id: "SP04", category: "חלל", query: "כמה לוויינים פעילים במסלול נמוך?", tier: "search-weak" },
  { id: "SP05", category: "חלל", query: "הצג את מסלול ה-ISS על הגלובוס", tier: "ui-globe" },

  // מפות
  { id: "G01", category: "מפות", query: "איפה נמצאת גרמניה?", tier: "ui-globe" },
  { id: "G02", category: "מפות", query: "התקרב לברלין", tier: "ui-globe" },
  { id: "G03", category: "מפות", query: "הצג את פריז", tier: "ui-globe" },
  { id: "G04", category: "מפות", query: "הצג את הר האוורסט", tier: "ui-globe" },
  { id: "G05", category: "מפות", query: "הצג את תעלת פנמה", tier: "ui-globe" },
  { id: "G06", category: "מפות", query: "הצג את משולש ברמודה", tier: "ui-globe" },

  // תחבורה
  { id: "T01", category: "תחבורה", query: "מהי תחנת הרכבת הקרובה ביותר לשדה התעופה בברלין?", tier: "search-partial", expectIntents: ["places"] },
  { id: "T02", category: "תחבורה", query: "הצג אותה על המפה", tier: "ui-globe" },
  { id: "T03", category: "תחבורה", query: "כמה זמן נסיעה משם למרכז העיר?", tier: "search-partial", expectIntents: ["distance"] },
  { id: "T04", category: "תחבורה", query: "אילו קווי רכבת מגיעים לשם?", tier: "unsupported", notesHe: "GTFS לא מחובר" },

  // GitHub
  { id: "GH01", category: "GitHub", query: "מצא פרויקטי WebGPU חדשים", tier: "search-live", expectIntents: ["github"] },
  { id: "GH02", category: "GitHub", query: "מצא פרויקטי AI שפורסמו השבוע", tier: "search-partial", expectIntents: ["github"] },
  { id: "GH03", category: "GitHub", query: "מהם הפרויקטים הפופולריים ביותר היום?", tier: "search-live", expectIntents: ["github"] },
  { id: "GH04", category: "GitHub", query: "מצא משחקים שנבנו עם Three.js", tier: "search-live", expectIntents: ["github"] },
  { id: "GH05", category: "GitHub", query: "מצא חלופות ל-Ollama", tier: "search-live", expectIntents: ["github"] },

  // Hugging Face
  { id: "HF01", category: "Hugging Face", query: "מצא מודלים חדשים השבוע", tier: "search-partial", expectIntents: ["huggingface"] },
  { id: "HF02", category: "Hugging Face", query: "מהם מודלי ה-VLM הפופולריים ביותר?", tier: "search-live", expectIntents: ["huggingface"] },
  { id: "HF03", category: "Hugging Face", query: "מצא מודלים לזיהוי אובייקטים", tier: "search-live", expectIntents: ["huggingface"] },
  { id: "HF04", category: "Hugging Face", query: "מצא מודלים לזיהוי תנוחות גוף", tier: "search-live", expectIntents: ["huggingface"] },
  { id: "HF05", category: "Hugging Face", query: "מצא מודלים ל-WebGPU", tier: "search-weak", expectIntents: ["huggingface"] },
  { id: "HF06", category: "Hugging Face", query: "מצא מודלים שמתאימים להרצה בדפדפן", tier: "search-live", expectIntents: ["huggingface"] },

  // משחקים
  { id: "GM01", category: "משחקים", query: "שחק Doom", tier: "ui-game" },
  { id: "GM02", category: "משחקים", query: "שחק Doom II", tier: "ui-game" },
  { id: "GM03", category: "משחקים", query: "שחק Dune II", tier: "ui-game" },
  { id: "GM04", category: "משחקים", query: "שחק Prince of Persia", tier: "ui-game" },
  { id: "GM05", category: "משחקים", query: "שחק Wolfenstein 3D", tier: "ui-game" },
  { id: "GM06", category: "משחקים", query: "מצא משחקי DOS אסטרטגיה", tier: "ui-game" },
  { id: "GM07", category: "משחקים", query: "מצא משחקי SEGA משנות ה-90", tier: "ui-game" },

  // שילוב
  { id: "F01", category: "שילוב", query: "כמה מטוסים נמצאים מעל האזור שבו נמצאת כרגע הסופה הגדולה בעולם?", tier: "llm-synthesis" },
  { id: "F02", category: "שילוב", query: "האם יש אוניות באזורי התרעת צונאמי?", tier: "llm-synthesis" },
  { id: "F03", category: "שילוב", query: "האם תחנת החלל נמצאת כרגע מעל מדינה שבה יש סערה משמעותית?", tier: "llm-synthesis" },
  { id: "F04", category: "שילוב", query: "הצג את כל רעידות האדמה שהתרחשו בטווח של 500 ק\"מ מנתיבי שיט ראשיים", tier: "llm-synthesis" },
  { id: "F05", category: "שילוב", query: "אילו שדות תעופה נמצאים במסלול של סופת הוריקן פעילה?", tier: "llm-synthesis" },

  // OpenAI style
  { id: "O01", category: "סקירה", query: "מה הדברים המעניינים שקורים בעולם עכשיו?", tier: "llm-synthesis" },
  { id: "O02", category: "סקירה", query: "האם יש משהו חריג שמתרחש כרגע?", tier: "llm-synthesis" },
  { id: "O03", category: "סקירה", query: "אילו אירועים חשובים התרחשו ב-24 השעות האחרונות?", tier: "llm-synthesis" },
  { id: "O04", category: "סקירה", query: "תן לי סקירה של מצב העולם כרגע", tier: "llm-synthesis" },
  { id: "O05", category: "סקירה", query: "הצג לי את המקומות הפעילים ביותר על פני כדור הארץ כרגע", tier: "llm-synthesis" },
  { id: "O06", category: "סקירה", query: "מה קורה עכשיו בחלל?", tier: "search-partial", expectIntents: ["satellite"] },
  { id: "O07", category: "סקירה", query: "מה קורה עכשיו באוקיינוסים?", tier: "search-partial", expectIntents: ["marine", "ships"] },
  { id: "O08", category: "סקירה", query: "מה קורה עכשיו בשמי אירופה?", tier: "search-weak", expectIntents: ["aviation"] },

  // Stress
  { id: "ST01", category: "stress", query: "תן לי תמונת מצב מלאה של כדור הארץ כרגע", tier: "llm-synthesis" },
  { id: "ST02", category: "stress", query: "מה 20 האירועים החריגים ביותר שמתרחשים עכשיו?", tier: "llm-synthesis" },
  { id: "ST03", category: "stress", query: "הצג על הגלובוס בו זמנית מטוסים, אוניות, רעידות אדמה, סופות ושריפות", tier: "ui-globe" },
  { id: "ST04", category: "stress", query: "סכם את כל ההתראות הפעילות בעולם", tier: "llm-synthesis" },
];
