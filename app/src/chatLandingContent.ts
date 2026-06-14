export type LandingCategory =
  | "news"
  | "overview"
  | "earthquake"
  | "weather"
  | "marine"
  | "aviation"
  | "military"
  | "ships"
  | "marine-infra"
  | "space"
  | "globe"
  | "transit"
  | "github"
  | "huggingface"
  | "games"
  | "fusion"
  | "currency"
  | "country"
  | "stress";

export type LandingSuggestion = {
  icon: string;
  label: string;
  prompt: string;
  category: LandingCategory;
};

export const LANDING_CATEGORY_LABELS: Record<LandingCategory, string> = {
  news: "חדשות ואקטואליה",
  overview: "סקירת עולם",
  earthquake: "רעידות אדמה",
  weather: "מזג אוויר",
  marine: "ים וגלים",
  aviation: "תעופה",
  military: "תעופה צבאית",
  ships: "אוניות ונמלים",
  "marine-infra": "מצופים ותשתיות ים",
  space: "חלל ולוויינים",
  globe: "מפות וגלובוס",
  transit: "תחבורה ציבורית",
  github: "GitHub",
  huggingface: "Hugging Face",
  games: "משחקים קלאסיים",
  fusion: "שילוב מקורות",
  currency: "מטבעות",
  country: "מדינות ועובדות",
  stress: "בדיקות קצה",
};

export const LANDING_HEADLINES = [
  "מה תרצה לבדוק?",
  "שאל על נתונים חיים",
  "גלה מה קורה בעולם עכשיו",
  "נסה שאלה על מטוסים, אוניות או רעידות",
  "מה מעניין אותך היום?",
  "3 הצעות — מתחלפות כל 10 שניות",
] as const;

/** Organized capability pool — one representative prompt per data source / feature. */
export const LANDING_CAPABILITY_CHIPS: LandingSuggestion[] = [
  // חדשות ואקטואליה
  { icon: "📰", label: "כותרות עולם", prompt: "מה הכותרות החשובות בעולם כרגע?", category: "news" },
  { icon: "🇮🇱", label: "ישראל היום", prompt: "מה קורה בישראל היום?", category: "news" },
  { icon: "🌋", label: "אסונות טבע", prompt: "האם יש אסונות טבע פעילים כרגע?", category: "news" },
  { icon: "⛈️", label: "התרעות מזג", prompt: "אילו מדינות נמצאות תחת התרעות מזג אוויר?", category: "news" },
  { icon: "🔥", label: "שריפות", prompt: "מהי השריפה הגדולה ביותר הפעילה כרגע בעולם?", category: "news" },
  { icon: "🌀", label: "סופות", prompt: "אילו סופות טרופיות פעילות כרגע?", category: "news" },

  // סקירה (OpenAI style)
  { icon: "🌍", label: "מצב העולם", prompt: "תן לי סקירה של מצב העולם כרגע", category: "overview" },
  { icon: "⚡", label: "משהו חריג?", prompt: "האם יש משהו חריג שמתרחש כרגע?", category: "overview" },
  { icon: "📡", label: "24 שעות", prompt: "אילו אירועים חשובים התרחשו ב-24 השעות האחרונות?", category: "overview" },

  // רעידות
  { icon: "📊", label: "20 רעידות", prompt: "הצג את 20 רעידות האדמה האחרונות בעולם", category: "earthquake" },
  { icon: "💥", label: "רעידה חזקה", prompt: "מה הייתה רעידת האדמה החזקה ביותר השבוע?", category: "earthquake" },
  { icon: "🗾", label: "רעידה ביפן", prompt: "האם הייתה רעידת אדמה ליד יפן ב-48 השעות האחרונות?", category: "earthquake" },
  { icon: "📈", label: "M5+ החודש", prompt: "כמה רעידות מעל 5.0 התרחשו החודש?", category: "earthquake" },
  { icon: "🌐", label: "רעידות על גלובוס", prompt: "הצג רעידות אדמה על הגלובוס", category: "earthquake" },

  // מזג + ים
  { icon: "🌊", label: "גלים ת״א", prompt: "מה גובה הגלים מול חופי תל אביב?", category: "marine" },
  { icon: "🌤️", label: "רוח בפריז", prompt: "מה מהירות הרוח בפריז?", category: "weather" },
  { icon: "🧊", label: "טמפ׳ איסלנד", prompt: "מה הטמפרטורה באיסלנד עכשיו?", category: "weather" },
  { icon: "💨", label: "רוחות חזקות", prompt: "איזה אזור בעולם חווה את הרוחות החזקות ביותר כרגע?", category: "weather" },

  // תעופה
  { icon: "✈️", label: "מטוסים מעל ישראל", prompt: "כמה מטוסים נמצאים כרגע מעל ישראל?", category: "aviation" },
  { icon: "🇬🇧", label: "מטוסים לונדון", prompt: "כמה מטוסים מעל לונדון?", category: "aviation" },
  { icon: "🛬", label: "נחיתות נתב״ג", prompt: "אילו טיסות מתקרבות לנחיתה בנתב\"ג?", category: "aviation" },
  { icon: "🌊", label: "ים תיכון", prompt: "הצג את כל המטוסים מעל הים התיכון", category: "aviation" },

  // צבא
  { icon: "🛩️", label: "צבאי ים תיכון", prompt: "כמה מטוסים צבאיים מזוהים כרגע באזור הים התיכון?", category: "military" },
  { icon: "⛽", label: "תדלוק אוויר", prompt: "האם יש מטוסי תדלוק אווירי מעל אירופה?", category: "military" },

  // אוניות
  { icon: "⛴️", label: "ספינות חיפה", prompt: "כמה כלי שייט או אוניות יש במפרץ חיפה?", category: "ships" },
  { icon: "🚢", label: "סואץ", prompt: "כמה אוניות נמצאות במפרץ סואץ?", category: "ships" },
  { icon: "🛢️", label: "מכליות פרסי", prompt: "כמה מכליות נפט נמצאות במפרץ הפרסי?", category: "ships" },
  { icon: "🇮🇱", label: "אוניות ישראל", prompt: "אילו אוניות נמצאות ליד חופי ישראל?", category: "ships" },
  { icon: "🇳🇱", label: "רוטרדם", prompt: "הצג אוניות מכולה באזור רוטרדם", category: "ships" },

  // תשתיות ים
  { icon: "⚓", label: "מצופים חיפה", prompt: "כמה מצופים יש במפרץ חיפה?", category: "marine-infra" },

  // חלל
  { icon: "🛰️", label: "תחנת החלל", prompt: "איפה נמצאת תחנת החלל כרגע?", category: "space" },
  { icon: "🇮🇱", label: "ISS מעל ישראל", prompt: "מתי תחנת החלל תעבור מעל ישראל?", category: "space" },
  { icon: "🛸", label: "מסלול ISS", prompt: "הצג את מסלול ה-ISS על הגלובוס", category: "space" },

  // מפות
  { icon: "🗺️", label: "גרמניה", prompt: "הצג על המפה את גרמניה", category: "globe" },
  { icon: "🏛️", label: "ברלין", prompt: "התקרב לברלין", category: "globe" },
  { icon: "🗼", label: "פריז", prompt: "הצג את פריז", category: "globe" },
  { icon: "🏔️", label: "אוורסט", prompt: "הצג את הר האוורסט", category: "globe" },
  { icon: "🚢", label: "תעלת פנמה", prompt: "הצג את תעלת פנמה", category: "globe" },

  // תחבורה
  { icon: "🚆", label: "רכבת ברלין", prompt: "מהי תחנת הרכבת הקרובה ביותר לשדה התעופה בברלין?", category: "transit" },
  { icon: "📏", label: "מרחק י-ם–חיפה", prompt: 'כמה ק"מ בין ירושלים לחיפה?', category: "transit" },

  // GitHub / HF
  { icon: "🐙", label: "WebGPU", prompt: "חפש פרויקטים בנושא WebGPU", category: "github" },
  { icon: "🤖", label: "AI השבוע", prompt: "מצא פרויקטי AI שפורסמו השבוע", category: "github" },
  { icon: "⭐", label: "GitHub פופולרי", prompt: "מהם הפרויקטים הפופולריים ביותר היום?", category: "github" },
  { icon: "🎮", label: "Three.js", prompt: "מצא משחקים שנבנו עם Three.js", category: "github" },
  { icon: "🦙", label: "חלופות Ollama", prompt: "מצא חלופות ל-Ollama", category: "github" },
  { icon: "🤗", label: "מודלים חדשים", prompt: "מצא מודלים חדשים השבוע", category: "huggingface" },
  { icon: "👁️", label: "VLM", prompt: "מהם מודלי ה-VLM הפופולריים ביותר?", category: "huggingface" },
  { icon: "🌐", label: "מודלים לדפדפן", prompt: "מצא מודלים שמתאימים להרצה בדפדפן", category: "huggingface" },

  // משחקים
  { icon: "👾", label: "Doom", prompt: "שחק Doom", category: "games" },
  { icon: "🗡️", label: "Prince of Persia", prompt: "שחק Prince of Persia", category: "games" },
  { icon: "🎮", label: "DOS אסטרטגיה", prompt: "מצא משחקי DOS אסטרטגיה", category: "games" },

  // שילוב + stress
  { icon: "🔗", label: "סופה + מטוסים", prompt: "כמה מטוסים נמצאים מעל האזור שבו נמצאת כרגע הסופה הגדולה בעולם?", category: "fusion" },
  { icon: "🌊", label: "צונאמי + אוניות", prompt: "האם יש אוניות באזורי התרעת צונאמי?", category: "fusion" },
  { icon: "🌐", label: "תמונת מצב", prompt: "תן לי תמונת מצב מלאה של כדור הארץ כרגע", category: "stress" },
  { icon: "🎯", label: "20 חריגים", prompt: "מה 20 האירועים החריגים ביותר שמתרחשים עכשיו?", category: "stress" },

  // עזר
  { icon: "💱", label: "דולר לשקל", prompt: "כמה שקלים שווים 100 דולר?", category: "currency" },
  { icon: "🌍", label: "תושבי קנדה", prompt: "כמה תושבים יש בקנדה?", category: "country" },
  { icon: "₿", label: "ביטקוין", prompt: "מה מחיר הביטקוין עכשיו?", category: "currency" },
];

/** @deprecated Use LANDING_CAPABILITY_CHIPS */
export const LANDING_SUGGESTION_SETS: LandingSuggestion[][] = [LANDING_CAPABILITY_CHIPS.slice(0, 3)];

export const LANDING_ROTATION_MS = 10_000;

/** Pick N suggestions from different categories when possible. */
export function pickRotatingLandingSuggestions(count = 3): LandingSuggestion[] {
  const byCat = new Map<LandingCategory, LandingSuggestion[]>();
  for (const item of LANDING_CAPABILITY_CHIPS) {
    const list = byCat.get(item.category) ?? [];
    list.push(item);
    byCat.set(item.category, list);
  }
  const categories = [...byCat.keys()].sort(() => Math.random() - 0.5);
  const picked: LandingSuggestion[] = [];
  for (const cat of categories) {
    if (picked.length >= count) break;
    const pool = byCat.get(cat)!;
    picked.push(pool[Math.floor(Math.random() * pool.length)]);
  }
  const rest = LANDING_CAPABILITY_CHIPS.filter((x) => !picked.some((p) => p.prompt === x.prompt));
  while (picked.length < count && rest.length) {
    const i = Math.floor(Math.random() * rest.length);
    picked.push(rest.splice(i, 1)[0]);
  }
  return picked.slice(0, count);
}

/** @deprecated alias */
export function pickRandomLandingSuggestions(count = 3): LandingSuggestion[] {
  return pickRotatingLandingSuggestions(count);
}

export function pickLandingHeadline(): string {
  return LANDING_HEADLINES[Math.floor(Math.random() * LANDING_HEADLINES.length)];
}
