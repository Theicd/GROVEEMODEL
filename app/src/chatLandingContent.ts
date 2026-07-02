export type LandingCategory =
  | "write"
  | "explain"
  | "summarize"
  | "translate"
  | "code"
  | "image"
  | "games"
  | "globe"
  | "search"
  | "ideas"
  | "plan"
  | "rewrite"
  | "learn"
  | "camera"
  | "think";

export type LandingSuggestion = {
  icon: string;
  label: string;
  prompt: string;
  category: LandingCategory;
};

export const LANDING_CATEGORY_LABELS: Record<LandingCategory, string> = {
  write: "כתיבה ועריכה",
  explain: "הסברים ולמידה",
  summarize: "סיכום וארגון",
  translate: "תרגום ושפות",
  code: "קוד ופיתוח",
  image: "תמונות וקבצים",
  games: "משחקים בממשק",
  globe: "מפה וגלובוס",
  search: "חיפוש מידע",
  ideas: "רעיונות ויצירתיות",
  plan: "תכנון וארגון",
  rewrite: "שיפור ניסוח",
  learn: "למידה והבנה",
  camera: "מצב מצלמה",
  think: "חשיבה מעמיקה",
};

export const LANDING_HEADLINES = [
  "במה אוכל לעזור לך?",
  "שאל אותי כל דבר",
  "מה תרצה לעשות היום?",
  "הקלד שאלה או בחר הצעה",
  "איך אוכל לסייע לך?",
  "3 הצעות — מתחלפות כל 10 שניות",
] as const;

/** Basic interface prompts — labels are at least 3 words. */
export const LANDING_CAPABILITY_CHIPS: LandingSuggestion[] = [
  // כתיבה
  { icon: "✍️", label: "עזור לי לכתוב", prompt: "עזור לי לכתוב הודעה קצרה ומנומסת", category: "write" },
  { icon: "📧", label: "נסח מייל מקצועי", prompt: "נסח לי מייל מקצועי וברור ללקוח", category: "write" },
  { icon: "📝", label: "כתוב פוסט קצר", prompt: "כתוב לי פוסט קצר לרשתות חברתיות", category: "write" },
  { icon: "💼", label: "נסח הודעה לעבודה", prompt: "נסח לי הודעה מקצועית לצוות בעבודה", category: "write" },

  // הסבר
  { icon: "💡", label: "הסבר מושג בפשטות", prompt: "הסבר לי מושג מורכב בצורה פשוטה וברורה", category: "explain" },
  { icon: "🧠", label: "הסבר כמו לילד", prompt: "הסבר לי נושא טכני כאילו אני בן עשר", category: "explain" },
  { icon: "❓", label: "ענה על שאלה שלי", prompt: "ענה לי על שאלה כללית בצורה מסודרת", category: "explain" },
  { icon: "📚", label: "הסבר נושא חדש לי", prompt: "הסבר לי נושא חדש מהבסיס, צעד אחר צעד", category: "explain" },

  // סיכום
  { icon: "📋", label: "סכם טקסט ארוך", prompt: "סכם לי טקסט ארוך לנקודות עיקריות", category: "summarize" },
  { icon: "🗒️", label: "הפק רשימת נקודות", prompt: "הפק לי רשימת נקודות מטקסט שאשלח", category: "summarize" },
  { icon: "⏱️", label: "סכם בשלוש נקודות", prompt: "סכם לי נושא בשלוש נקודות עיקריות בלבד", category: "summarize" },

  // תרגום
  { icon: "🌐", label: "תרגם לי לאנגלית", prompt: "תרגם לי את הטקסט הבא לאנגלית", category: "translate" },
  { icon: "🇮🇱", label: "תרגם לי לעברית", prompt: "תרגם לי את הטקסט הבא לעברית", category: "translate" },
  { icon: "🔤", label: "שפר את התרגום שלי", prompt: "שפר את התרגום שלי לאנגלית מקצועית", category: "translate" },

  // קוד
  { icon: "💻", label: "כתוב קוד פשוט", prompt: "כתוב לי קוד פשוט עם הסבר קצר", category: "code" },
  { icon: "🐛", label: "עזור לי לתקן", prompt: "עזור לי לתקן שגיאה בקוד שאשלח", category: "code" },
  { icon: "⚙️", label: "הסבר קטע קוד", prompt: "הסבר לי מה עושה קטע הקוד הבא", category: "code" },
  { icon: "🧩", label: "הצע פתרון לבעיה", prompt: "הצע לי פתרון קוד לבעיה שאתאר", category: "code" },

  // תמונות וקבצים
  { icon: "🎨", label: "תאר וצור תמונה", prompt: "תאר לי נוסח חייזר ירוק על מאדים", category: "image" },
  { icon: "🖼️", label: "צור תמונה מהתיאור", prompt: "צור מזה תמונה", category: "image" },
  { icon: "🖼️", label: "נתח תמונה שצירפתי", prompt: "נתח את התמונה שצירפתי ותאר מה רואים בה", category: "image" },
  { icon: "📄", label: "סכם קובץ שצירפתי", prompt: "סכם לי את הקובץ שצירפתי לשיחה", category: "image" },
  { icon: "🔍", label: "מצא פרטים בתמונה", prompt: "מצא פרטים חשובים בתמונה שצירפתי", category: "image" },

  // משחקים
  { icon: "👾", label: "פתח משחק קלאסי", prompt: "פתח לי משחק קלאסי מהרשימה", category: "games" },
  { icon: "🎮", label: "המלץ על משחק מהיר", prompt: "המלץ לי על משחק מהיר וכיף לשחק", category: "games" },
  { icon: "🕹️", label: "מה יש במשחקים מומלצים", prompt: "מה יש במשחקים המומלצים בממשק?", category: "games" },

  // מפה
  { icon: "🗺️", label: "הצג עיר על המפה", prompt: "הצג לי עיר על המפה בגלובוס", category: "globe" },
  { icon: "📍", label: "מצא מקום על המפה", prompt: "מצא לי מקום ספציפי על המפה", category: "globe" },
  { icon: "🌍", label: "הראה לי את העולם", prompt: "הראה לי נקודת עניין על גלובוס העולם", category: "globe" },

  // חיפוש
  { icon: "🔎", label: "חפש מידע עדכני", prompt: "חפש לי מידע עדכני על נושא שאבחר", category: "search" },
  { icon: "📰", label: "מה חדש בנושא", prompt: "מה חדש בנושא שאני שואל עליו?", category: "search" },
  { icon: "🎮", label: "חדשות משחקי מחשב", prompt: "חפש חדשות על משחקי מחשב וקונסולה?", category: "search" },
  { icon: "🔗", label: "מצא מקורות אמינים", prompt: "מצא לי מקורות אמינים לנושא שאשאל", category: "search" },

  // רעיונות
  { icon: "✨", label: "תן לי רעיונות חדשים", prompt: "תן לי רעיונות חדשים לפרויקט שאתאר", category: "ideas" },
  { icon: "🎯", label: "הצע כותרות לפוסט", prompt: "הצע לי כותרות מעניינות לפוסט בבלוג", category: "ideas" },
  { icon: "🌱", label: "רעיונות לשיפור יומי", prompt: "תן לי רעיונות פשוטים לשיפור היום שלי", category: "ideas" },

  // תכנון
  { icon: "📅", label: "עזור לי לתכנן יום", prompt: "עזור לי לתכנן יום עבודה מסודר", category: "plan" },
  { icon: "✅", label: "בנה רשימת משימות", prompt: "בנה לי רשימת משימות לשבוע הקרוב", category: "plan" },
  { icon: "🗂️", label: "ארגן לי את הנושא", prompt: "ארגן לי נושא לשלבים ברורים ופשוטים", category: "plan" },

  // שיפור ניסוח
  { icon: "✏️", label: "שפר את הניסוח שלי", prompt: "שפר את הניסוח של הטקסט שאשלח", category: "rewrite" },
  { icon: "🎩", label: "הפוך לניסוח מקצועי", prompt: "הפוך את הטקסט שלי לניסוח מקצועי יותר", category: "rewrite" },
  { icon: "😊", label: "הפוך לניסוח ידידותי", prompt: "הפוך את הטקסט שלי לניסוח ידידותי וחם", category: "rewrite" },

  // למידה
  { icon: "📖", label: "למד אותי נושא חדש", prompt: "למד אותי נושא חדש מההתחלה", category: "learn" },
  { icon: "🧪", label: "תן דוגמה מעשית", prompt: "תן לי דוגמה מעשית לנושא שאשאל", category: "learn" },
  { icon: "📝", label: "בחן אותי בנושא", prompt: "בחן אותי בנושא שאבחר עם שאלות קצרות", category: "learn" },

  // מצלמה וחשיבה
  { icon: "🎥", label: "איך עובד מצב מצלמה", prompt: "איך עובד מצב המצלמה בממשק?", category: "camera" },
  { icon: "👁️", label: "מה המצלמה רואה עכשיו", prompt: "מה המצלמה רואה כרגע במצב חי?", category: "camera" },
  { icon: "💭", label: "הסבר עם חשיבה מפורטת", prompt: "ענה לי עם חשיבה מפורטת לפני התשובה", category: "think" },
  { icon: "🧩", label: "פתור בעיה צעד אחר", prompt: "פתור בעיה צעד אחר צעד עם חשיבה", category: "think" },
];

/** @deprecated Use LANDING_CAPABILITY_CHIPS */
export const LANDING_SUGGESTION_SETS: LandingSuggestion[][] = [LANDING_CAPABILITY_CHIPS.slice(0, 3)];

export const LANDING_ROTATION_MS = 10_000;

export function labelWordCount(label: string): number {
  return label.trim().split(/\s+/).filter(Boolean).length;
}

/** Pick N suggestions from different categories when possible. */
export function pickRotatingLandingSuggestions(count = 3): LandingSuggestion[] {
  const pool = LANDING_CAPABILITY_CHIPS.filter((item) => labelWordCount(item.label) >= 3);
  const byCat = new Map<LandingCategory, LandingSuggestion[]>();
  for (const item of pool) {
    const list = byCat.get(item.category) ?? [];
    list.push(item);
    byCat.set(item.category, list);
  }
  const categories = [...byCat.keys()].sort(() => Math.random() - 0.5);
  const picked: LandingSuggestion[] = [];
  for (const cat of categories) {
    if (picked.length >= count) break;
    const catPool = byCat.get(cat)!;
    picked.push(catPool[Math.floor(Math.random() * catPool.length)]);
  }
  const rest = pool.filter((x) => !picked.some((p) => p.prompt === x.prompt));
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
