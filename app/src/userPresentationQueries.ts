/** User-defined presentation QA list — category label + prompt sent to chat. */
export type UserPresentationQuery = {
  id: string;
  group: "basic" | "cross" | "natural";
  category: string;
  prompt: string;
};

export const USER_PRESENTATION_QUERIES: UserPresentationQuery[] = [
  { id: "B01", group: "basic", category: "חדשות", prompt: "מה הכותרת הראשית בעולם כרגע?" },
  { id: "B02", group: "basic", category: "פוליטיקה", prompt: "מי ראש ממשלת בריטניה?" },
  { id: "B03", group: "basic", category: "מטבעות", prompt: "מה שער הדולר מול השקל כרגע?" },
  { id: "B04", group: "basic", category: "שוק ההון", prompt: "מה ערך מדד S&P 500 כרגע?" },
  { id: "B05", group: "basic", category: "קריפטו", prompt: "מה מחיר הביטקוין כרגע?" },
  { id: "B06", group: "basic", category: "מזג אוויר", prompt: "מה הטמפרטורה כרגע בטוקיו?" },
  { id: "B07", group: "basic", category: "רעידות אדמה", prompt: "מה הייתה רעידת האדמה האחרונה מעל 5.0?" },
  { id: "B08", group: "basic", category: "גלים", prompt: "מה גובה הגלים כרגע מול חופי חיפה?" },
  { id: "B09", group: "basic", category: "תעופה", prompt: "כמה מטוסים נמצאים כרגע מעל ישראל?" },
  { id: "B10", group: "basic", category: "תעופה צבאית", prompt: "כמה מטוסי AWACS פעילים כרגע?" },
  { id: "B11", group: "basic", category: "אוניות", prompt: "כמה אוניות נמצאות כרגע בתעלת סואץ?" },
  { id: "B12", group: "basic", category: "חלל", prompt: "היכן נמצאת תחנת החלל הבינלאומית כרגע?" },
  { id: "B13", group: "basic", category: "לוויינים", prompt: "כמה לווייני Starlink פעילים כרגע?" },
  { id: "B14", group: "basic", category: "גיאוגרפיה", prompt: "מהי עיר הבירה של קזחסטן?" },
  {
    id: "B15",
    group: "basic",
    category: "תחבורה ציבורית",
    prompt: "מה תחנת הרכבת הקרובה ביותר לשדה התעופה BER בברלין?",
  },
  { id: "B16", group: "basic", category: "GitHub", prompt: "מהו הפרויקט הפופולרי ביותר היום ב-GitHub?" },
  { id: "B17", group: "basic", category: "Hugging Face", prompt: "מהו המודל החדש הפופולרי ביותר השבוע?" },
  { id: "B18", group: "basic", category: "משחקים", prompt: "האם Doom זמין בארכיון המשחקים?" },
  { id: "B19", group: "basic", category: "אסונות טבע", prompt: "אילו סופות טרופיות פעילות כרגע?" },
  { id: "C01", group: "cross", category: "הצלבה", prompt: "האם יש כרגע מטוסים מעל אזור שבו יש סופה פעילה?" },
  { id: "C02", group: "cross", category: "הצלבה", prompt: "האם יש אוניות באזור שבו יש אזהרת מזג אוויר?" },
  { id: "C03", group: "cross", category: "הצלבה", prompt: "האם יש רעידת אדמה פעילה באזור שבו יש ערים גדולות?" },
  { id: "C04", group: "cross", category: "הצלבה", prompt: "האם תחנת החלל נמצאת כרגע מעל יבשה או ים?" },
  { id: "C05", group: "cross", category: "הצלבה", prompt: "האם יש מטוסים מעל אזור שבו מתרחשת שריפה גדולה?" },
  {
    id: "C06",
    group: "cross",
    category: "הצלבה",
    prompt: "האם יש אוניות ליד מדינה שמופיעה בכותרות החדשות היום?",
  },
  {
    id: "C07",
    group: "cross",
    category: "הצלבה",
    prompt: "האם יש טיסות מעל אזור שבו התרחשה רעידת אדמה ב-24 השעות האחרונות?",
  },
  { id: "C08", group: "cross", category: "הצלבה", prompt: "האם יש סופה פעילה באזור שבו נמצאים נמלים מרכזיים?" },
  {
    id: "C09",
    group: "cross",
    category: "הצלבה",
    prompt: "האם יש פעילות תעופה חריגה באזור שבו יש אירוע חדשותי משמעותי?",
  },
  {
    id: "C10",
    group: "cross",
    category: "הצלבה",
    prompt: "האם יש לווייני Starlink מעל המדינה שמוזכרת הכי הרבה בחדשות היום?",
  },
  { id: "C11", group: "cross", category: "הצלבה", prompt: "האם יש אוניות באזור שבו גובה הגלים חריג?" },
  { id: "C12", group: "cross", category: "הצלבה", prompt: "האם יש מטוסים באזור שבו קיימת התרעת צונאמי?" },
  {
    id: "C13",
    group: "cross",
    category: "הצלבה",
    prompt: "האם תחנת החלל עוברת מעל מדינה שחווה מזג אוויר קיצוני?",
  },
  {
    id: "C14",
    group: "cross",
    category: "הצלבה",
    prompt: "האם יש אוניות ליד אזור שבו התרחשה רעידת אדמה חזקה השבוע?",
  },
  {
    id: "C15",
    group: "cross",
    category: "הצלבה",
    prompt: "האם יש קשר בין סופה פעילה לבין שיבושים בתעבורה האווירית באזור?",
  },
  { id: "N01", group: "natural", category: "טבעי", prompt: "מה קורה כרגע באזור הכי עמוס בעולם?" },
  { id: "N02", group: "natural", category: "טבעי", prompt: "האם יש משהו חריג שמתרחש כרגע מעל ישראל?" },
  {
    id: "N03",
    group: "natural",
    category: "טבעי",
    prompt: "האם יש אזור בעולם שבו גם מזג האוויר וגם התנועה האווירית חריגים?",
  },
  {
    id: "N04",
    group: "natural",
    category: "טבעי",
    prompt: "האם יש כרגע מקום בעולם שבו מתרחשים כמה אירועים משמעותיים במקביל?",
  },
  {
    id: "N05",
    group: "natural",
    category: "טבעי",
    prompt: "איזה אזור בעולם נראה הכי פעיל כרגע לפי כל הנתונים שיש לך?",
  },
];
