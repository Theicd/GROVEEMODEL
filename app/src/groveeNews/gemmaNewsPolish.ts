import { cleanRephrasedBrief } from "./articleSummaryDisplay";

export const GEMMA_NEWS_POLISH_SYSTEM = `אתה עורך חדשות בעברית. אתה מקבל טקסט מסוכם באנגלית (הערות מ-Qwen).
המשימה שלך: לנסח מחדש בצורה ברורה, טבעית וקריאה בעברית — לא לתרגם מילה-במילה, אלא לנסח כמו כתבה בעיתון.

הפלט חייב להיות רק הטקסט הסופי למשתמש, בפורמט:

כותרת: <כותרת עברית ברורה בשורה אחת>

תקציר: <שתיים-שלוש משפטים קצרים ושוטפים בעברית>

חוקים:
- עברית בלבד בפלט
- בלי אנגלית, בלי נקודות, בלי מספור עובדות
- בלי הערות מטא או הוראות
- אל תחזור על החוקים האלה`;

export function buildGemmaNewsPolishUserPrompt(qwenNotes: string, articleTitle: string): string {
  const notes = qwenNotes.trim();
  const title = articleTitle.trim() || "כתבה";
  return `כותרת הכתבה: ${title}

הטקסט שקיבלת לניסוח מחדש:
${notes}

נסח בעברית ברורה: כותרת + תקציר.`;
}

const PROMPT_LEAK_PATTERNS = [
  /\(One Clear Headline Line\)/gi,
  /\(Two or Three Short Fluent Sentences\)/gi,
  /^Rules:\s*$/gim,
  /^-\s*No Bullet Points.*$/gim,
  /^-\s*No Numbers.*$/gim,
  /^-\s*Hebrew only.*$/gim,
  /^-\s*No English.*$/gim,
  /^-\s*Do not repeat.*$/gim,
];

/** Strip Gemma/Qwen prompt echoes and normalize to כותרת/תקציר. */
export function cleanGemmaNewsPolishOutput(raw: string): string {
  let text = raw.replace(/\r\n/g, "\n").trim();
  for (const pattern of PROMPT_LEAK_PATTERNS) {
    text = text.replace(pattern, " ");
  }
  text = text.replace(/\n{3,}/g, "\n\n").trim();
  return cleanRephrasedBrief(text, "he");
}
