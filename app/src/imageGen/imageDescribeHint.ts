/** System append for LLM turns that describe a scene before cloud image generation. */

export function imageDescribeSystemAppend(uiLang: "he" | "en"): string {
  return uiLang === "he"
    ? "המשתמש מבקש תיאור ויזואלי ליצירת תמונה. תאר את הסצנה ב-2-3 משפטים קצרים ועשירים. דמיון ופנטזיה מותרים — אל תסרב בטענה שזה בלתי אפשרי."
    : "The user wants a visual description for image generation. Describe the scene in 2-3 vivid short sentences. Fantasy and surreal subjects are allowed — do not refuse as impossible.";
}
