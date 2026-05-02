/** Pure helpers for routing user text (used by App and unit tests). */

const stripModelOutputNoise = (input: string): string => {
  const cleaned = input
    .replace(/\r/g, "")
    .split("\n")
    .filter((line) => !/^\s*(User|Assistant|System)\s*:/i.test(line))
    .join("\n")
    .replace(/^["']+|["']+$/g, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();

  const lines = cleaned
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  const deduped: string[] = [];
  for (const line of lines) {
    if (deduped[deduped.length - 1] !== line) deduped.push(line);
  }
  const normalized = deduped.join("\n");
  if (normalized.length) return normalized;
  const raw = input.trim();
  return raw.length ? raw.slice(0, 2000) : "";
};

/** Single-line English prompt for Pollinations or local SD-Turbo. */
export const cleanEnglishImagePrompt = (raw: string): string => {
  let t = stripModelOutputNoise(raw).replace(/^[\s\-*]+/, "");
  t = t.split("\n")[0] ?? t;
  t = t.replace(/^["']|["']$/g, "").trim();
  return t.slice(0, 500) || "high quality detailed scene";
};

export const isSimpleGreeting = (text: string): boolean => {
  const normalized = text.trim().toLowerCase();
  return /^(hi|hey|hello|shalom|שלום|היי|הי)$/.test(normalized);
};

export const isRtlText = (text: string): boolean => /[\u0590-\u05FF]/.test(text);

export const isCodeRequest = (text: string): boolean => {
  const t = text.toLowerCase();
  const he = /קוד|פונקציה|דיבאג|שגיאה|תוכנית|סקריפט|html|css|פייתון|ג'אווהסקריפט|טייפסקריפט/.test(text);
  return (
    he ||
    /(code|debug|bug|stack trace|typescript|javascript|python|function|class|compile|error|implement|refactor|api)\b/.test(t)
  );
};

export const isImageGenerationRequest = (text: string): boolean => {
  const t = text.toLowerCase();
  const he =
    /צור תמונה|תמונה של|הפק תמונה|ייצר תמונה|צייר|איור של|תאר תמונה ש|בנה תמונה/.test(text);
  return he || /(create|generate|draw|make)\s+(an?\s+)?image|text-to-image|image of\b/.test(t);
};
