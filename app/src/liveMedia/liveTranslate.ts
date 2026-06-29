import { proxyAwareFetch } from "../webSearch/proxyFetch";

export const CAPTION_TARGET_NONE = "none";

/** Lightweight client-side translation via Google translate gtx endpoint. */
export async function translateLiveCaption(text: string, targetLang: string): Promise<string> {
  const q = text.trim();
  if (!q) return "";
  if (!targetLang || targetLang === CAPTION_TARGET_NONE) return q;
  const tl = targetLang.trim().slice(0, 5) || "he";
  const url = `https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=${encodeURIComponent(tl)}&dt=t&q=${encodeURIComponent(q)}`;

  let response: Response;
  try {
    response = await fetch(url);
    if (!response.ok) throw new Error(`translate ${response.status}`);
  } catch {
    response = await proxyAwareFetch(url);
    if (!response.ok) throw new Error(`translate ${response.status}`);
  }

  const data = (await response.json()) as unknown;
  if (!Array.isArray(data) || !Array.isArray(data[0])) return q;
  let out = "";
  for (const part of data[0] as unknown[]) {
    if (Array.isArray(part) && typeof part[0] === "string") out += part[0];
  }
  return out.trim() || q;
}

export const CAPTION_SOURCE_LANGS = [
  { code: "en-US", labelHe: "אנגלית", labelEn: "English" },
  { code: "he-IL", labelHe: "עברית", labelEn: "Hebrew" },
  { code: "ru-RU", labelHe: "רוסית", labelEn: "Russian" },
  { code: "es-ES", labelHe: "ספרדית", labelEn: "Spanish" },
  { code: "fr-FR", labelHe: "צרפתית", labelEn: "French" },
  { code: "ar-SA", labelHe: "ערבית", labelEn: "Arabic" },
] as const;

export const CAPTION_TARGET_LANGS = [
  { code: CAPTION_TARGET_NONE, labelHe: "ללא תרגום", labelEn: "No translation" },
  { code: "he", labelHe: "עברית", labelEn: "Hebrew" },
  { code: "en", labelHe: "אנגלית", labelEn: "English" },
  { code: "ru", labelHe: "רוסית", labelEn: "Russian" },
  { code: "ar", labelHe: "ערבית", labelEn: "Arabic" },
] as const;

/** BCP-47 or target code → short language key (en, he, …). */
export function captionLangBase(code: string): string {
  if (!code || code === CAPTION_TARGET_NONE) return "";
  return code.split("-")[0]?.toLowerCase() ?? code.toLowerCase();
}

/** Skip translation when target is "none" or matches source language. */
export function shouldTranslateCaptions(sourceLang: string, targetLang: string): boolean {
  if (!targetLang || targetLang === CAPTION_TARGET_NONE) return false;
  const src = captionLangBase(sourceLang);
  const tgt = captionLangBase(targetLang);
  if (!src || !tgt) return false;
  return src !== tgt;
}

export function broadcastLangToSpeechCode(lang?: string): string {
  switch (lang) {
    case "heb":
      return "he-IL";
    case "rus":
      return "ru-RU";
    case "ara":
      return "ar-SA";
    case "fra":
      return "fr-FR";
    case "deu":
      return "de-DE";
    case "spa":
      return "es-ES";
    case "eng":
    default:
      return "en-US";
  }
}

/** Whisper language token for Xenova/whisper-tiny. */
export function speechLangToWhisperLanguage(code: string): string {
  const base = code.split("-")[0]?.toLowerCase() ?? "en";
  switch (base) {
    case "he":
      return "hebrew";
    case "ru":
      return "russian";
    case "ar":
      return "arabic";
    case "fr":
      return "french";
    case "de":
      return "german";
    case "es":
      return "spanish";
    case "en":
    default:
      return "english";
  }
}

export function broadcastLangToWhisperLanguage(lang?: string): string {
  return speechLangToWhisperLanguage(broadcastLangToSpeechCode(lang));
}
