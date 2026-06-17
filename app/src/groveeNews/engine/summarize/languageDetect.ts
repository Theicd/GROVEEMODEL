// @ts-nocheck
import { detectChineseText } from "./prompts";

const EN_WORD =
  /\b(the|and|of|to|in|for|on|with|is|are|was|were|has|have|had|said|says|new|after|from|that|this|will|not|police|government)\b/i;

function hasNonLatinScript(text: string): boolean {
  return /[\u0400-\u04FF\u0590-\u05FF\u0600-\u06FF\u0900-\u097F\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]/.test(
    text,
  );
}

/** Heuristic: is this text already usable as English UI copy? */
export function isLikelyEnglish(text: string): boolean {
  const t = text.trim();
  if (!t || t.length < 4) return true;
  if (detectChineseText(t)) return false;
  if (hasNonLatinScript(t)) return false;

  if (/[äöüßÄÖÜ]/.test(t)) return false;
  if (/\b(und|der|die|das|nicht|sind|wurde|gegen|auch|beamte|polizei|ermittlungen)\b/i.test(t)) return false;

  if (/[àâçéèêëîïôùûœæ]/i.test(t)) return false;
  if (/\b(les|des|une|dans|pour|avec|cette|sont|été|contre|police)\b/i.test(t) && !EN_WORD.test(t)) return false;

  if (/\b(el|los|las|del|por|para|como|más|gobierno)\b/i.test(t) && !EN_WORD.test(t)) return false;

  if (/\b(bir|ve|için|değil|olarak|türkiye|cumhurbaşkanı|polis)\b/i.test(t) && !EN_WORD.test(t)) return false;

  if (/\b(và|của|không|trong|việt|chính|phủ)\b/i.test(t) && !EN_WORD.test(t)) return false;

  const enHits = (t.match(EN_WORD) ?? []).length;
  if (enHits >= 2) return true;
  if (enHits >= 1 && t.split(/\s+/).length <= 12) return true;

  const letters = t.replace(/\s/g, "");
  const latin = (letters.match(/[a-zA-Z]/g) ?? []).length;
  const ratio = latin / Math.max(1, letters.length);
  return ratio > 0.9 && enHits >= 1;
}

export function needsEnglishDisplay(title: string, summary: string): boolean {
  return !isLikelyEnglish(title) || !isLikelyEnglish(summary.slice(0, 400));
}

/** Rough check that text is already in the user's chosen UI language. */
export function isLikelyInLanguage(text: string, lang: string): boolean {
  const t = text.trim();
  if (!t || t.length < 3) return true;
  const code = lang.toLowerCase();

  if (code === "en") return isLikelyEnglish(t);

  if (code === "he") return /[\u0590-\u05FF]/.test(t);
  if (code === "ar") return /[\u0600-\u06FF]/.test(t);
  if (code === "fa") return /[\u0600-\u06FF]/.test(t);
  if (code === "ru") return /[\u0400-\u04FF]/.test(t);
  if (code === "ja") return /[\u3040-\u30FF\u4E00-\u9FFF]/.test(t);
  if (code === "zh") return detectChineseText(t);
  if (code === "ko") return /[\uAC00-\uD7AF]/.test(t);

  if (code === "fr") {
    return (
      (/[àâçéèêëîïôùûœæ]/i.test(t) ||
        /\b(les|des|une|dans|pour|avec|cette|sont|été|bonjour|france|français)\b/i.test(t)) &&
      !EN_WORD.test(t)
    );
  }
  if (code === "de") {
    return (/[äöüßÄÖÜ]/.test(t) || /\b(und|der|die|das|nicht|sind|wurde)\b/i.test(t)) && !EN_WORD.test(t);
  }
  if (code === "es") {
    return /\b(el|los|las|del|por|para|como|más|gobierno|según)\b/i.test(t) && !EN_WORD.test(t);
  }
  if (code === "pt") {
    return /\b(não|para|como|mais|governo|sobre|pelo|pela)\b/i.test(t) && !EN_WORD.test(t);
  }

  return false;
}

export function needsDisplayTranslation(title: string, summary: string, targetLang: string): boolean {
  const summarySlice = summary.slice(0, 400);
  if (targetLang === "en") return needsEnglishDisplay(title, summarySlice);
  return !isLikelyInLanguage(title, targetLang) || !isLikelyInLanguage(summarySlice, targetLang);
}
