// @ts-nocheck
import { isDeepReadEnabled } from "../engine/deepReadGate";
import { getUserNewsProfile } from "../settings/userNewsProfile";
import { isLikelyInLanguage, needsDisplayTranslation } from "../summarize/languageDetect";
import type { ArticleRecord } from "../types";
import { translateHeadlineToEnglish } from "../summarize/summarizerClient";
import { translateTexts } from "../translate/googleTranslate";

async function translateToLanguage(
  title: string,
  summary: string,
  target: string,
): Promise<{ title: string; summary: string }> {
  const summarySrc = summary.trim().slice(0, 480) || title;
  const { texts } = await translateTexts([title, summarySrc], target, "auto");
  return {
    title: texts[0]?.trim() || title,
    summary: texts[1]?.trim() || summary,
  };
}

/** Fill UI display fields in the user's chosen language. */
export async function ensureEnglishDisplay(article: ArticleRecord): Promise<ArticleRecord> {
  const target = getUserNewsProfile().uiLanguage;
  const title = article.displayTitle ?? article.title;
  const summary = article.displaySummary ?? article.summary;

  if (!needsDisplayTranslation(title, summary || article.articleText, target)) {
    return {
      ...article,
      displayTitle: title,
      displaySummary: summary,
    };
  }

  if (target === "en" && isDeepReadEnabled()) {
    const translated = await translateHeadlineToEnglish(article.title, summary || article.articleText);
    if (isLikelyInLanguage(translated.title, "en")) {
      return {
        ...article,
        displayTitle: translated.title,
        displaySummary: translated.summary,
      };
    }
  }

  const translated = await translateToLanguage(article.title, summary || article.articleText, target);
  return {
    ...article,
    displayTitle: translated.title,
    displaySummary: translated.summary,
  };
}

export async function backfillEnglishDisplay(articles: ArticleRecord[], limit = 20): Promise<number> {
  const target = getUserNewsProfile().uiLanguage;
  let updated = 0;
  for (const a of articles) {
    if (updated >= limit) break;
    if (a.displayTitle && a.displaySummary && isLikelyInLanguage(a.displayTitle, target)) continue;
    if (!needsDisplayTranslation(a.title, a.summary, target)) continue;
    await ensureEnglishDisplay(a);
    updated++;
  }
  return updated;
}
