// @ts-nocheck
import { getFeedLang } from "../feeds/feedRegistry";
import { getUserNewsProfile } from "../settings/userNewsProfile";
import { isLikelyInLanguage, needsDisplayTranslation } from "../summarize/languageDetect";
import { translateTexts } from "../translate/googleTranslate";
import type { ArticleRecord } from "../types";

const SUMMARY_MAX = 480;

/** Fill displayTitle/displaySummary in the user's UI language (Google Translate + cache). */
export async function applyDisplayLanguageBatch(
  articles: ArticleRecord[],
  targetLang?: string,
): Promise<ArticleRecord[]> {
  if (!articles.length) return articles;

  const target = (targetLang ?? getUserNewsProfile().uiLanguage).trim().toLowerCase() || "en";
  const out = articles.map((a) => ({ ...a }));
  const indices: number[] = [];
  const titles: string[] = [];
  const summaries: string[] = [];

  for (let i = 0; i < out.length; i++) {
    const a = out[i];
    const title = a.title.trim();
    const summarySrc = (a.summary || a.articleText || "").trim();
    const feedLang = getFeedLang(a.sourceKey);

    const alreadyInTarget =
      feedLang === target &&
      isLikelyInLanguage(title, target) &&
      (!summarySrc || isLikelyInLanguage(summarySrc.slice(0, 400), target));

    if (alreadyInTarget || !needsDisplayTranslation(title, summarySrc, target)) {
      out[i] = {
        ...a,
        displayTitle: a.displayTitle ?? title,
        displaySummary: a.displaySummary ?? (summarySrc || a.summary),
      };
      continue;
    }

    indices.push(i);
    titles.push(title);
    summaries.push(summarySrc.slice(0, SUMMARY_MAX) || title);
  }

  if (!indices.length) return out;

  try {
    const [titleBatch, summaryBatch] = await Promise.all([
      translateTexts(titles, target, "auto"),
      translateTexts(summaries, target, "auto"),
    ]);

    indices.forEach((idx, j) => {
      out[idx] = {
        ...out[idx],
        displayTitle: titleBatch.texts[j] || out[idx].title,
        displaySummary: summaryBatch.texts[j] || out[idx].summary || out[idx].articleText,
      };
    });
  } catch {
    indices.forEach((idx) => {
      out[idx] = {
        ...out[idx],
        displayTitle: out[idx].displayTitle ?? out[idx].title,
        displaySummary: out[idx].displaySummary ?? out[idx].summary,
      };
    });
  }

  return out;
}

/** @deprecated use applyDisplayLanguageBatch */
export const applyEnglishDisplayBatch = applyDisplayLanguageBatch;
