import { searchNews } from "./engine/engine/pipeline";
import { getEngineLibraryStats } from "./engine/engine/engineStats";
import { getSearchIndexSize } from "./engine/search/flexIndex";
import { hitsToDisplayCards } from "./searchAdapter";
import { fetchTopicsBundle } from "./topicsAdapter";
import { extractArticleFromUrl } from "./engine/extract/readabilityExtract";
import { buildBriefNotes, formatArticleSummaryForUser } from "./articleSummaryDisplay";
import { isLikelyEnglish, needsEnglishDisplay } from "./engine/summarize/languageDetect";
import { normalizeSummarizerResult } from "./engine/summarize/summaryQuality";
import { summarizeArticle } from "./summarizerApp";
import { getUserNewsProfile } from "./engine/settings/userNewsProfile";
import type { ArticleReadResult, GroveeNewsCard, GroveeTopicsBundle } from "./types";
import { startGroveeNewsBoot, isGroveeNewsReady } from "./engineBoot";

export { buildNewsPanelGuideReply } from "./newsPanelGuideReply";
export type { NewsPanelGuideOptions } from "./newsPanelGuideReply";
export { fetchGroveeNewsSearch } from "./newsToSearchBrief";
export { startGroveeNewsBoot, isGroveeNewsReady };
export type { GroveeNewsCard, GroveeTopicsBundle, NewsPanelPayload, ArticleReadResult, NewsSummaryGemmaProgress } from "./types";
export {
  getNewsPanelPayload,
  setNewsPanelPayload,
  subscribeNewsPanelPayload,
  clearNewsPanelPayload,
} from "./newsPanelStore";
export { isTopicsOverviewQuery } from "./headlineIntent";

export async function getGroveeNewsLibraryStats() {
  return getEngineLibraryStats(getSearchIndexSize());
}

export async function groveeNewsSearch(query: string): Promise<GroveeNewsCard[]> {
  await startGroveeNewsBoot();
  const hits = await searchNews(query);
  return hitsToDisplayCards(hits);
}

export async function groveeNewsTopics(): Promise<GroveeTopicsBundle> {
  await startGroveeNewsBoot();
  return fetchTopicsBundle();
}

export async function readAndSummarizeArticle(
  url: string,
  options: { onQwenToken?: (tokens: number) => void } = {},
): Promise<ArticleReadResult> {
  await startGroveeNewsBoot();
  const profile = getUserNewsProfile();
  const targetLang = profile.uiLanguage || "he";

  try {
    const extracted = await extractArticleFromUrl(url);
    const body = extracted?.text?.trim() ?? "";
    const title = extracted?.title?.trim() || "ידיעה";
    if (!body) {
      return { title, summaryHe: "לא ניתן לשלוף את תוכן הכתבה.", url, usedQwen: false, error: "empty" };
    }

    const hebrewArticle =
      !isLikelyEnglish(body.slice(0, 600)) && !needsEnglishDisplay(title, body.slice(0, 600));

    if (hebrewArticle) {
      const summaryHe = body.slice(0, 900).trim();
      return { title, summaryHe, url, usedQwen: false };
    }

    const summarized = await summarizeArticle(body, { onDemand: true, onQwenToken: options.onQwenToken });
    const normalized = normalizeSummarizerResult(summarized, "", body, title);
    const notes = buildBriefNotes(normalized, title);
    const summaryHe = formatArticleSummaryForUser(
      normalized,
      title,
      targetLang === "en" ? "en" : "he",
    );

    return {
      title,
      summaryHe,
      qwenDraft: targetLang === "en" ? undefined : notes,
      url,
      usedQwen: true,
    };
  } catch (err) {
    return {
      title: "ידיעה",
      summaryHe: "",
      url,
      usedQwen: false,
      error: err instanceof Error ? err.message : "שגיאה",
    };
  }
}
