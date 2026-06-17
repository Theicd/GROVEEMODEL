import { searchNews } from "./engine/engine/pipeline";
import { getEngineLibraryStats } from "./engine/engine/engineStats";
import { getSearchIndexSize } from "./engine/search/flexIndex";
import { hitsToDisplayCards } from "./searchAdapter";
import { fetchTopicsBundle } from "./topicsAdapter";
import { extractArticleFromUrl } from "./engine/extract/readabilityExtract";
import { buildArticleExcerptForGemma } from "./gemmaNewsPolish";
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

/** Fetch article body and prepare a long excerpt for Gemma (no Qwen). */
export async function readAndSummarizeArticle(url: string): Promise<ArticleReadResult> {
  await startGroveeNewsBoot();

  try {
    const extracted = await extractArticleFromUrl(url);
    const body = extracted?.text?.trim() ?? "";
    const title = extracted?.title?.trim() || "ידיעה";
    if (!body) {
      return { title, summaryHe: "לא ניתן לשלוף את תוכן הכתבה.", url, usedQwen: false, error: "empty" };
    }

    const gemmaInput = buildArticleExcerptForGemma(body);
    if (!gemmaInput.trim()) {
      return { title, summaryHe: "לא ניתן להכין טקסט לסיכום.", url, usedQwen: false, error: "empty" };
    }

    return {
      title,
      summaryHe: "",
      gemmaInput,
      url,
      usedQwen: false,
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
