import { needsDisplayTranslation } from "../groveeNews/engine/summarize/languageDetect";
import { translateTexts } from "../groveeNews/engine/translate/googleTranslate";
import type { ChatUiLanguage } from "../ui/useUiLanguage";
import { cleanDisplaySnippet } from "./snippetCleanup";
import type { UnifiedSearchHit } from "./types";

const needsHitTranslation = (hit: UnifiedSearchHit, targetLang: string): boolean => {
  const title = hit.titleOriginal ?? hit.title;
  const snippet = hit.snippetOriginal ?? hit.snippet;
  return needsDisplayTranslation(title, snippet, targetLang);
};

/** Translate SERP titles/snippets to the UI language (Google Translate). */
export async function translateSearchHits(
  hits: UnifiedSearchHit[],
  uiLang: ChatUiLanguage,
): Promise<UnifiedSearchHit[]> {
  if (!hits.length) return hits;

  const target = uiLang;
  const indices: number[] = [];
  const titles: string[] = [];
  const snippets: string[] = [];
  const hadSnippet: boolean[] = [];

  for (let i = 0; i < hits.length; i++) {
    const hit = hits[i];
    const title = hit.titleOriginal ?? hit.title;
    const snippet = hit.snippetOriginal ?? hit.snippet;
    if (!needsDisplayTranslation(title, snippet, target)) continue;
    indices.push(i);
    titles.push(title);
    hadSnippet.push(Boolean(snippet.trim()));
    snippets.push(snippet.trim());
  }

  if (!indices.length) {
    return hits.map((h) => normalizeHit(h, h.title, h.snippet));
  }

  try {
    const snippetTexts = snippets.filter((s) => s.length > 0);

    const titleBatch = await translateTexts(titles, target, "auto");
    const snippetBatch =
      snippetTexts.length > 0
        ? await translateTexts(snippetTexts, target, "auto")
        : { texts: [] as string[], provider: "cache" as const };

    let snippetCursor = 0;

    return hits.map((hit, i) => {
      const j = indices.indexOf(i);
      if (j < 0) {
        return normalizeHit(hit, hit.title, hit.snippet);
      }

      const translatedTitle = titleBatch.texts[j] || hit.title;
      let translatedSnippet = "";
      if (hadSnippet[j]) {
        translatedSnippet = snippetBatch.texts[snippetCursor] || hit.snippet;
        snippetCursor += 1;
      }

      return normalizeHit(hit, translatedTitle, translatedSnippet);
    });
  } catch {
    return hits.map((h) => normalizeHit(h, h.title, h.snippet));
  }
}

const normalizeHit = (
  hit: UnifiedSearchHit,
  title: string,
  snippet: string,
): UnifiedSearchHit => {
  const titleOriginal = hit.titleOriginal ?? hit.title;
  const snippetOriginal = hit.snippetOriginal ?? hit.snippet;
  const cleanSnippet = cleanDisplaySnippet(title, snippet, hit.url);

  return {
    ...hit,
    titleOriginal,
    snippetOriginal,
    title,
    snippet: cleanSnippet,
  };
};

export const hitNeedsTranslatePageLink = (hit: UnifiedSearchHit, uiLang: ChatUiLanguage): boolean =>
  needsHitTranslation(
    {
      ...hit,
      title: hit.titleOriginal ?? hit.title,
      snippet: hit.snippetOriginal ?? hit.snippet,
    },
    uiLang,
  );
