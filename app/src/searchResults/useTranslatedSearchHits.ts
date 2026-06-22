import { useEffect, useState } from "react";
import { needsDisplayTranslation } from "../groveeNews/engine/summarize/languageDetect";
import type { ChatUiLanguage } from "../ui/useUiLanguage";
import { translateSearchHits } from "./translateHits";
import type { UnifiedSearchHit } from "./types";

export const TRANSLATE_SKIP_KINDS = new Set<UnifiedSearchHit["kind"]>([
  "product",
  "image",
  "video",
  "youtube",
  "livetv",
  "radio",
  "hfmodel",
  "movie",
]);

export const hitsNeedTranslation = (hits: UnifiedSearchHit[], uiLang: ChatUiLanguage): boolean =>
  hits.some((hit) => {
    if (TRANSLATE_SKIP_KINDS.has(hit.kind)) return false;
    const title = hit.titleOriginal ?? hit.title;
    const snippet = hit.snippetOriginal ?? hit.snippet;
    return needsDisplayTranslation(title, snippet, uiLang);
  });

export function useTranslatedSearchHits(
  hits: UnifiedSearchHit[],
  uiLang: ChatUiLanguage,
): { hits: UnifiedSearchHit[]; translating: boolean } {
  const [translated, setTranslated] = useState(hits);
  const [translating, setTranslating] = useState(false);

  useEffect(() => {
    let cancelled = false;
    if (!hitsNeedTranslation(hits, uiLang)) {
      setTranslated(hits);
      setTranslating(false);
      return;
    }
    setTranslating(true);
    void translateSearchHits(hits, uiLang).then((next) => {
      if (!cancelled) {
        setTranslated(next);
        setTranslating(false);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [hits, uiLang]);

  return { hits: translated, translating };
}
