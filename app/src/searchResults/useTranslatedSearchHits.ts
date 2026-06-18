import { useEffect, useState } from "react";
import type { ChatUiLanguage } from "../ui/useUiLanguage";
import { translateSearchHits } from "./translateHits";
import type { UnifiedSearchHit } from "./types";

export function useTranslatedSearchHits(
  hits: UnifiedSearchHit[],
  uiLang: ChatUiLanguage,
): { hits: UnifiedSearchHit[]; translating: boolean } {
  const [translated, setTranslated] = useState(hits);
  const [translating, setTranslating] = useState(false);

  useEffect(() => {
    let cancelled = false;
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
