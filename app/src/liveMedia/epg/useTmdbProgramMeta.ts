import { useEffect, useState } from "react";
import { TMDB_KEY_SAVED_EVENT } from "../../apiKeys/apiKeyStore";
import type { ChatUiLanguage } from "../../ui/useUiLanguage";
import { lookupTmdbProgram, tmdbLocaleForUi, type TmdbProgramMeta } from "../../tmdb/tmdbClient";

export function useTmdbProgramMeta(
  title: string | undefined,
  enabled: boolean,
  uiLang: ChatUiLanguage,
  season?: number,
  episode?: number,
): TmdbProgramMeta | null {
  const [meta, setMeta] = useState<TmdbProgramMeta | null>(null);
  const language = tmdbLocaleForUi(uiLang);

  useEffect(() => {
    if (!enabled || !title?.trim()) {
      setMeta(null);
      return;
    }
    let alive = true;
    const load = () => {
      void lookupTmdbProgram(title, { season, episode, language }).then((m) => {
        if (alive) setMeta(m);
      });
    };
    load();
    window.addEventListener(TMDB_KEY_SAVED_EVENT, load);
    return () => {
      alive = false;
      window.removeEventListener(TMDB_KEY_SAVED_EVENT, load);
    };
  }, [title, enabled, season, episode, language]);

  return meta;
}
