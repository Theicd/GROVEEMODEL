import { useEffect, useState } from "react";
import { TMDB_KEY_SAVED_EVENT } from "../../apiKeys/apiKeyStore";
import type { ChatUiLanguage } from "../../ui/useUiLanguage";
import type { EpgProgram } from "./types";
import { lookupTmdbProgram, tmdbLocaleForUi, type TmdbProgramMeta } from "../../tmdb/tmdbClient";

export function useTmdbProgramMeta(
  program: EpgProgram | null | undefined,
  enabled: boolean,
  uiLang: ChatUiLanguage,
): TmdbProgramMeta | null {
  const [meta, setMeta] = useState<TmdbProgramMeta | null>(null);
  const language = tmdbLocaleForUi(uiLang);
  const title = program?.title;

  useEffect(() => {
    if (!enabled || !title?.trim() || !program) {
      setMeta(null);
      return;
    }
    setMeta(null);
    let alive = true;
    const load = () => {
      void lookupTmdbProgram(title, {
        season: program.season,
        episode: program.episode,
        language,
        program,
      }).then((m) => {
        if (alive) setMeta(m);
      });
    };
    load();
    window.addEventListener(TMDB_KEY_SAVED_EVENT, load);
    return () => {
      alive = false;
      window.removeEventListener(TMDB_KEY_SAVED_EVENT, load);
    };
  }, [title, enabled, program, program?.season, program?.episode, language]);

  return meta;
}
