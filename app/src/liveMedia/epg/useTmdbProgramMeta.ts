import { useEffect, useState } from "react";
import { TMDB_KEY_SAVED_EVENT } from "../../apiKeys/apiKeyStore";
import type { ChatUiLanguage } from "../../ui/useUiLanguage";
import type { EpgProgram } from "./types";
import { lookupTmdbProgram, tmdbLocaleForUi, type TmdbProgramMeta } from "../../tmdb/tmdbClient";

export function useTmdbProgramMeta(
  program: EpgProgram | null | undefined,
  enabled: boolean,
  uiLang: ChatUiLanguage,
  channelTitle?: string,
  channelKey?: string | null,
): TmdbProgramMeta | null {
  const [meta, setMeta] = useState<TmdbProgramMeta | null>(null);
  const [appliedKey, setAppliedKey] = useState<string | null>(null);
  const language = tmdbLocaleForUi(uiLang);
  const title = program?.title;
  const programKey =
    program && channelKey ?
      `${channelKey}|${program.start.toISOString()}|${program.title}|${program.season ?? ""}|${program.episode ?? ""}`
    : null;

  useEffect(() => {
    if (!enabled || !title?.trim() || !program || !programKey) {
      setMeta(null);
      setAppliedKey(null);
      return;
    }
    setMeta(null);
    setAppliedKey(null);
    let alive = true;
    const load = () => {
      void lookupTmdbProgram(title, {
        season: program.season,
        episode: program.episode,
        language,
        program,
        channelTitle,
      }).then((m) => {
        if (!alive) return;
        setMeta(m);
        setAppliedKey(programKey);
      });
    };
    load();
    window.addEventListener(TMDB_KEY_SAVED_EVENT, load);
    return () => {
      alive = false;
      window.removeEventListener(TMDB_KEY_SAVED_EVENT, load);
    };
  }, [title, enabled, program, program?.season, program?.episode, language, channelTitle, programKey]);

  if (!programKey || appliedKey !== programKey) return null;
  return meta;
}
