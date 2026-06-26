import { useEffect, useState } from "react";
import type { ChatUiLanguage } from "../../ui/useUiLanguage";
import { localizeEpgCopyForUi } from "./epgLocalize";
import type { NowPlayingInfo } from "./epgNowPlaying";

export type LocalizedNowPlaying = NowPlayingInfo & {
  displayTitle: string;
  description: string | null;
};

export function useLocalizedEpgNowPlaying(
  info: NowPlayingInfo | null,
  uiLang: ChatUiLanguage,
  tmdbTitle?: string | null,
  tmdbOverview?: string | null,
): LocalizedNowPlaying | null {
  const [localized, setLocalized] = useState<LocalizedNowPlaying | null>(null);

  useEffect(() => {
    if (!info) {
      setLocalized(null);
      return;
    }

    const preferTmdbCopy = uiLang === "he";
    const base: LocalizedNowPlaying = {
      ...info,
      displayTitle: (preferTmdbCopy ? tmdbTitle?.trim() : null) || info.program.title,
      description: preferTmdbCopy ? (tmdbOverview?.trim() || null) : null,
    };

    if (!preferTmdbCopy) {
      setLocalized({
        ...base,
        displayTitle: info.program.title,
        description: info.program.description?.trim() || info.program.subTitle?.trim() || tmdbOverview?.trim() || null,
      });
      return;
    }

    let alive = true;
    void (async () => {
      const epgCopy = await localizeEpgCopyForUi(
        {
          title: info.program.title,
          description: info.program.description,
          subTitle: info.program.subTitle,
        },
        uiLang,
      );
      if (!alive) return;
      const epgDesc = epgCopy.description?.trim() || epgCopy.subTitle?.trim() || null;
      setLocalized({
        ...info,
        displayTitle: tmdbTitle?.trim() || epgCopy.title,
        description: tmdbOverview?.trim() || epgDesc,
      });
    })();

    return () => {
      alive = false;
    };
  }, [
    info,
    uiLang,
    tmdbTitle,
    tmdbOverview,
    info?.program.title,
    info?.program.description,
    info?.program.subTitle,
    info?.displayStart,
    info?.displayEnd,
  ]);

  return localized;
}
