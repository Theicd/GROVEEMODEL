import { useEffect, useState } from "react";

import type { ChatUiLanguage } from "../../ui/useUiLanguage";

import { localizeEpgCopyForUi } from "./epgLocalize";

import type { NowPlayingInfo } from "./epgNowPlaying";

export type LocalizedNowPlaying = NowPlayingInfo & {
  displayTitle: string;
  seriesTitle: string | null;
  description: string | null;
};

export function useLocalizedEpgNowPlaying(
  info: NowPlayingInfo | null,
  uiLang: ChatUiLanguage,
  tmdbTitle?: string | null,
  tmdbOverview?: string | null,
  tmdbSeriesTitle?: string | null,
  channelKey?: string | null,
): LocalizedNowPlaying | null {
  const [localized, setLocalized] = useState<LocalizedNowPlaying | null>(null);
  const [appliedKey, setAppliedKey] = useState<string | null>(null);

  useEffect(() => {
    if (!info || !channelKey) {
      setLocalized(null);
      setAppliedKey(null);
      return;
    }

    const displayTitle = tmdbTitle?.trim() || info.program.title;
    const seriesTitle = tmdbSeriesTitle?.trim() || null;
    const englishDesc = info.program.description?.trim() || info.program.subTitle?.trim() || null;

    const apply = (description: string | null) => {
      setLocalized({
        ...info,
        displayTitle,
        seriesTitle,
        description,
      });
      setAppliedKey(channelKey);
    };

    if (uiLang !== "he") {
      apply(tmdbOverview?.trim() || englishDesc);
      return;
    }

    apply(tmdbOverview?.trim() || englishDesc);

    if (tmdbOverview?.trim()) return;

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
      apply(epgDesc);
    })();

    return () => {
      alive = false;
    };
  }, [
    info,
    channelKey,
    uiLang,
    tmdbTitle,
    tmdbOverview,
    tmdbSeriesTitle,
    info?.program.title,
    info?.program.description,
    info?.program.subTitle,
    info?.displayStart,
    info?.displayEnd,
  ]);

  if (!info || !channelKey || appliedKey !== channelKey) return null;
  return localized;
}
