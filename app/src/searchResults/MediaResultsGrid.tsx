import { useState } from "react";
import type { ChatUiLanguage } from "../ui/useUiLanguage";
import type { UnifiedSearchHit } from "./types";
import { MediaLightbox } from "./MediaLightbox";

type Props = {
  hits: UnifiedSearchHit[];
  uiLang: ChatUiLanguage;
  mode: "image" | "video" | "youtube";
};

const formatDuration = (sec?: number): string => {
  if (!sec || sec <= 0) return "";
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return m > 0 ? `${m}:${String(s).padStart(2, "0")}` : `0:${String(s).padStart(2, "0")}`;
};

export function MediaResultsGrid({ hits, uiLang, mode }: Props) {
  const [active, setActive] = useState<UnifiedSearchHit | null>(null);

  return (
    <>
      <div className="serp-media-grid" role="list">
        {hits.map((hit) => (
          <button
            key={hit.id}
            type="button"
            className={`serp-media-card serp-media-card--${hit.kind === "youtube" ? "youtube" : mode}`}
            onClick={() => setActive(hit)}
            role="listitem"
          >
            <span className="serp-media-thumb-wrap">
              {hit.imageUrl ? (
                <img
                  className="serp-media-thumb"
                  src={hit.imageUrl}
                  alt={hit.title}
                  loading="lazy"
                  referrerPolicy="no-referrer"
                />
              ) : (
                <span className="serp-media-thumb serp-media-thumb--placeholder" />
              )}
              {(hit.kind === "video" || hit.kind === "youtube") && hit.durationSec ? (
                <span className="serp-media-duration">{formatDuration(hit.durationSec)}</span>
              ) : null}
              {hit.kind === "video" || (hit.kind === "youtube" && hit.mediaPlayUrl) ? (
                <span className="serp-media-play" aria-hidden="true">
                  ▶
                </span>
              ) : null}
            </span>
            <span className="serp-media-meta">
              <span className="serp-media-title">{hit.title}</span>
              {hit.author ? <span className="serp-media-author">{hit.author}</span> : null}
            </span>
          </button>
        ))}
      </div>

      {active ? (
        <MediaLightbox hit={active} uiLang={uiLang} onClose={() => setActive(null)} />
      ) : null}
    </>
  );
}
