import { useState } from "react";
import type { ChatUiLanguage } from "../ui/useUiLanguage";
import type { UnifiedSearchHit } from "./types";
import { MediaLightbox } from "./MediaLightbox";

type Props = {
  hits: UnifiedSearchHit[];
  uiLang: ChatUiLanguage;
  mode: "livetv" | "radio";
  favoriteIds?: Set<string>;
  onToggleFavorite?: (hit: UnifiedSearchHit) => void;
  onHideChannel?: (hit: UnifiedSearchHit) => void;
};

export function LiveMediaResultsGrid({
  hits,
  uiLang,
  mode,
  favoriteIds,
  onToggleFavorite,
  onHideChannel,
}: Props) {
  const [active, setActive] = useState<UnifiedSearchHit | null>(null);
  const liveLabel = uiLang === "he" ? "שידור חי" : "LIVE";

  return (
    <>
      <div className="serp-media-grid serp-live-grid" role="list">
        {hits.map((hit) => (
          <button
            key={hit.id}
            type="button"
            className={`serp-media-card serp-media-card--${mode} serp-live-card`}
            onClick={() => setActive(hit)}
            role="listitem"
          >
            <span className="serp-media-thumb-wrap serp-live-thumb-wrap">
              {hit.imageUrl ? (
                <img
                  className="serp-media-thumb"
                  src={hit.imageUrl}
                  alt={hit.title}
                  loading="lazy"
                  referrerPolicy="no-referrer"
                />
              ) : (
                <span className={`serp-media-thumb serp-media-thumb--placeholder serp-live-placeholder serp-live-placeholder--${mode}`} />
              )}
              <span className="serp-live-badge">{liveLabel}</span>
              {hit.meta?.status ? (
                <span className={`serp-live-status serp-live-status--${hit.meta.status}`}>
                  {hit.meta.status === "working"
                    ? uiLang === "he"
                      ? "פעיל"
                      : "OK"
                    : hit.meta.status === "warning"
                      ? uiLang === "he"
                        ? "איטי"
                        : "Slow"
                      : hit.meta.status === "offline"
                        ? uiLang === "he"
                          ? "לא פעיל"
                          : "Off"
                        : uiLang === "he"
                          ? "?"
                          : "?"}
                </span>
              ) : null}
              <span className="serp-media-play" aria-hidden="true">
                {mode === "radio" ? "♫" : "▶"}
              </span>
              {onToggleFavorite ? (
                <button
                  type="button"
                  className={`serp-live-fav${favoriteIds?.has(hit.id) ? " is-active" : ""}`}
                  aria-label={
                    favoriteIds?.has(hit.id)
                      ? uiLang === "he"
                        ? "הסר ממועדפים"
                        : "Remove favorite"
                      : uiLang === "he"
                        ? "הוסף למועדפים"
                        : "Add favorite"
                  }
                  title={
                    favoriteIds?.has(hit.id)
                      ? uiLang === "he"
                        ? "הסר ממועדפים"
                        : "Remove favorite"
                      : uiLang === "he"
                        ? "הוסף למועדפים"
                        : "Add favorite"
                  }
                  onClick={(e) => {
                    e.stopPropagation();
                    onToggleFavorite(hit);
                  }}
                >
                  {favoriteIds?.has(hit.id) ? "★" : "☆"}
                </button>
              ) : null}
              {onHideChannel ? (
                <button
                  type="button"
                  className="serp-live-hide"
                  aria-label={uiLang === "he" ? "הסר מהרשימה" : "Hide channel"}
                  title={uiLang === "he" ? "הסר לרשימה השחורה" : "Add to blacklist"}
                  onClick={(e) => {
                    e.stopPropagation();
                    onHideChannel(hit);
                  }}
                >
                  ✕
                </button>
              ) : null}
            </span>
            <span className="serp-media-meta">
              <span className="serp-media-title">{hit.title}</span>
              <span className="serp-live-sub">{hit.snippet}</span>
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
