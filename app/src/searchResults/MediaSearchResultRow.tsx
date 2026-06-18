import { useState } from "react";
import type { ChatUiLanguage } from "../ui/useUiLanguage";
import { MediaLightbox } from "./MediaLightbox";
import type { UnifiedSearchHit } from "./types";

type Props = {
  hit: UnifiedSearchHit;
  uiLang: ChatUiLanguage;
};

const formatDuration = (sec?: number): string => {
  if (!sec || sec <= 0) return "";
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return m > 0 ? `${m}:${String(s).padStart(2, "0")}` : `0:${String(s).padStart(2, "0")}`;
};

const labels = {
  he: { image: "תמונה", video: "וידאו", youtube: "YouTube", preview: "תצוגה מקדימה", open: "פתח" },
  en: { image: "Image", video: "Video", youtube: "YouTube", preview: "Preview", open: "Open" },
} as const;

/** Image / video hit in the unified «הכל» tab — thumbnail + lightbox (same assets as dedicated tabs). */
export function MediaSearchResultRow({ hit, uiLang }: Props) {
  const [lightbox, setLightbox] = useState(false);
  const L = labels[uiLang];
  const isVideo = hit.kind === "video" || hit.kind === "youtube";
  const isYoutube = hit.kind === "youtube";
  const thumb = hit.imageUrl;
  const pill = isYoutube ? L.youtube : isVideo ? L.video : L.image;
  const playable = isVideo && Boolean(hit.mediaPlayUrl?.trim());

  return (
    <>
      <article
        className={`serp-row serp-row--media-inline${isVideo ? " serp-row--media-video" : ""}`}
        dir={uiLang === "he" ? "rtl" : "ltr"}
      >
        <div className="serp-row-site">
          <div className="serp-row-site-main">
            <span className="serp-row-site-name">{hit.sourceLabel}</span>
            {hit.author ? <span className="serp-row-meta-inline">{hit.author}</span> : null}
            {isVideo && hit.durationSec ? (
              <span className="serp-row-meta-inline">{formatDuration(hit.durationSec)}</span>
            ) : null}
          </div>
          <span className={isVideo ? "serp-video-pill" : "serp-image-pill"}>{pill}</span>
        </div>

        <div className="serp-media-inline-body">
          <button
            type="button"
            className="serp-media-inline-thumb-btn"
            onClick={() => setLightbox(true)}
            aria-label={`${L.preview}: ${hit.title}`}
          >
            <span className="serp-media-inline-thumb-wrap">
              {thumb ? (
                <img
                  className="serp-media-inline-thumb"
                  src={thumb}
                  alt=""
                  loading="lazy"
                  referrerPolicy="no-referrer"
                />
              ) : (
                <span className="serp-media-inline-thumb serp-media-inline-thumb--placeholder" />
              )}
              {playable ? <span className="serp-media-inline-play" aria-hidden="true">▶</span> : null}
            </span>
          </button>

          <div className="serp-media-inline-text">
            <button
              type="button"
              className="serp-row-title serp-media-inline-title-btn"
              onClick={() => setLightbox(true)}
            >
              {hit.title}
            </button>
            {hit.snippet ? <p className="serp-row-snippet serp-media-inline-snippet">{hit.snippet}</p> : null}
            <div className="serp-media-inline-actions">
              <button type="button" className="serp-btn serp-btn--ghost" onClick={() => setLightbox(true)}>
                {L.open}
              </button>
              <a className="serp-btn serp-btn--ghost" href={hit.url} target="_blank" rel="noopener noreferrer">
                {isYoutube ? "YouTube" : "Pixabay"}
              </a>
            </div>
          </div>
        </div>
      </article>

      {lightbox ? <MediaLightbox hit={hit} uiLang={uiLang} onClose={() => setLightbox(false)} /> : null}
    </>
  );
}
