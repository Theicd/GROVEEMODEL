import { useEffect, type MouseEvent } from "react";
import type { ChatUiLanguage } from "../ui/useUiLanguage";
import type { UnifiedSearchHit } from "./types";

type Props = {
  hit: UnifiedSearchHit;
  uiLang: ChatUiLanguage;
  onClose: () => void;
};

const labels = {
  he: {
    close: "סגור",
    download: "הורדה",
    sourceOn: (name: string) => `מקור ב-${name}`,
    by: "על ידי",
  },
  en: {
    close: "Close",
    download: "Download",
    sourceOn: (name: string) => `View on ${name}`,
    by: "by",
  },
} as const;

const usesEmbedPlayer = (hit: UnifiedSearchHit): boolean =>
  hit.mediaEmbedMode === true ||
  hit.kind === "youtube" ||
  hit.provider === "invidious-videos" ||
  /\/embed\//i.test(hit.mediaPlayUrl ?? "");

export function MediaLightbox({ hit, uiLang, onClose }: Props) {
  const L = labels[uiLang];
  const isVideo = hit.kind === "video" || hit.kind === "youtube";
  const embed = isVideo && usesEmbedPlayer(hit);
  const mediaSrc = hit.mediaPlayUrl || hit.imageUrl || hit.url;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const stop = (e: MouseEvent) => e.stopPropagation();

  return (
    <div className="serp-media-lightbox" role="dialog" aria-modal="true" onClick={onClose}>
      <div className="serp-media-lightbox-inner" onClick={stop}>
        <button type="button" className="serp-media-lightbox-close" onClick={onClose} aria-label={L.close}>
          ×
        </button>

        {isVideo ? (
          embed ? (
            <iframe
              className="serp-media-lightbox-embed"
              src={mediaSrc}
              title={hit.title}
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
            />
          ) : (
            <video
              className="serp-media-lightbox-video"
              src={mediaSrc}
              controls
              autoPlay
              playsInline
              poster={hit.imageUrl}
            />
          )
        ) : (
          <img className="serp-media-lightbox-image" src={mediaSrc} alt={hit.title} />
        )}

        <div className="serp-media-lightbox-foot">
          <div className="serp-media-lightbox-title">{hit.title}</div>
          {hit.author ? (
            <div className="serp-media-lightbox-sub">
              {L.by} {hit.author}
            </div>
          ) : null}
          <div className="serp-media-lightbox-actions">
            {hit.downloadUrl ? (
              <a
                className="serp-btn"
                href={hit.downloadUrl}
                download
                target="_blank"
                rel="noopener noreferrer"
              >
                {L.download}
              </a>
            ) : null}
            <a className="serp-btn serp-btn--ghost" href={hit.url} target="_blank" rel="noopener noreferrer">
              {L.sourceOn(hit.sourceLabel || "מקור")}
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}
