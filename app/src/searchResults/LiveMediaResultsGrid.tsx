import { useState, type KeyboardEvent } from "react";
import type { ChatUiLanguage } from "../ui/useUiLanguage";
import type { UnifiedSearchHit } from "./types";
import { MediaLightbox } from "./MediaLightbox";
import { categoryLabelEn, categoryLabelHe, type UserChannelCategory } from "../liveMedia/channelUserTaxonomy";
import { channelMayHaveEpg, hitToEpgLookup } from "../liveMedia/epg/epgService";

type Props = {
  hits: UnifiedSearchHit[];
  uiLang: ChatUiLanguage;
  mode: "livetv" | "radio";
  favoriteIds?: Set<string>;
  onToggleFavorite?: (hit: UnifiedSearchHit) => void;
  onHideChannel?: (hit: UnifiedSearchHit) => void;
  onEditChannel?: (hit: UnifiedSearchHit) => void;
  showCategoryBadge?: boolean;
  showEpgBadge?: boolean;
};

export function LiveMediaResultsGrid({
  hits,
  uiLang,
  mode,
  favoriteIds,
  onToggleFavorite,
  onHideChannel,
  onEditChannel,
  showCategoryBadge = false,
  showEpgBadge = false,
}: Props) {
  const [active, setActive] = useState<UnifiedSearchHit | null>(null);
  const liveLabel = uiLang === "he" ? "שידור חי" : "LIVE";
  const rtl = uiLang === "he";

  const openPlay = (hit: UnifiedSearchHit) => setActive(hit);

  const onThumbKey = (hit: UnifiedSearchHit, e: KeyboardEvent) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      openPlay(hit);
    }
  };

  return (
    <>
      <div className="serp-media-grid serp-live-grid" role="list">
        {hits.map((hit) => {
          const userCat = hit.meta?.userCategory as UserChannelCategory | undefined;
          const epgLookup = showEpgBadge && hit.kind === "livetv" ? hitToEpgLookup(hit) : null;
          const mayHaveEpg =
            epgLookup != null
              ? channelMayHaveEpg(epgLookup.title, epgLookup.tvgId, epgLookup.streamUrl)
              : false;
          const editable = Boolean(onEditChannel && mode === "livetv");
          return (
            <div
              key={hit.id}
              className={`serp-media-card serp-media-card--${mode} serp-live-card`}
              role="listitem"
            >
              <div
                className="serp-live-thumb-btn"
                role="button"
                tabIndex={0}
                onClick={() => openPlay(hit)}
                onKeyDown={(e) => onThumbKey(hit, e)}
                aria-label={hit.title}
              >
                <span className="serp-media-thumb-wrap serp-live-thumb-wrap">
                  {hit.imageUrl ? (
                    <img
                      className="serp-media-thumb"
                      src={hit.imageUrl}
                      alt=""
                      loading="lazy"
                      referrerPolicy="no-referrer"
                    />
                  ) : (
                    <span
                      className={`serp-media-thumb serp-media-thumb--placeholder serp-live-placeholder serp-live-placeholder--${mode}`}
                    />
                  )}
                  <span className="serp-live-badge">{liveLabel}</span>
                  {showCategoryBadge && userCat ? (
                    <span className="serp-live-cat-badge">
                      {rtl ? categoryLabelHe(userCat) : categoryLabelEn(userCat)}
                    </span>
                  ) : null}
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
                  {editable ? (
                    <button
                      type="button"
                      className="serp-live-edit"
                      aria-label={uiLang === "he" ? "ערוך ערוץ" : "Edit channel"}
                      title={uiLang === "he" ? "ערוך שם, קטגוריה ושפה" : "Edit name, category, language"}
                      onClick={(e) => {
                        e.stopPropagation();
                        onEditChannel!(hit);
                      }}
                    >
                      ✎
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
              </div>
              <span className="serp-media-meta">
                {editable ? (
                  <button
                    type="button"
                    className="serp-media-title serp-media-title--editable"
                    title={uiLang === "he" ? "לחץ לעריכת שם הערוץ" : "Click to edit channel name"}
                    onClick={() => onEditChannel!(hit)}
                  >
                    {hit.title}
                  </button>
                ) : (
                  <span className="serp-media-title">{hit.title}</span>
                )}
                <span className="serp-live-sub">{hit.snippet}</span>
                {showEpgBadge && epgLookup ? (
                  <span className={`serp-live-epg-badge serp-live-epg-badge--${mayHaveEpg ? "yes" : "no"}`}>
                    {mayHaveEpg
                      ? rtl
                        ? "לוח שידורים זמין"
                        : "EPG available"
                      : rtl
                        ? "ללא לוח שידורים"
                        : "No EPG"}
                  </span>
                ) : null}
              </span>
            </div>
          );
        })}
      </div>

      {active ? (
        <MediaLightbox hit={active} uiLang={uiLang} onClose={() => setActive(null)} />
      ) : null}
    </>
  );
}
