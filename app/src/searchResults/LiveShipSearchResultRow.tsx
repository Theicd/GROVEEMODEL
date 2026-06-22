import type { ChatUiLanguage } from "../ui/useUiLanguage";
import type { UnifiedSearchHit } from "./types";

type Props = {
  hit: UnifiedSearchHit;
  uiLang: ChatUiLanguage;
};

export function isLiveShipHit(hit: UnifiedSearchHit): boolean {
  return hit.kind === "ship" || hit.kind === "marine";
}

const sourceCss = (source?: string): string => {
  if (source === "ais") return "serp-ship-card__badge--ais";
  if (source === "aisstream") return "serp-ship-card__badge--aisstream";
  if (source === "globe") return "serp-ship-card__badge--globe";
  if (source === "route-marker") return "serp-ship-card__badge--demo";
  return "serp-ship-card__badge--infra";
};

const infraIcon = (kind?: string): string => {
  switch (kind) {
    case "harbour":
      return "⚓";
    case "buoy":
      return "🔴";
    case "lighthouse":
      return "🗼";
    case "pier":
      return "🛟";
    default:
      return "🌊";
  }
};

export function LiveShipSearchResultRow({ hit, uiLang }: Props) {
  const he = uiLang === "he";
  const lat = hit.meta?.shipLat;
  const lon = hit.meta?.shipLon;
  const hasCoords = lat != null && lon != null;

  if (hit.kind === "marine") {
    const kind = hit.meta?.marineInfraKind;
    return (
      <article className="serp-ship-card serp-ship-card--infra" dir={he ? "rtl" : "ltr"}>
        <div className="serp-ship-card__stripe serp-ship-card__stripe--infra" />
        <div className="serp-ship-card__body">
          <div className="serp-ship-card__head">
            <span className="serp-ship-card__icon" aria-hidden="true">
              {infraIcon(kind)}
            </span>
            <span className={`serp-ship-card__badge ${sourceCss()}`}>
              {hit.meta?.engine ?? (he ? "תשתית ימית" : "Marine infra")}
            </span>
            {hit.meta?.regionLabel ? (
              <span className="serp-ship-card__region">{hit.meta.regionLabel}</span>
            ) : null}
          </div>
          <h3 className="serp-ship-card__title">
            <a href={hit.url} target="_blank" rel="noopener noreferrer">
              {hit.title}
            </a>
          </h3>
          {hasCoords ? (
            <p className="serp-ship-card__coords" dir="ltr">
              📍 {lat!.toFixed(2)}°, {lon!.toFixed(2)}°
            </p>
          ) : null}
          <p className="serp-ship-card__hint">
            {he
              ? "נתונים סטטיים מ-OpenStreetMap — לא AIS חי."
              : "Static OpenStreetMap data — not live AIS."}
          </p>
          <div className="serp-ship-card__foot">
            <span className="serp-ship-card__source">OSM</span>
            <a className="serp-ship-card__link" href={hit.url} target="_blank" rel="noopener noreferrer">
              {he ? "מפה →" : "Map →"}
            </a>
          </div>
        </div>
      </article>
    );
  }

  const shipSource = hit.meta?.shipSource;
  const speed = hit.meta?.speedKn;
  const dest = hit.meta?.destination;

  return (
    <article className="serp-ship-card serp-ship-card--vessel" dir={he ? "rtl" : "ltr"}>
      <div className="serp-ship-card__stripe serp-ship-card__stripe--vessel" />
      <div className="serp-ship-card__body">
        <div className="serp-ship-card__head">
          <span className="serp-ship-card__icon" aria-hidden="true">
            ⛴
          </span>
          <span className={`serp-ship-card__badge ${sourceCss(shipSource)}`}>
            {hit.meta?.engine ?? (he ? "AIS" : "AIS")}
          </span>
          {speed != null ? (
            <span className="serp-ship-card__speed" dir="ltr">
              {speed.toFixed(1)} kn
            </span>
          ) : null}
        </div>
        <h3 className="serp-ship-card__title">
          <a href={hit.url} target="_blank" rel="noopener noreferrer">
            {hit.title}
          </a>
        </h3>
        {hasCoords ? (
          <p className="serp-ship-card__coords" dir="ltr">
            📍 {lat!.toFixed(2)}°, {lon!.toFixed(2)}°
          </p>
        ) : null}
        {dest ? (
          <p className="serp-ship-card__dest">
            {he ? "יעד:" : "Dest:"} {dest}
          </p>
        ) : null}
        {hit.meta?.regionLabel ? (
          <p className="serp-ship-card__region-line">{hit.meta.regionLabel}</p>
        ) : null}
        <div className="serp-ship-card__foot">
          <span className="serp-ship-card__source">
            {shipSource === "aisstream" ? "AISStream" : "Digitraffic AIS"}
          </span>
          <a className="serp-ship-card__link" href={hit.url} target="_blank" rel="noopener noreferrer">
            {he ? "מיקום במפה →" : "Map →"}
          </a>
        </div>
      </div>
    </article>
  );
}
