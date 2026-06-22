import type { ChatUiLanguage } from "../ui/useUiLanguage";
import { resolveAlertLevel, resolveDisasterType } from "./disasterDisplay";
import type { UnifiedSearchHit } from "./types";

type Props = {
  hit: UnifiedSearchHit;
  uiLang: ChatUiLanguage;
};

export function isLiveDisasterHit(hit: UnifiedSearchHit): boolean {
  return hit.kind === "earthquake" || hit.kind === "disaster";
}

const eqSeverityFromMag = (mag: number): ReturnType<typeof resolveAlertLevel> => {
  if (mag >= 7) return resolveAlertLevel("Red");
  if (mag >= 5) return resolveAlertLevel("Orange");
  return resolveAlertLevel("Green");
};

export function LiveDisasterSearchResultRow({ hit, uiLang }: Props) {
  const he = uiLang === "he";

  if (hit.kind === "earthquake") {
    const mag = hit.meta?.magnitude;
    const severity = mag != null ? eqSeverityFromMag(mag) : resolveAlertLevel("Green");
    const type = resolveDisasterType("EQ");
    return (
      <article
        className={`serp-disaster-card serp-disaster-card--${severity.severity}`}
        dir={he ? "rtl" : "ltr"}
      >
        <div className={`serp-disaster-card__stripe serp-disaster-card__stripe--${severity.severity}`} />
        <div className="serp-disaster-card__body">
          <div className="serp-disaster-card__head">
            <span className={`serp-disaster-card__type ${type.css}`}>
              <span className="serp-disaster-card__type-icon" aria-hidden="true">
                {type.icon}
              </span>
              <span>{he ? type.labelHe : type.labelEn}</span>
            </span>
            <span className={`serp-disaster-card__severity ${severity.css}`}>
              {he ? severity.labelHe : severity.labelEn}
            </span>
            {mag != null ? (
              <span className="serp-disaster-card__mag" dir="ltr">
                M{mag.toFixed(1)}
              </span>
            ) : null}
          </div>
          <h3 className="serp-disaster-card__title">
            <a href={hit.url} target="_blank" rel="noopener noreferrer">
              {hit.title.replace(/^M[\d.?]+\s*·\s*/, "")}
            </a>
          </h3>
          {hit.snippet ? <p className="serp-disaster-card__meta">{hit.snippet}</p> : null}
          <div className="serp-disaster-card__foot">
            <span className="serp-disaster-card__source">USGS</span>
            <a className="serp-disaster-card__link" href={hit.url} target="_blank" rel="noopener noreferrer">
              {he ? "דוח מלא →" : "Full report →"}
            </a>
          </div>
        </div>
      </article>
    );
  }

  const type = resolveDisasterType(hit.meta?.disasterType, hit.title);
  const severity = resolveAlertLevel(hit.meta?.alertLevel);
  const country = hit.snippet?.split(" · ")[0]?.trim() || "";

  return (
    <article
      className={`serp-disaster-card serp-disaster-card--${severity.severity}`}
      dir={he ? "rtl" : "ltr"}
    >
      <div className={`serp-disaster-card__stripe serp-disaster-card__stripe--${severity.severity}`} />
      <div className="serp-disaster-card__body">
        <div className="serp-disaster-card__head">
          <span className={`serp-disaster-card__type ${type.css}`}>
            <span className="serp-disaster-card__type-icon" aria-hidden="true">
              {type.icon}
            </span>
            <span>{he ? type.labelHe : type.labelEn}</span>
          </span>
          <span
            className={`serp-disaster-card__severity ${severity.css}`}
            title={he ? severity.hintHe : severity.hintEn}
          >
            {he ? severity.labelHe : severity.labelEn}
          </span>
        </div>
        <h3 className="serp-disaster-card__title">
          <a href={hit.url} target="_blank" rel="noopener noreferrer">
            {hit.title}
          </a>
        </h3>
        {country ? (
          <p className="serp-disaster-card__location">
            <span className="serp-disaster-card__location-icon" aria-hidden="true">
              📍
            </span>
            {country}
          </p>
        ) : null}
        <p className="serp-disaster-card__hint">{he ? severity.hintHe : severity.hintEn}</p>
        <div className="serp-disaster-card__foot">
          <span className="serp-disaster-card__source">GDACS</span>
          <a className="serp-disaster-card__link" href={hit.url} target="_blank" rel="noopener noreferrer">
            {he ? "דוח מלא →" : "Full report →"}
          </a>
        </div>
      </div>
    </article>
  );
}
