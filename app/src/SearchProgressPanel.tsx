import type { SearchBrief, SearchSourceResult } from "./webSearch/types";
import { CONNECTED_SEARCH_PROVIDERS } from "./webSearch/searchProviders";

type SearchPanelProps = {
  active: boolean;
  query?: string;
  sources: SearchSourceResult[];
  summary?: string;
  brief?: SearchBrief;
};

const PROVIDER_ICONS: Record<string, string> = {
  "open-meteo": "🌤",
  "world-time": "🕐",
  github: "🐙",
  "wikipedia-en": "📚",
  "wikipedia-he": "📚",
  "rest-countries": "🌍",
  "frankfurter-fx": "💱",
  "nominatim-places": "📍",
  "usgs-earthquake": "🌋",
  coingecko: "₿",
  "hacker-news": "📰",
  "news-rss": "📰",
  "adsb-aviation": "✈",
  "ais-ships": "⛴",
  "osm-overpass-marine": "⚓",
  celestrak: "🛰",
  "starlink-catalog": "🛰",
  "spacex-launches": "🚀",
  "iss-tracker": "🛰",
  "open-meteo-marine": "🌊",
  "huggingface-models": "🤗",
  "huggingface-datasets": "🤗",
  "noaa-space": "☀",
  "israel-alerts": "🚨",
  "gdacs-disasters": "🌀",
  "market-stocks": "📈",
  reddit: "🔴",
  "flight-status": "🛫",
  "wikidata-gov": "🏛",
};

/** Compact Perplexity-style search chip — one row collapsed, expand for details. */
export function SearchProgressPanel({ active, query, sources, summary, brief }: SearchPanelProps) {
  if (!active && !sources.length) return null;

  const ok = sources.filter((s) => s.ok && s.text.trim());
  const failed = sources.filter((s) => !s.ok && s.error);
  const okLabels = ok.map((s) => s.label).join(" · ");
  const chipSummary =
    summary ??
    (active
      ? "מחפש…"
      : ok.length
        ? `${ok.length} מקורות${okLabels ? `: ${okLabels}` : ""}`
        : "אין תוצאות");

  return (
    <details
      className={`search-chip${active ? " search-chip--live" : ""}`}
      dir="rtl"
      open={active}
    >
      <summary className="search-chip-summary">
        <span className={`search-chip-dot${active ? " search-chip-dot--live" : ok.length ? " search-chip-dot--ok" : " search-chip-dot--fail"}`} />
        <span className="search-chip-label">{active ? "מחפש ברשת…" : "מקורות"}</span>
        <span className="search-chip-meta">{chipSummary}</span>
      </summary>

      <div className="search-chip-body">
        {query ? <div className="search-chip-query">{query}</div> : null}

        {sources.length ? (
          <ul className="search-chip-sources">
            {sources.map((s) => {
              const icon = PROVIDER_ICONS[s.provider] ?? "🔍";
              const state = s.ok ? "ok" : s.error ? "fail" : "pending";
              return (
                <li key={`${s.provider}-${s.label}`} className={`search-chip-source search-chip-source--${state}`}>
                  <span>{icon}</span>
                  <span className="search-chip-source-name">{s.label}</span>
                  {s.ok ? (
                    <span className="search-chip-source-ms">{s.latencyMs}ms</span>
                  ) : s.error ? (
                    <span className="search-chip-source-ms search-chip-source-ms--fail" title={s.error}>
                      ✗
                    </span>
                  ) : (
                    <span className="search-chip-source-ms">…</span>
                  )}
                </li>
              );
            })}
          </ul>
        ) : active ? (
          <p className="search-chip-wait">מנתב שאילתה…</p>
        ) : null}

        {brief?.facts.length && !active ? (
          <ul className="search-chip-facts">
            {brief.facts.slice(0, 3).map((f) => (
              <li key={f}>{f}</li>
            ))}
          </ul>
        ) : null}

        {ok.map((s) =>
          s.url ? (
            <a
              key={`link-${s.provider}`}
              className="search-chip-link"
              href={s.url}
              target="_blank"
              rel="noopener noreferrer"
            >
              {PROVIDER_ICONS[s.provider] ?? "🔗"} {s.label}
            </a>
          ) : null,
        )}

        {failed.length ? (
          <p className="search-chip-failed">
            {failed.map((f) => f.label).join(" · ")}
          </p>
        ) : null}

        {!active ? (
          <details className="search-chip-catalog">
            <summary>מקורות חינמיים מחוברים (ללא Google)</summary>
            <ul className="search-chip-catalog-list">
              {CONNECTED_SEARCH_PROVIDERS.map((p) => (
                <li key={p.id}>
                  <span>{p.icon}</span> {p.labelHe}
                </li>
              ))}
            </ul>
            <p className="search-chip-catalog-note">
              עולם חי (🌐): מטוסים · ספינות · לוויינים · רעידות — לצפייה: «הצג על המפה»
            </p>
          </details>
        ) : null}
      </div>
    </details>
  );
}
