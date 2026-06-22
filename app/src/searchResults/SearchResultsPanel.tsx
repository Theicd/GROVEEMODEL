import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";

import type { GroveeNewsCard, NewsSummaryGemmaProgress } from "../groveeNews/types";

import { useUiLanguage } from "../ui/useUiLanguage";

import { MediaResultsGrid } from "./MediaResultsGrid";

import { LiveMediaResultsGrid } from "./LiveMediaResultsGrid";

import { MediaSearchResultRow } from "./MediaSearchResultRow";

import { ProductSearchResultRow, isProductHit } from "./ProductSearchResultRow";

import { LiveDisasterSearchResultRow, isLiveDisasterHit } from "./LiveDisasterSearchResultRow";

import { LiveShipSearchResultRow, isLiveShipHit } from "./LiveShipSearchResultRow";

import { SearchResultRow } from "./SearchResultRow";

import { HfModelSearchResultRow, isHfModelHit } from "./HfModelSearchResultRow";

import { isWikiHitWithImage, WikiSearchResultRow } from "./WikiSearchResultRow";

import { filterHits } from "./rankHits";

import type { SearchResultsFilter, SearchResultsPayload } from "./types";

import { isAisStreamConfigured } from "../apiKeys/apiKeyStore";

import { useTranslatedSearchHits, hitsNeedTranslation } from "./useTranslatedSearchHits";

import { GroVeeSearchLogo } from "./GroVeeSearchLogo";

import "./searchResults.css";



type Props = {

  payload: SearchResultsPayload;

  onClose: () => void;

  onSearch?: (query: string) => void | Promise<void>;

  searching?: boolean;

  onSummaryReady?: (

    card: GroveeNewsCard,

    gemmaInput: string,

    progress?: NewsSummaryGemmaProgress,

  ) => Promise<string>;

  onHfAddedToRack?: () => void;

};



const FILTERS: { id: SearchResultsFilter; labelHe: string; labelEn: string }[] = [

  { id: "all", labelHe: "הכל", labelEn: "All" },

  { id: "rss", labelHe: "חדשות", labelEn: "News" },

  { id: "images", labelHe: "תמונות", labelEn: "Images" },

  { id: "video", labelHe: "וידאו", labelEn: "Video" },

  { id: "events", labelHe: "אירועים", labelEn: "Events" },

  { id: "products", labelHe: "מוצרים", labelEn: "Products" },

  { id: "movies", labelHe: "סרטים", labelEn: "Movies" },

  { id: "hfmodels", labelHe: "מודל HF", labelEn: "HF Models" },

  { id: "ships", labelHe: "אוניות / ים", labelEn: "Ships" },

];



const emptyPanelMessage = (

  payload: SearchResultsPayload,

  filter: SearchResultsFilter,

  uiLang: "he" | "en",

): string => {

  if (!payload.query.trim() && !payload.hits.length) {

    return uiLang === "he"

      ? "הקלד שאילתה למעלה — חדשות, רעידות אדמה, אסונות, אתרים, תמונות, וידאו, סרטים ועוד."

      : "Type a query above — news, earthquakes, disasters, sites, images, video, movies, and more.";

  }

  if (filter === "events" || filter === "earthquakes" || filter === "disasters" || filter === "weather") {
    return uiLang === "he"
      ? "אין נתוני אירועים כרגע — USGS/GDACS/Open-Meteo מתעדכנים ברקע. נסה שוב בעוד דקה."
      : "No live events yet — USGS/GDACS/Open-Meteo refresh in the background. Try again shortly.";
  }

  if (filter === "ships") {
    const shipErr = payload.providerErrors.find((e) => /AIS|Digitraffic|Overpass|ספינ|אוני/i.test(e));
    if (shipErr) return shipErr.replace(/^[^:]+:\s*/, "");
    if (uiLang === "he") {
      if (!isAisStreamConfigured()) {
        return "אין כלי שייט חיים — הוסף מפתח AISStream (🔑) והרץ npm run dev; Digitraffic מכסה בעיקר את הבלטי.";
      }
      return "טוען AIS חי… אם אין תוצאות אחרי ~20 שניות, לחץ «בדוק חיבור» במסך המפתחות ורענן את הדף.";
    }
    return isAisStreamConfigured()
      ? "Loading live AIS… If empty after ~20s, test the key in API keys (🔑) and refresh."
      : "No live vessels — add AISStream key (🔑) and run npm run dev; Digitraffic covers Baltic mainly.";
  }

  if (filter === "livetv" || filter === "radio") {
    const liveErr = payload.providerErrors.find((e) => /TV LIVE|Radio|live-tv|קטלוג/i.test(e));
    if (liveErr) return liveErr.replace(/^[^:]+:\s*/, "");
    return uiLang === "he"
      ? "אין תוצאות בערוצים/רדיו — לחץ «TV/רדיו» למעלה, סנכרן מקורות, והרץ בדיקת QA."
      : "No live TV/radio matches — open «TV/Radio» control, sync sources, and run QA.";
  }

  if (filter === "rss") {
    const newsErr = payload.providerErrors.find((e) => /GROVEE NEWS|חדשות|RSS|מאגר/i.test(e));
    if (newsErr) return newsErr.replace(/^[^:]+:\s*/, "");
    return uiLang === "he"
      ? "אין כותרות RSS — המתן לסריקה חיה או בדוק חיבור לרשת."
      : "No RSS headlines — wait for live scan or check network.";
  }

  if (filter === "companion") {
    if (payload.companionWebError) {
      return payload.companionWebError.replace(/^[^:]+:\s*/, "");
    }
    return uiLang === "he"
      ? "אין תוצאות מ-OpenSERP — הפעל «Grove Search» משולחן העבודה ובדוק ב-🧩 תוספים → «בדוק חיפוש»."
      : "No OpenSERP results — start Grove Search from Desktop and test in Plugins (🧩).";
  }

  if (filter === "web") {
    const webErr = payload.providerErrors.find((e) => /Tavily|Scavio|SearXNG/i.test(e));
    if (webErr) return webErr.replace(/^[^:]+:\s*/, "");
  }

  if (payload.hits.length && filter !== "all") {

    return uiLang === "he" ? "אין תוצאות בסינון זה." : "No results in this filter.";

  }

  const searxMissing = payload.providerErrors.some((e) => /SearXNG/i.test(e));

  if (searxMissing) {

    return uiLang === "he"

      ? "לא נמצאו תוצאות באתרים — SearXNG לא מוגדר. עדיין ייתכנו תוצאות מ-RSS, תמונות ווידאו."

      : "No web results — SearXNG is not configured. RSS, images and video may still appear.";

  }

  return uiLang === "he"

    ? "לא נמצאו תוצאות — נסה מילות חיפוש אחרות."

    : "No results found — try different keywords.";

};



/** One-line status — only for the active filter tab (keeps header compact). */
const filterContextHint = (
  payload: SearchResultsPayload,
  filter: SearchResultsFilter,
  uiLang: "he" | "en",
): string | null => {
  if (filter === "companion") {
    if (payload.companionWebError) return payload.companionWebError.replace(/^[^:]+:\s*/, "");
    if (payload.facets.companionWeb) {
      return uiLang === "he"
        ? `${payload.facets.companionWeb} תוצאות OpenSERP — כל כרטיס מסומן בירוק`
        : `${payload.facets.companionWeb} OpenSERP results`;
    }
    return null;
  }
  if (filter === "rss" && payload.newsRssNote) return payload.newsRssNote;
  if (
    (filter === "earthquakes" || filter === "disasters" || filter === "events" || filter === "weather") &&
    payload.liveDisastersNote
  ) {
    return payload.liveDisastersNote;
  }
  if (filter === "ships" && payload.liveShipsNote) return payload.liveShipsNote;
  if (filter === "web") {
    const webErr = payload.providerErrors.find((e) => /Tavily|Scavio|SearXNG/i.test(e));
    if (webErr) return webErr.replace(/^[^:]+:\s*/, "");
  }
  return null;
};



export function SearchResultsPanel({

  payload,

  onClose,

  onSearch,

  searching = false,

  onSummaryReady,

  onHfAddedToRack,

}: Props) {

  const uiLang = useUiLanguage();

  const { hits: translatedHits, translating } = useTranslatedSearchHits(payload.hits, uiLang);



  const [filter, setFilter] = useState<SearchResultsFilter>("all");

  const [queryDraft, setQueryDraft] = useState(payload.query);

  const filtersScrollRef = useRef<HTMLDivElement>(null);

  const lastQueryRef = useRef(payload.query);

  useEffect(() => {

    setQueryDraft(payload.query);

    if (payload.query !== lastQueryRef.current) {

      lastQueryRef.current = payload.query;

      setFilter("all");

    }

  }, [payload.query]);



  const visible = useMemo(

    () => filterHits(translatedHits, filter),

    [translatedHits, filter],

  );

  const visibleNeedsTranslation = useMemo(
    () => hitsNeedTranslation(visible, uiLang),
    [visible, uiLang],
  );



  const isMediaGrid = filter === "images" || filter === "video";

  const isLiveGrid = filter === "livetv" || filter === "radio";

  const contextHint = filterContextHint(payload, filter, uiLang);

  const allTabCount = useMemo(
    () => filterHits(translatedHits, "all").length,
    [translatedHits],
  );

  const videoTabCount = useMemo(
    () => filterHits(translatedHits, "video").length,
    [translatedHits],
  );

  const eventsTabCount = useMemo(
    () => filterHits(translatedHits, "events").length,
    [translatedHits],
  );

  const tabTotal = useMemo(
    () => (filter === "all" ? allTabCount : filterHits(translatedHits, filter).length),
    [filter, allTabCount, translatedHits],
  );

  const visibleCountLabel = String(tabTotal);

  const scrollFilters = (toward: "start" | "end") => {
    const el = filtersScrollRef.current;
    if (!el) return;
    const step = 160;
    const rtl = uiLang === "he";
    const delta = toward === "start" ? (rtl ? step : -step) : rtl ? -step : step;
    el.scrollBy({ left: delta, behavior: "smooth" });
  };

  const submitSearch = (e: FormEvent) => {

    e.preventDefault();

    const q = queryDraft.trim();

    if (!q || searching || !onSearch) return;

    void onSearch(q);

  };



  const panelLabels =

    uiLang === "he"

      ? {

          title: "חיפוש",

          results: "תוצאות",

          placeholder: payload.query.trim() || "חפש…",

          search: "חיפוש",

          searching: "מחפש…",

          loading: "מתרגם ומציג תוצאות…",

          close: "סגור תוצאות",

        }

      : {

          title: "Search",

          results: "results",

          placeholder: payload.query.trim() || "Search…",

          search: "Search",

          searching: "Searching…",

          loading: "Translating results…",

          close: "Close results",

        };



  const isLanding = !payload.query.trim() && !searching;

  const searchForm = (variant: "landing" | "compact") => (
    <div className={`serp-search-glow-wrap serp-search-glow-wrap--${variant}`}>
      <div className="serp-search-glow" aria-hidden="true" />
      <form
        className={`serp-search-form serp-search-form--glow serp-search-form--${variant}`}
        onSubmit={submitSearch}
      >
        <span className="serp-search-leading" aria-hidden="true">
          ⌕
        </span>
        <input
          type="search"
          className={`serp-search-input serp-search-input--glow${variant === "landing" ? " serp-search-input--landing" : ""}`}
          value={queryDraft}
          onChange={(e) => setQueryDraft(e.target.value)}
          placeholder={variant === "landing" ? (uiLang === "he" ? "חפש ב-GroVee" : "Search GroVee") : panelLabels.placeholder}
          dir="auto"
          disabled={searching || !onSearch}
          aria-label={panelLabels.search}
          autoFocus={variant === "landing"}
        />
        {variant === "compact" ? (
          <button
            type="submit"
            className="serp-search-btn serp-search-btn--glow"
            disabled={searching || !onSearch || !queryDraft.trim()}
          >
            {searching ? panelLabels.searching : panelLabels.search}
          </button>
        ) : null}
      </form>
    </div>
  );

  return (

    <div
      className={`serp-panel-inner${isLanding ? " serp-panel-inner--landing" : " serp-panel-inner--results"}`}
      dir={uiLang === "he" ? "rtl" : "ltr"}
    >

      {isLanding ? (
        <>
          <button
            type="button"
            className="serp-panel-back serp-panel-back--landing"
            onClick={onClose}
            aria-label={uiLang === "he" ? "חזרה לשיחה" : "Back to chat"}
            title={uiLang === "he" ? "חזרה" : "Back"}
          >
            <span className="serp-panel-back-icon" aria-hidden="true">
              →
            </span>
          </button>
        <div className="serp-landing">
          <GroVeeSearchLogo />
          {searchForm("landing")}
          <p className="serp-landing-tagline">
            {uiLang === "he"
              ? "חדשות · תמונות · וידאו · אירועים · מוצרים · ועוד"
              : "News · images · video · events · products · and more"}
          </p>
          <div className="serp-landing-actions">
            <button
              type="button"
              className="serp-landing-btn serp-landing-btn--primary"
              disabled={searching || !onSearch || !queryDraft.trim()}
              onClick={() => {
                const q = queryDraft.trim();
                if (q && onSearch) void onSearch(q);
              }}
            >
              {uiLang === "he" ? "חיפוש GroVee" : "GroVee Search"}
            </button>
          </div>
        </div>
        </>
      ) : (
        <>
      <header className="serp-google-head">
        <button
          type="button"
          className="serp-panel-back"
          onClick={onClose}
          aria-label={uiLang === "he" ? "חזרה לשיחה" : "Back to chat"}
          title={uiLang === "he" ? "חזרה" : "Back"}
        >
          <span className="serp-panel-back-icon" aria-hidden="true">
            →
          </span>
        </button>
        {searchForm("compact")}
      </header>

      <div className="serp-google-meta">
        <GroVeeSearchLogo compact />
        <span className="serp-google-meta-count">
          {uiLang === "he" ? `כ-${visibleCountLabel} תוצאות` : `About ${visibleCountLabel} results`}
        </span>
        {searching ? (
          <span className="serp-google-meta-status">{panelLabels.searching}</span>
        ) : null}
      </div>

      <div className="serp-filters-wrap">
        <button
          type="button"
          className="serp-filters-scroll serp-filters-scroll--prev"
          onClick={() => scrollFilters("start")}
          aria-label={uiLang === "he" ? "לשוניות קודמות" : "Previous tabs"}
        >
          ‹
        </button>
        <div
          className="serp-filters"
          ref={filtersScrollRef}
          role="tablist"
          aria-label={uiLang === "he" ? "סינון תוצאות" : "Filter results"}
        >

        {FILTERS.map((f) => (

          <button

            key={f.id}

            type="button"

            role="tab"

            aria-selected={filter === f.id}

            className={`serp-filter-btn${filter === f.id ? " serp-filter-btn--active" : ""}${f.id === "hfmodels" ? " serp-filter-btn--hf" : ""}`}

            onClick={() => setFilter(f.id)}

          >

            {uiLang === "he" ? f.labelHe : f.labelEn}

            {f.id === "all" && allTabCount ? ` (${allTabCount})` : ""}

            {f.id === "rss" && payload.facets.rss ? ` (${payload.facets.rss})` : ""}

            {f.id === "images" && payload.facets.images ? ` (${payload.facets.images})` : ""}

            {f.id === "video" && videoTabCount ? ` (${videoTabCount})` : ""}

            {f.id === "events" && eventsTabCount ? ` (${eventsTabCount})` : ""}

            {f.id === "products" && payload.facets.products ? ` (${payload.facets.products})` : ""}

            {f.id === "ships" && payload.facets.ships ? ` (${payload.facets.ships})` : ""}

            {f.id === "movies" && payload.facets.movies ? ` (${payload.facets.movies})` : ""}

            {f.id === "hfmodels" && payload.facets.hfModels ? ` (${payload.facets.hfModels})` : ""}

          </button>

        ))}

        </div>
        <button
          type="button"
          className="serp-filters-scroll serp-filters-scroll--next"
          onClick={() => scrollFilters("end")}
          aria-label={uiLang === "he" ? "לשוניות הבאות" : "Next tabs"}
        >
          ›
        </button>
      </div>

      {contextHint ? (
        <p className="serp-context-hint" role="status">
          {contextHint}
        </p>
      ) : null}

      <div className={`serp-panel-body${isMediaGrid || isLiveGrid ? " serp-panel-body--media" : ""}`}>

        {searching || (translating && visibleNeedsTranslation) ? (

          <div className="serp-panel-loading">{searching ? panelLabels.searching : panelLabels.loading}</div>

        ) : null}



        {!visible.length && !searching && !(translating && visibleNeedsTranslation) ? (

          <div className="serp-panel-empty">{emptyPanelMessage(payload, filter, uiLang)}</div>

        ) : isLiveGrid ? (

          <LiveMediaResultsGrid
            hits={visible}
            uiLang={uiLang}
            mode={filter === "radio" ? "radio" : "livetv"}
          />

        ) : isMediaGrid ? (

          <MediaResultsGrid
            hits={visible}
            uiLang={uiLang}
            mode={filter === "images" ? "image" : "video"}
          />

        ) : (

          visible.map((hit) =>
            isLiveDisasterHit(hit) ? (
              <LiveDisasterSearchResultRow key={hit.id} hit={hit} uiLang={uiLang} />
            ) : isLiveShipHit(hit) ? (
              <LiveShipSearchResultRow key={hit.id} hit={hit} uiLang={uiLang} />
            ) : hit.kind === "image" || hit.kind === "video" || hit.kind === "youtube" || hit.kind === "livetv" || hit.kind === "radio" ? (
              <MediaSearchResultRow key={hit.id} hit={hit} uiLang={uiLang} />
            ) : isWikiHitWithImage(hit) ? (
              <WikiSearchResultRow
                key={hit.id}
                hit={hit}
                uiLang={uiLang}
                onSummaryReady={onSummaryReady}
              />
            ) : isProductHit(hit) ? (
              <ProductSearchResultRow key={hit.id} hit={hit} uiLang={uiLang} />
            ) : isHfModelHit(hit) ? (
              <HfModelSearchResultRow
                key={hit.id}
                hit={hit}
                uiLang={uiLang}
                onAddedToRack={onHfAddedToRack}
              />
            ) : (
              <SearchResultRow
                key={hit.id}
                hit={hit}
                uiLang={uiLang}
                onSummaryReady={onSummaryReady}
              />
            ),
          )

        )}

      </div>

        </>
      )}

    </div>

  );

}

