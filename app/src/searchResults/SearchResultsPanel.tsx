import { useEffect, useMemo, useState, type FormEvent } from "react";

import type { GroveeNewsCard, NewsSummaryGemmaProgress } from "../groveeNews/types";

import { useUiLanguage } from "../ui/useUiLanguage";

import { MediaResultsGrid } from "./MediaResultsGrid";

import { MediaSearchResultRow } from "./MediaSearchResultRow";

import { ProductSearchResultRow, isProductHit } from "./ProductSearchResultRow";

import { SearchResultRow } from "./SearchResultRow";

import { HfModelSearchResultRow, isHfModelHit } from "./HfModelSearchResultRow";

import { isWikiHitWithImage, WikiSearchResultRow } from "./WikiSearchResultRow";

import { filterHits } from "./rankHits";

import type { SearchResultsFilter, SearchResultsPayload } from "./types";

import { useTranslatedSearchHits } from "./useTranslatedSearchHits";

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

};



const FILTERS: { id: SearchResultsFilter; labelHe: string; labelEn: string }[] = [

  { id: "all", labelHe: "הכל", labelEn: "All" },

  { id: "rss", labelHe: "חדשות", labelEn: "News" },

  { id: "web", labelHe: "אתרים", labelEn: "Web" },

  { id: "images", labelHe: "תמונות", labelEn: "Images" },

  { id: "video", labelHe: "וידאו", labelEn: "Video" },

  { id: "youtube", labelHe: "YouTube", labelEn: "YouTube" },

  { id: "movies", labelHe: "סרטים", labelEn: "Movies" },

  { id: "products", labelHe: "מוצרים", labelEn: "Products" },

  { id: "hfmodels", labelHe: "מודל HF", labelEn: "HF Models" },

];



const initialFilter = (payload: SearchResultsPayload): SearchResultsFilter => {

  if (payload.preferHfModelsFilter) return "hfmodels";

  if (payload.preferRssFilter) return "rss";

  if (payload.preferImagesFilter) return "images";

  if (payload.preferVideoFilter) return "video";

  if (payload.preferYouTubeFilter) return "youtube";

  if (payload.preferMoviesFilter) return "movies";

  if (payload.preferProductsFilter) return "products";

  return "all";

};



const emptyPanelMessage = (

  payload: SearchResultsPayload,

  filter: SearchResultsFilter,

  uiLang: "he" | "en",

): string => {

  if (!payload.query.trim() && !payload.hits.length) {

    return uiLang === "he"

      ? "הקלד שאילתה למעלה — חדשות, אתרים, תמונות, וידאו, סרטים ועוד."

      : "Type a query above — news, sites, images, video, movies, and more.";

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



export function SearchResultsPanel({

  payload,

  onClose,

  onSearch,

  searching = false,

  onSummaryReady,

}: Props) {

  const uiLang = useUiLanguage();

  const { hits: translatedHits, translating } = useTranslatedSearchHits(payload.hits, uiLang);



  const [filter, setFilter] = useState<SearchResultsFilter>(initialFilter(payload));

  const [queryDraft, setQueryDraft] = useState(payload.query);



  useEffect(() => {

    setQueryDraft(payload.query);

    setFilter(initialFilter(payload));

  }, [

    payload.query,

    payload.generatedAt,

    payload.preferRssFilter,

    payload.preferMoviesFilter,

    payload.preferImagesFilter,

    payload.preferVideoFilter,

    payload.preferYouTubeFilter,

    payload.preferProductsFilter,

    payload.preferHfModelsFilter,

  ]);



  const visible = useMemo(

    () => filterHits(translatedHits, filter),

    [translatedHits, filter],

  );



  const isMediaGrid = filter === "images" || filter === "video" || filter === "youtube";



  const facetLine = [

    payload.facets.rss ? `RSS ${payload.facets.rss}` : "",

    payload.facets.web ? `Web ${payload.facets.web}` : "",

    payload.facets.images ? `${uiLang === "he" ? "תמונות" : "Images"} ${payload.facets.images}` : "",

    payload.facets.videos ? `${uiLang === "he" ? "וידאו" : "Video"} ${payload.facets.videos}` : "",

    payload.facets.youtube ? `YouTube ${payload.facets.youtube}` : "",

    payload.facets.movies ? `${uiLang === "he" ? "סרטים" : "Movies"} ${payload.facets.movies}` : "",

    payload.facets.products ? `${uiLang === "he" ? "מוצרים" : "Products"} ${payload.facets.products}` : "",

    payload.facets.hfModels ? `HF ${payload.facets.hfModels}` : "",

    payload.facets.papers ? `arXiv ${payload.facets.papers}` : "",

    payload.facets.other ? `${uiLang === "he" ? "אחר" : "Other"} ${payload.facets.other}` : "",

  ]

    .filter(Boolean)

    .join(" · ");



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

          placeholder: "חפש מידע, תמונות, וידאו…",

          search: "חפש",

          searching: "מחפש…",

          loading: "מתרגם ומציג תוצאות…",

          close: "סגור תוצאות",

        }

      : {

          title: "Search",

          results: "results",

          placeholder: "Search news, images, video…",

          search: "Search",

          searching: "Searching…",

          loading: "Translating results…",

          close: "Close results",

        };



  return (

    <div className="serp-panel-inner" dir={uiLang === "he" ? "rtl" : "ltr"}>

      <header className="serp-panel-head">

        <div className="serp-panel-title">

          <div className="serp-panel-title-row">

            <span className="serp-panel-dot" aria-hidden="true" />

            <span>{panelLabels.title}</span>

          </div>

          <span className="serp-panel-meta">

            {payload.hits.length} {panelLabels.results}

            {facetLine ? ` · ${facetLine}` : ""}

          </span>

        </div>

        <button type="button" className="serp-panel-close" onClick={onClose} aria-label={panelLabels.close}>

          ×

        </button>

      </header>



      <form className="serp-search-form" onSubmit={submitSearch}>

        <input

          type="search"

          className="serp-search-input"

          value={queryDraft}

          onChange={(e) => setQueryDraft(e.target.value)}

          placeholder={panelLabels.placeholder}

          dir="auto"

          disabled={searching || !onSearch}

          aria-label={panelLabels.search}

        />

        <button type="submit" className="serp-search-btn" disabled={searching || !onSearch || !queryDraft.trim()}>

          {searching ? panelLabels.searching : panelLabels.search}

        </button>

      </form>



      <div className="serp-filters" role="tablist" aria-label={uiLang === "he" ? "סינון תוצאות" : "Filter results"}>

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

            {f.id === "rss" && payload.facets.rss ? ` (${payload.facets.rss})` : ""}

            {f.id === "web" && payload.facets.web ? ` (${payload.facets.web})` : ""}

            {f.id === "images" && payload.facets.images ? ` (${payload.facets.images})` : ""}

            {f.id === "video" && payload.facets.videos ? ` (${payload.facets.videos})` : ""}

            {f.id === "youtube" && payload.facets.youtube ? ` (${payload.facets.youtube})` : ""}

            {f.id === "movies" && payload.facets.movies ? ` (${payload.facets.movies})` : ""}

            {f.id === "products" && payload.facets.products ? ` (${payload.facets.products})` : ""}

            {f.id === "hfmodels" && payload.facets.hfModels ? ` (${payload.facets.hfModels})` : ""}

          </button>

        ))}

      </div>



      <div className={`serp-panel-body${isMediaGrid ? " serp-panel-body--media" : ""}`}>

        {searching || translating ? (

          <div className="serp-panel-loading">{searching ? panelLabels.searching : panelLabels.loading}</div>

        ) : null}



        {!visible.length && !searching && !translating ? (

          <div className="serp-panel-empty">{emptyPanelMessage(payload, filter, uiLang)}</div>

        ) : isMediaGrid ? (

          <MediaResultsGrid
            hits={visible}
            uiLang={uiLang}
            mode={filter === "images" ? "image" : filter === "youtube" ? "youtube" : "video"}
          />

        ) : (

          visible.map((hit) =>
            hit.kind === "image" || hit.kind === "video" || hit.kind === "youtube" ? (
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
              <HfModelSearchResultRow key={hit.id} hit={hit} uiLang={uiLang} />
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

    </div>

  );

}

