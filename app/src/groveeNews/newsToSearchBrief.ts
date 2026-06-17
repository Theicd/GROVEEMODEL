import type { SearchSourceResult } from "../webSearch/types";
import { isTopicsOverviewQuery } from "./headlineIntent";
import { setNewsPanelPayload } from "./newsPanelStore";
import {
  isBroadNewsOverviewQuery,
  isSpecificNewsTopicQuery,
  normalizeNewsEngineQuery,
} from "./newsQueryNormalize";
import { buildRecentHeadlineHits } from "./recentHeadlineHits";
import { hitsToDisplayCards } from "./searchAdapter";
import { fetchTopicsBundle } from "./topicsAdapter";
import { startGroveeNewsBoot } from "./engineBoot";
import type { GroveeNewsCard } from "./types";
import { searchNews } from "./engine/engine/pipeline";
import { getEngineLibraryStats } from "./engine/engine/engineStats";
import { getSearchIndexSize } from "./engine/search/flexIndex";
import { isTopHitRelevant } from "./engine/search/relevance";



const PROVIDER = "grovee-news" as const;



function formatHeadlinesForBrief(cards: { title: string; source: string }[]): string {

  const lines = cards.slice(0, 10).map((c, i) => `[${c.source}] ${i + 1}. ${c.title}`);

  if (!lines.length) return "";

  const lead = cards[0];

  return [

    `ANSWER (headline): [${lead.source}] ${lead.title}`,

    "מקור: GROVEE NEWS (מנוע מקומי)",

    ...lines,

  ].join("\n");

}



async function searchNewsWithFallback(query: string) {
  const engineQuery = normalizeNewsEngineQuery(query);
  const specific = isSpecificNewsTopicQuery(query);

  if (specific && !engineQuery) return [];

  let hits = await searchNews(engineQuery || query);

  if (engineQuery && specific) {
    hits = hits.filter((h) => isTopHitRelevant(h.article, engineQuery));
  }

  if (!hits.length && engineQuery && engineQuery !== query && !specific) {
    hits = await searchNews(query);
  }

  if (!hits.length && (!specific || isBroadNewsOverviewQuery(engineQuery))) {
    hits = await buildRecentHeadlineHits(query);
  }

  if (!hits.length && specific && engineQuery) {
    hits = await buildRecentHeadlineHits(query);
  }

  return hits;
}



/** Search or Topics → SearchSourceResult + side panel payload. */

export async function fetchGroveeNewsSearch(query: string): Promise<SearchSourceResult> {

  const started = performance.now();

  const label = "חדשות (GROVEE NEWS)";



  try {

    await startGroveeNewsBoot();

    const stats = await getEngineLibraryStats(getSearchIndexSize());



    if (isTopicsOverviewQuery(query)) {

      const bundle = await fetchTopicsBundle();

      let cards: GroveeNewsCard[] = bundle.cards.slice(0, 24);

      if (!cards.length && stats.rssHeadlines > 0) {

        const hits = await buildRecentHeadlineHits(query, 24);

        cards = await hitsToDisplayCards(hits);

      }

      setNewsPanelPayload({

        mode: "topics",

        query,

        cards,

        generatedAt: bundle.generatedAt || Date.now(),

      });



      const text =

        cards.length > 0

          ? formatHeadlinesForBrief(cards)

          : `מאגר מקומי: ${stats.rssHeadlines} כותרות — Topics עדיין מתמלא`;



      return {

        provider: PROVIDER,

        label: `חדשות (Topics · ${bundle.stats.lanesWithHits || cards.length} נושאים)`,

        ok: cards.length > 0,

        text,

        error: cards.length ? undefined : "אין עדיין נושאים מוכנים — המתן לאיסוף ברקע",

        latencyMs: Math.round(performance.now() - started),

      };

    }



    const hits = await searchNewsWithFallback(query);

    const cards = await hitsToDisplayCards(hits);

    setNewsPanelPayload({

      mode: "search",

      query,

      cards,

      generatedAt: Date.now(),

    });



    const text =

      cards.length > 0

        ? formatHeadlinesForBrief(cards)

        : `מאגר: ${stats.rssHeadlines} כותרות RSS, אינדקס ${getSearchIndexSize()} — לא נמצאו תוצאות ל«${query.slice(0, 80)}»`;



    return {

      provider: PROVIDER,

      label,

      ok: cards.length > 0,

      text,

      url: cards[0]?.url,

      error: cards.length ? undefined : "לא נמצאו ידיעות תואמות במאגר המקומי",

      latencyMs: Math.round(performance.now() - started),

    };

  } catch (err) {

    return {

      provider: PROVIDER,

      label,

      ok: false,

      text: "",

      error: err instanceof Error ? err.message : "שגיאת מנוע חדשות",

      latencyMs: Math.round(performance.now() - started),

    };

  }

}


