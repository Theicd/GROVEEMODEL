import { fetchGdacsDisastersForCache } from "../realityData/providers/disasters";
import { fetchUsgsEarthquakesForCache } from "../webSearch/providers/usgsEarthquake";
import { disasterItemToHit, earthquakeItemToHit } from "../searchResults/liveDisastersHits";
import type { UnifiedSearchHit } from "../searchResults/types";
import type { GlobeAlertEvent } from "./types";
import { fetchSpaceTabEvents } from "./mapSpaceToGlobeEvents";
import { hitsToGlobeEvents } from "./mapHitsToGlobeEvents";
import { filterEventsForRange, type AlertTimeRange } from "./alertTimeRange";
import { EQ_LIVE_WINDOW_MS, GLOBAL_ALERTS_EQ_MIN_MAG } from "./types";

/** Fetch live earth alerts or upcoming space alerts for the selected tab. */
export async function fetchGlobalAlertEventsForRange(
  range: AlertTimeRange = "live",
): Promise<GlobeAlertEvent[]> {
  const now = Date.now();

  if (range === "space") {
    const space = await fetchSpaceTabEvents().catch(() => []);
    return filterEventsForRange(space, "space", now);
  }

  const [usgs, gdacs] = await Promise.all([
    fetchUsgsEarthquakesForCache(GLOBAL_ALERTS_EQ_MIN_MAG, "hour", 14_000).catch(() => null),
    fetchGdacsDisastersForCache().catch(() => null),
  ]);

  const hits: UnifiedSearchHit[] = [];
  if (usgs) {
    for (const [i, item] of usgs.items.entries()) {
      if (item.lat == null || item.lon == null) continue;
      if (now - item.time > EQ_LIVE_WINDOW_MS) continue;
      hits.push(earthquakeItemToHit(item, i, usgs.feedLabel));
    }
  }
  if (gdacs) {
    for (const [i, item] of gdacs.items.entries()) {
      if (item.lat == null || item.lon == null) continue;
      hits.push(disasterItemToHit(item, i, gdacs.feedLabel));
    }
  }

  const raw = hitsToGlobeEvents(hits);
  return filterEventsForRange(raw, "live", now);
}

/** @deprecated use fetchGlobalAlertEventsForRange("live") */
export async function fetchGlobalAlertEvents(): Promise<GlobeAlertEvent[]> {
  return fetchGlobalAlertEventsForRange("live");
}
