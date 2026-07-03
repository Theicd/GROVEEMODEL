import { fetchGdacsDisastersForCache } from "../realityData/providers/disasters";
import { fetchUsgsEarthquakesForCache } from "../webSearch/providers/usgsEarthquake";
import { disasterItemToHit, earthquakeItemToHit } from "../searchResults/liveDisastersHits";
import type { UnifiedSearchHit } from "../searchResults/types";
import type { GlobeAlertEvent } from "./types";
import { fetchSpaceGlobeEvents } from "./mapSpaceToGlobeEvents";
import { hitsToGlobeEvents } from "./mapHitsToGlobeEvents";
import { filterSidebarAlerts } from "./alertFilters";
import { EARTH_LIVE_WINDOW_MS, EQ_LIVE_WINDOW_MS, GLOBAL_ALERTS_EQ_MIN_MAG } from "./types";

/** Fetch live USGS + GDACS + NASA JPL space events. */
export async function fetchGlobalAlertEvents(): Promise<GlobeAlertEvent[]> {
  const [usgs, gdacs, space] = await Promise.all([
    fetchUsgsEarthquakesForCache(GLOBAL_ALERTS_EQ_MIN_MAG, "hour"),
    fetchGdacsDisastersForCache(),
    fetchSpaceGlobeEvents().catch(() => []),
  ]);
  const hits: UnifiedSearchHit[] = [];
  if (usgs) {
    const now = Date.now();
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
  return filterSidebarAlerts(hitsToGlobeEvents(hits).concat(space));
}
