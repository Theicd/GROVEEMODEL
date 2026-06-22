import { normalizeNewsEngineQuery } from "../groveeNews/newsQueryNormalize";
import { classifySearchIntents } from "./intents";
import { resolveUsgsMinMagnitude } from "./providers/usgsEarthquake";
import { resolveSearchHandoff, type SearchHandoff } from "./resolveSearchHandoff";
import type { LiveWorldLayer } from "../liveWorld/types";
import type { SearchIntent, SearchProviderId } from "./types";

const INTENT_PROVIDERS: Partial<Record<SearchIntent, SearchProviderId[]>> = {
  news: ["grovee-news"],
  earthquake: ["usgs-earthquake", "grovee-news", "gdacs-disasters"],
  disaster: ["gdacs-disasters", "grovee-news"],
  aviation: ["adsb-aviation"],
  ships: ["ais-ships"],
  satellite: ["iss-tracker"],
  weather: ["open-meteo"],
  marine: ["open-meteo-marine"],
};

const INTENT_LIVE_LAYERS: Partial<Record<SearchIntent, LiveWorldLayer[]>> = {
  earthquake: ["earthquake"],
  aviation: ["aviation"],
  ships: ["ships"],
  satellite: ["iss"],
};

export type LiveDataHandoff = SearchHandoff & {
  intents: SearchIntent[];
  providers: SearchProviderId[];
  liveWorldLayers: LiveWorldLayer[];
  rssEngineQuery: string;
  minMagnitude: number | null;
};

export function resolveLiveDataHandoff(query: string): LiveDataHandoff {
  const intents = classifySearchIntents(query);
  const base = resolveSearchHandoff(query);
  const providers = new Set<SearchProviderId>();
  const layers = new Set<LiveWorldLayer>();

  for (const intent of intents) {
    for (const p of INTENT_PROVIDERS[intent] ?? []) providers.add(p);
    for (const l of INTENT_LIVE_LAYERS[intent] ?? []) layers.add(l);
  }

  return {
    ...base,
    intents,
    providers: [...providers],
    liveWorldLayers: [...layers],
    rssEngineQuery: normalizeNewsEngineQuery(query),
    minMagnitude: resolveUsgsMinMagnitude(query),
  };
}
