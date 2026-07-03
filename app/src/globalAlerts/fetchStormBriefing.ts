import { fetchJson } from "../webSearch/fetchJson";
import { parseStormGeometry, stormPositionNow, type StormTrack } from "./parseStormGeometry";

export type GdacsAffectedCountry = {
  name: string;
  iso2?: string;
  iso3?: string;
};

export type StormBriefing = {
  track: StormTrack;
  eventName?: string;
  gdacsCountry?: string;
  gdacsCountryOnLand?: string;
  affectedCountries: GdacsAffectedCountry[];
  currentPos: { lat: number; lon: number };
  forecastTarget?: { lat: number; lon: number; label?: string; time?: number };
};

type GdacsProps = {
  eventname?: string;
  country?: string;
  countryonland?: string;
  affectedcountries?: Array<{ countryname?: string; iso2?: string; iso3?: string }>;
};

type GeoCollection = {
  features?: Array<{
    geometry?: { type?: string; coordinates?: unknown };
    properties?: GdacsProps & { polygonlabel?: string; Class?: string };
  }>;
};

const briefingCache = new Map<string, { briefing: StormBriefing; fetchedAt: number }>();
const CACHE_MS = 5 * 60 * 1000;

export function stormBriefingCacheKey(eventId: number, episodeId: number): string {
  return `${eventId}:${episodeId}`;
}

function extractGdacsMeta(features: GeoCollection["features"]): {
  eventName?: string;
  gdacsCountry?: string;
  gdacsCountryOnLand?: string;
  affectedCountries: GdacsAffectedCountry[];
} {
  const props = features?.find((f) => f.properties?.eventname)?.properties;
  if (!props) return { affectedCountries: [] };
  const affectedCountries: GdacsAffectedCountry[] = (props.affectedcountries ?? [])
    .map((c) => ({
      name: c.countryname ?? "",
      iso2: c.iso2,
      iso3: c.iso3,
    }))
    .filter((c) => c.name);
  return {
    eventName: props.eventname,
    gdacsCountry: props.country || undefined,
    gdacsCountryOnLand: props.countryonland || undefined,
    affectedCountries,
  };
}

export async function fetchStormBriefing(
  eventId: number,
  episodeId: number,
): Promise<StormBriefing | null> {
  const key = stormBriefingCacheKey(eventId, episodeId);
  const cached = briefingCache.get(key);
  if (cached && Date.now() - cached.fetchedAt < CACHE_MS) return cached.briefing;

  try {
    const url = `https://www.gdacs.org/gdacsapi/api/polygons/getgeometry?eventtype=TC&eventid=${eventId}&episodeid=${episodeId}`;
    const data = await fetchJson<GeoCollection>(url, undefined, { timeoutMs: 14_000 });
    const features = data.features ?? [];
    const track = parseStormGeometry(features);
    if (!track.observed.length && !track.forecast.length) return null;

    const meta = extractGdacsMeta(features);
    const currentPos = stormPositionNow(track);
    const lastFc = track.forecast[track.forecast.length - 1];
    const forecastTarget = lastFc
      ? { lat: lastFc.lat, lon: lastFc.lon, label: lastFc.label, time: lastFc.time }
      : undefined;

    const briefing: StormBriefing = {
      track,
      ...meta,
      currentPos,
      forecastTarget,
    };
    briefingCache.set(key, { briefing, fetchedAt: Date.now() });
    return briefing;
  } catch {
    return null;
  }
}

export type { StormTrack };
