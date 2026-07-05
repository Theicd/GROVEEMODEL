import type { LiveDisasterItem, LiveEarthquakeItem, LiveWorldSnapshot } from "../liveWorld/types";
import { getLiveWorldSnapshotForPanel } from "../liveWorld/snapshotStore";
import { resolveUsgsMinMagnitude } from "../webSearch/providers/usgsEarthquake";
import { isDisasterQuery, isEarthquakeQuery } from "../webSearch/intents";
import type { SearchResultsFacets, SearchResultsPayload, UnifiedSearchHit } from "./types";
import { faviconForUrl } from "./sourceBranding";
import { coerceHttpUrl, coerceText } from "./coerceHitUrl";

const USGS_HOME = "https://earthquake.usgs.gov";
const GDACS_HOME = "https://www.gdacs.org";

const slug = (s: string): string =>
  s
    .toLowerCase()
    .replace(/[^\w\u0590-\u05ff]+/g, "-")
    .slice(0, 80);

export const earthquakeItemToHit = (
  item: LiveEarthquakeItem,
  index: number,
  feedLabel = "USGS",
): UnifiedSearchHit => {
  const mag = item.magnitude != null ? item.magnitude.toFixed(1) : "?";
  const when = new Date(item.time).toISOString().replace("T", " ").slice(0, 19);
  const tsunamiNote = item.tsunami ? " · אזהרת צונאמי" : "";
  const place = coerceText(item.place, "unknown");
  return {
    id: `eq-${item.time}-${slug(place)}-${index}`,
    kind: "earthquake",
    title: `M${mag} · ${place}`,
    titleOriginal: `M${mag} · ${place}`,
    url: coerceHttpUrl(item.url, USGS_HOME),
    snippet: `${when} UTC${tsunamiNote}`,
    snippetOriginal: `${when} UTC${tsunamiNote}`,
    sourceLabel: feedLabel,
    provider: "usgs-earthquake",
    faviconUrl: faviconForUrl(USGS_HOME),
    publishedTs: item.time,
    score: 40 + (item.magnitude ?? 0) * 12,
    meta: {
      magnitude: item.magnitude ?? undefined,
      lat: item.lat,
      lon: item.lon,
      depth: item.depth,
      engine: item.tsunami ? "Earthquake · Tsunami alert" : "Earthquake",
    },
    summarizable: false,
  };
};

export const disasterItemToHit = (
  item: LiveDisasterItem,
  index: number,
  feedLabel = "GDACS",
): UnifiedSearchHit => {
  const alert = coerceText(item.alertLevel);
  const type = coerceText(item.eventType) || undefined;
  const eventName = coerceText(item.eventName, "—");
  const country = coerceText(item.country);
  return {
    id:
      item.eventId != null && item.episodeId != null
        ? `gdacs-${item.eventId}-${item.episodeId}`
        : `gdacs-${slug(eventName)}-${index}`,
    kind: "disaster",
    title: eventName,
    titleOriginal: eventName,
    url: coerceHttpUrl(item.url, GDACS_HOME),
    snippet: country,
    snippetOriginal: country,
    sourceLabel: "GDACS",
    provider: "gdacs-disasters",
    faviconUrl: faviconForUrl(GDACS_HOME),
    publishedTs: item.dateModified ?? item.startTime ?? Date.now(),
    score: /red/i.test(alert) ? 92 : /orange/i.test(alert) ? 78 : 55,
    meta: {
      alertLevel: alert,
      disasterType: type,
      lat: item.lat,
      lon: item.lon,
      engine: type || "Disaster",
      gdacsEventId: item.eventId,
      gdacsEpisodeId: item.episodeId,
      severityText: item.severityText,
      gdacsIsCurrent: item.isCurrent,
      gdacsEndTime: item.endTime,
      startTime: item.startTime,
      dateModified: item.dateModified,
    },
    summarizable: false,
  };
};

/** Parse USGS provider text into earthquake hits. */
export const parseUsgsEarthquakeText = (text: string): UnifiedSearchHit[] => {
  const items: LiveEarthquakeItem[] = [];
  const lines = text.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    const m = line.match(/^-?\s*M([\d.]+)\s*·\s*(.+?)\s*·\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})/);
    if (!m) continue;
    const urlLine = lines[i + 1]?.trim();
    const url = urlLine?.startsWith("http") ? urlLine : undefined;
    items.push({
      magnitude: parseFloat(m[1]),
      place: m[2].trim(),
      time: Date.parse(`${m[3].replace(" ", "T")}Z`) || Date.now(),
      url,
      tsunami: /צונאמי|tsunami/i.test(line),
    });
  }
  return items.map((item, idx) => earthquakeItemToHit(item, idx));
};

/** Parse GDACS provider text into disaster hits. */
export const parseGdacsDisasterText = (text: string): UnifiedSearchHit[] => {
  const items: LiveDisasterItem[] = [];
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    const m = trimmed.match(
      /^\d+\.\s+(.+?)\s*·\s*(.+?)\s*·\s*(Green|Orange|Red|ירוק|כתום|אדום)(?:\s*·\s*([A-Z]{2}))?/i,
    );
    if (!m) continue;
    items.push({
      eventName: m[1].trim(),
      country: m[2].trim(),
      alertLevel: m[3].trim(),
      eventType: m[4]?.trim(),
    });
  }
  return items.map((item, idx) => disasterItemToHit(item, idx));
};

export const buildLiveDisastersHitsFromSnapshot = (
  snapshot: LiveWorldSnapshot | null,
): UnifiedSearchHit[] => {
  if (!snapshot) return [];
  const out: UnifiedSearchHit[] = [];
  const eqLabel = snapshot.earthquake?.feedLabel ?? "USGS";
  for (const [i, item] of (snapshot.earthquake?.items ?? []).entries()) {
    try {
      out.push(earthquakeItemToHit(item, i, eqLabel));
    } catch {
      /* skip malformed cache row */
    }
  }
  const gdacsLabel = snapshot.disasters?.feedLabel ?? "GDACS";
  for (const [i, item] of (snapshot.disasters?.items ?? []).entries()) {
    try {
      out.push(disasterItemToHit(item, i, gdacsLabel));
    } catch {
      /* skip malformed cache row */
    }
  }
  return out.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
};

export const getLiveDisastersHits = (): UnifiedSearchHit[] =>
  buildLiveDisastersHitsFromSnapshot(getLiveWorldSnapshotForPanel());

const isLiveDisasterHit = (h: UnifiedSearchHit): boolean =>
  h.kind === "earthquake" || h.kind === "disaster";

export const filterLiveDisasterHitsForQuery = (
  hits: UnifiedSearchHit[],
  query: string,
): UnifiedSearchHit[] => {
  const q = query.trim();
  if (!q) return hits;
  let filtered = hits;
  if (isEarthquakeQuery(q)) {
    filtered = filtered.filter((h) => h.kind === "earthquake" || h.kind === "disaster");
    const minMag = resolveUsgsMinMagnitude(q);
    if (minMag != null) {
      filtered = filtered.filter(
        (h) => h.kind === "disaster" || (h.meta?.magnitude ?? 0) >= minMag,
      );
    }
  } else if (isDisasterQuery(q)) {
    filtered = filtered.filter((h) => h.kind === "disaster" || h.kind === "earthquake");
  }
  return filtered;
};

export const mergeLiveDisasterHits = (
  hits: UnifiedSearchHit[],
  query = "",
): UnifiedSearchHit[] => {
  const live = filterLiveDisasterHitsForQuery(getLiveDisastersHits(), query);
  if (!live.length) return hits;
  const existing = new Set(
    hits.filter(isLiveDisasterHit).map((h) => `${h.kind}:${h.title}:${h.publishedTs ?? 0}`),
  );
  const merged = [...hits];
  for (const h of live) {
    const key = `${h.kind}:${h.title}:${h.publishedTs ?? 0}`;
    if (existing.has(key)) continue;
    merged.push(h);
    existing.add(key);
  }
  return merged.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
};

export const liveDisastersFacetCounts = (
  hits: UnifiedSearchHit[],
): { earthquakes: number; disasters: number } => ({
  earthquakes: hits.filter((h) => h.kind === "earthquake").length,
  disasters: hits.filter((h) => h.kind === "disaster").length,
});

export const liveDisastersTotal = (hits: UnifiedSearchHit[]): number =>
  liveDisastersFacetCounts(hits).earthquakes + liveDisastersFacetCounts(hits).disasters;

export const formatLiveDisastersNote = (
  snapshot: LiveWorldSnapshot | null,
  uiLang: "he" | "en",
): string | undefined => {
  if (!snapshot) return undefined;
  const eq = snapshot.earthquake?.items?.length ?? 0;
  const gd = snapshot.disasters?.items?.length ?? 0;
  if (!eq && !gd) return undefined;
  const when = new Date(snapshot.fetchedAt).toLocaleTimeString(uiLang === "he" ? "he-IL" : "en-GB", {
    hour: "2-digit",
    minute: "2-digit",
  });
  if (uiLang === "he") {
    return `מידע חי: ${eq} רעידות (USGS) · ${gd} אסונות (GDACS) · עודכן ${when}`;
  }
  return `Live: ${eq} earthquakes (USGS) · ${gd} disasters (GDACS) · updated ${when}`;
};

const emptyFacets = (): SearchResultsFacets => ({
  rss: 0,
  web: 0,
  companionWeb: 0,
  repos: 0,
  papers: 0,
  movies: 0,
  images: 0,
  videos: 0,
  youtube: 0,
  liveTv: 0,
  radio: 0,
  products: 0,
  hfModels: 0,
  earthquakes: 0,
  disasters: 0,
  ships: 0,
  weather: 0,
  marine: 0,
  places: 0,
  other: 0,
});

const buildFacetsFromHits = (hits: UnifiedSearchHit[]): SearchResultsFacets => ({
  ...emptyFacets(),
  rss: hits.filter((h) => h.kind === "rss").length,
  web: hits.filter((h) => h.kind === "web" && h.provider !== "openserp").length,
  companionWeb: hits.filter((h) => h.kind === "web" && h.provider === "openserp").length,
  repos: hits.filter((h) => h.kind === "github").length,
  papers: hits.filter((h) => h.kind === "arxiv").length,
  movies: hits.filter((h) => h.kind === "movie").length,
  images: hits.filter((h) => h.kind === "image").length,
  videos: hits.filter((h) => h.kind === "video").length,
  youtube: hits.filter((h) => h.kind === "youtube").length,
  liveTv: hits.filter((h) => h.kind === "livetv").length,
  radio: hits.filter((h) => h.kind === "radio").length,
  products: hits.filter((h) => h.kind === "product").length,
  hfModels: hits.filter((h) => h.kind === "hfmodel").length,
  earthquakes: hits.filter((h) => h.kind === "earthquake").length,
  disasters: hits.filter((h) => h.kind === "disaster").length,
  ships: hits.filter((h) => h.kind === "ship" || h.kind === "marine").length,
  weather: hits.filter((h) => h.kind === "weather").length,
  marine: hits.filter((h) => h.kind === "marine").length,
  places: hits.filter((h) => h.kind === "place" || h.kind === "route").length,
  other: hits.filter((h) => h.kind === "hackernews" || h.kind === "structured").length,
});

export const buildLiveDisastersPayload = (query = ""): SearchResultsPayload => {
  const snapshot = getLiveWorldSnapshotForPanel();
  const hits = filterLiveDisasterHitsForQuery(getLiveDisastersHits(), query);
  const { earthquakes, disasters } = liveDisastersFacetCounts(hits);
  const eventsCount = earthquakes + disasters;
  const q = query.trim();
  const eqQuery = isEarthquakeQuery(q);
  const disQuery = isDisasterQuery(q) && !eqQuery;
  const emptyOpen = !q;
  const wantsEvents =
    eventsCount > 0 &&
    !emptyOpen &&
    (eqQuery || disQuery || (earthquakes > 0 && disasters > 0));
  return {
    query,
    generatedAt: Date.now(),
    hits,
    facets: buildFacetsFromHits(hits),
    providerErrors: [],
    preferEventsFilter: wantsEvents,
    liveDisastersNote: formatLiveDisastersNote(snapshot, "he"),
  };
};

export const refreshLiveDisastersInPayload = (payload: SearchResultsPayload): SearchResultsPayload => {
  const nonLive = payload.hits.filter((h) => !isLiveDisasterHit(h));
  const live = filterLiveDisasterHitsForQuery(getLiveDisastersHits(), payload.query);
  const hits = [...nonLive, ...live].sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
  const snapshot = getLiveWorldSnapshotForPanel();
  const { earthquakes, disasters } = liveDisastersFacetCounts(hits);
  const eventsCount = earthquakes + disasters;
  const q = payload.query.trim();
  const eqQuery = isEarthquakeQuery(q);
  const disQuery = isDisasterQuery(q) && !eqQuery;
  const emptyOpen = !q;
  const preferEvents =
    !emptyOpen &&
    (payload.preferEventsFilter ||
      (eventsCount > 0 && (eqQuery || disQuery)));
  return {
    ...payload,
    generatedAt: Date.now(),
    hits,
    facets: {
      ...payload.facets,
      earthquakes: hits.filter((h) => h.kind === "earthquake").length,
      disasters: hits.filter((h) => h.kind === "disaster").length,
    },
    liveDisastersNote: formatLiveDisastersNote(snapshot, "he"),
    preferEventsFilter: preferEvents,
  };
};
