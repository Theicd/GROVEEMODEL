import { isAisStreamConfigured } from "../apiKeys/apiKeyStore";
import { fetchAisStreamGlobeShips, fetchAisStreamShips } from "../realityData/providers/aisStream";
import { aggregateShipHits, type ShipHit } from "../realityData/shipAggregate";
import { MED_BBOX } from "../realityData/shipRegion";
import { fetchOverpassMarineSearch } from "../webSearch/providers/overpassMarine";
import type { LiveShipItem, LiveWorldSnapshot } from "./types";

export const SHIP_SNAPSHOT_ITEM_CAP = 500;
export const SHIP_SERP_SAMPLE_CAP = 64;

const shipHitToLiveItem = (h: ShipHit): LiveShipItem => ({
  name: h.name,
  lat: h.lat,
  lon: h.lon,
  speedKn: h.speed,
  destination: h.destination,
  source: h.source,
});

const shipItemKey = (s: LiveShipItem): string =>
  `${s.name.toLowerCase()}|${s.lat.toFixed(3)}|${s.lon.toFixed(3)}`;

const sourceRank = (source?: string): number => {
  if (source === "aisstream") return 0;
  if (source === "globe") return 1;
  if (source === "ais" || source === "digitraffic") return 2;
  return 9;
};

/** Merge ship lists — AISStream wins on duplicate positions; prefer live over demo. */
export const mergeLiveShipItems = (
  ...groups: LiveShipItem[][]
): LiveShipItem[] => {
  const byKey = new Map<string, LiveShipItem>();
  for (const group of groups) {
    for (const item of group) {
      if (!Number.isFinite(item.lat) || !Number.isFinite(item.lon)) continue;
      const key = shipItemKey(item);
      const prev = byKey.get(key);
      if (!prev || sourceRank(item.source) < sourceRank(prev.source)) {
        byKey.set(key, item);
      }
    }
  }
  return [...byKey.values()].sort((a, b) => {
    const sr = sourceRank(a.source) - sourceRank(b.source);
    if (sr !== 0) return sr;
    return a.name.localeCompare(b.name);
  });
};

/** Multi-source ship layer for SERP + snapshot cache (AISStream globe + Med + Digitraffic filtered). */
export const fetchLiveShipLayerForCache = async (): Promise<LiveWorldSnapshot["ships"]> => {
  const medRegion = {
    label: "ים תיכון + AISStream גלובלי",
    bbox: MED_BBOX,
    center: { lat: 34.0, lon: 28.0 },
    radiusNm: 500,
  };

  const groups: LiveShipItem[][] = [];

  if (isAisStreamConfigured()) {
    try {
      const globeHits = await fetchAisStreamGlobeShips({ timeoutMs: 20_000 });
      groups.push(globeHits.map(shipHitToLiveItem));
    } catch {
      /* optional */
    }
    try {
      const medHits = await fetchAisStreamShips(MED_BBOX, { timeoutMs: 10_000 });
      groups.push(medHits.map(shipHitToLiveItem));
    } catch {
      /* optional */
    }
  }

  try {
    const agg = await aggregateShipHits("כמה ספינות יש בים התיכון?", medRegion);
    groups.push(agg.liveHits.map(shipHitToLiveItem));
  } catch {
    /* optional */
  }

  const merged = mergeLiveShipItems(...groups).slice(0, SHIP_SNAPSHOT_ITEM_CAP);
  const aisStreamCount = merged.filter((s) => s.source === "aisstream").length;

  return {
    regionLabel:
      aisStreamCount > 0
        ? `AISStream (${aisStreamCount}) + Digitraffic · ים תיכון/גלובלי`
        : "Digitraffic · ים תיכון (אין AISStream — הוסף מפתח 🔑)",
    count: merged.length,
    items: merged,
  };
};

export const fetchLiveMarineInfraForCache = async (): Promise<LiveWorldSnapshot["marineInfra"]> => {
  try {
    const r = await fetchOverpassMarineSearch("מצופים ומגדלורים בים התיכון");
    if (!r.ok || !r.text.trim()) return undefined;
    const regionMatch = r.text.match(/^אזור:\s*(.+)$/m);
    const items: NonNullable<LiveWorldSnapshot["marineInfra"]>["items"] = [];
    for (const line of r.text.split("\n")) {
      const m = line.match(/^\d+\.\s+(.+?)\s·\s*(harbour|buoy|lighthouse|pier|other)(?:\s·\s*([-\d.]+),([-\d.]+))?$/i);
      if (!m) continue;
      const lat = m[3] != null ? parseFloat(m[3]) : undefined;
      const lon = m[4] != null ? parseFloat(m[4]) : undefined;
      items.push({
        name: m[1].trim(),
        kind: m[2].toLowerCase(),
        lat: Number.isFinite(lat) ? lat : undefined,
        lon: Number.isFinite(lon) ? lon : undefined,
      });
      if (items.length >= 24) break;
    }
    if (!items.length) return undefined;
    return {
      regionLabel: regionMatch?.[1]?.trim() ?? "ים תיכון (OSM)",
      items,
    };
  } catch {
    return undefined;
  }
};

export const shipHitsFromLayer = (layer: LiveWorldSnapshot["ships"]): ShipHit[] =>
  (layer?.items ?? []).map((s) => ({
    name: s.name,
    lat: s.lat,
    lon: s.lon,
    speed: s.speedKn,
    destination: s.destination,
    source:
      s.source === "aisstream"
        ? "aisstream"
        : s.source === "route-marker" || s.source === "med-fallback"
          ? "route-marker"
          : s.source === "globe" || s.source === "digitraffic"
            ? "globe"
            : "ais",
  }));

export const countShipSources = (items: LiveShipItem[]): { aisstream: number; digitraffic: number; other: number } => {
  let aisstream = 0;
  let digitraffic = 0;
  let other = 0;
  for (const s of items) {
    if (s.source === "aisstream") aisstream++;
    else if (s.source === "ais" || s.source === "digitraffic") digitraffic++;
    else other++;
  }
  return { aisstream, digitraffic, other };
};

/** Prefer AISStream when merging globe postMessage ingest with fetch cache. */
export const mergeShipLayers = (
  existing: LiveWorldSnapshot["ships"] | undefined,
  incoming: LiveWorldSnapshot["ships"] | undefined,
): LiveWorldSnapshot["ships"] | undefined => {
  if (!existing?.items?.length && !incoming?.items?.length) return undefined;
  const merged = mergeLiveShipItems(existing?.items ?? [], incoming?.items ?? []).slice(
    0,
    SHIP_SNAPSHOT_ITEM_CAP,
  );
  const counts = countShipSources(merged);
  return {
    regionLabel:
      counts.aisstream > 0
        ? `AISStream (${counts.aisstream}) + Digitraffic · עולם חי`
        : existing?.regionLabel ?? incoming?.regionLabel ?? "עולם חי (AIS)",
    count: merged.length,
    items: merged,
  };
};
