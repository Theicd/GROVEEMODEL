import { getCachedLiveWorldSnapshot } from "../liveWorld/snapshotStore";
import type { LiveShipItem } from "../liveWorld/types";
import { fetchJson } from "../webSearch/fetchJson";
import {
  MED_PORTS,
  WORLD_PORTS,
  inShipBbox,
  portsInBbox,
  portsToRouteMarkers,
  type RouteMarkerHit,
  type ShipBbox,
} from "./medPorts";

type AisFeature = {
  properties?: {
    mmsi?: number;
    sog?: number;
    cog?: number;
    navStat?: number;
    timestampExternal?: string;
  };
  geometry?: { coordinates?: [number, number] };
};

type AisGeo = { features?: AisFeature[] };

type VesselMeta = {
  name?: string;
  destination?: string;
  shipType?: number;
};

export type ShipHit = {
  name: string;
  lat: number;
  lon: number;
  speed?: number;
  destination?: string;
  source: "ais" | "route-marker" | "globe";
  timestamp?: string;
};

const LIVE_GLOBE_SOURCES = new Set(["ais", "digitraffic", "globe"]);

export const isLiveShipSource = (source?: string): boolean =>
  !source || LIVE_GLOBE_SOURCES.has(source);

export const isCountShipsQuery = (query: string): boolean => /כמה|how\s+many/i.test(query);

const hitKey = (s: ShipHit) => `${s.name}|${s.lat.toFixed(3)}|${s.lon.toFixed(3)}`;

export const dedupeHits = (hits: ShipHit[]): ShipHit[] => {
  const seen = new Set<string>();
  const out: ShipHit[] = [];
  for (const h of hits) {
    const k = hitKey(h);
    if (seen.has(k)) continue;
    seen.add(k);
    out.push(h);
  }
  return out;
};

let vesselMetaCache: Map<number, VesselMeta> | null = null;
let vesselMetaFetchedAt = 0;

export const loadVesselMeta = async (): Promise<Map<number, VesselMeta>> => {
  if (vesselMetaCache && Date.now() - vesselMetaFetchedAt < 3_600_000) return vesselMetaCache;
  try {
    const arr = await fetchJson<VesselMeta[]>(
      "https://meri.digitraffic.fi/api/ais/v1/vessels",
      undefined,
      { timeoutMs: 12_000 },
    );
    const map = new Map<number, VesselMeta>();
    for (const v of arr ?? []) {
      const mmsi = (v as { mmsi?: number }).mmsi;
      if (mmsi) map.set(mmsi, v);
    }
    vesselMetaCache = map;
    vesselMetaFetchedAt = Date.now();
    return map;
  } catch {
    return vesselMetaCache ?? new Map();
  }
};

export const hitsFromGlobeCache = (bbox: ShipBbox | null, liveOnly = false): ShipHit[] => {
  const snap = getCachedLiveWorldSnapshot(120_000);
  if (!snap?.ships?.items?.length) return [];
  return snap.ships.items
    .filter((s) => {
      if (liveOnly && !isLiveShipSource(s.source)) return false;
      return !bbox || inShipBbox(s.lat, s.lon, bbox);
    })
    .map((s: LiveShipItem) => ({
      name: s.name || "AIS",
      lat: s.lat,
      lon: s.lon,
      speed: s.speedKn,
      destination: s.destination,
      source: isLiveShipSource(s.source) ? ("globe" as const) : ("route-marker" as const),
      timestamp: snap.fetchedAt ? new Date(snap.fetchedAt).toISOString() : undefined,
    }));
};

export const parseAisFeatures = (
  features: AisFeature[],
  bbox: ShipBbox | null,
  meta: Map<number, VesselMeta>,
): ShipHit[] => {
  const out: ShipHit[] = [];
  for (const f of features) {
    const c = f.geometry?.coordinates;
    const p = f.properties;
    if (!c?.length || c[1] == null || c[0] == null) continue;
    const lat = c[1];
    const lon = c[0];
    if (bbox && !inShipBbox(lat, lon, bbox)) continue;
    const mmsi = p?.mmsi;
    const metaRow = mmsi != null ? meta.get(mmsi) : undefined;
    out.push({
      name: metaRow?.name?.trim() || (mmsi != null ? `MMSI ${mmsi}` : "AIS"),
      lat,
      lon,
      speed: p?.sog != null ? p.sog / 10 : undefined,
      destination: metaRow?.destination ?? "",
      source: "ais",
      timestamp: p?.timestampExternal ?? new Date().toISOString(),
    });
  }
  return out;
};

export const fetchDigitrafficAis = async (
  center: { lat: number; lon: number } | null,
  radiusNm: number,
  bbox: ShipBbox | null,
  meta: Map<number, VesselMeta>,
): Promise<ShipHit[]> => {
  const url = center
    ? `https://meri.digitraffic.fi/api/ais/v1/locations?latitude=${center.lat}&longitude=${center.lon}&radius=${radiusNm}`
    : "https://meri.digitraffic.fi/api/ais/v1/locations";
  const geo = await fetchJson<AisGeo>(url, undefined, { timeoutMs: 16_000 });
  return parseAisFeatures(geo.features ?? [], bbox, meta);
};

export const routeMarkersForRegion = (bbox: ShipBbox | null): RouteMarkerHit[] => {
  const seeds = [...MED_PORTS, ...WORLD_PORTS];
  return portsToRouteMarkers(bbox ? portsInBbox(seeds, bbox) : seeds);
};

export type ShipAggregateResult = {
  liveHits: ShipHit[];
  demoHits: ShipHit[];
  allHits: ShipHit[];
  aisCount: number;
  globeCount: number;
  routeCount: number;
  fetchedAt: string;
};

export const aggregateShipHits = async (
  query: string,
  region: { label: string; bbox: ShipBbox | null; center: { lat: number; lon: number } | null; radiusNm: number },
): Promise<ShipAggregateResult> => {
  const fetchedAt = new Date().toISOString();
  const globeLive = hitsFromGlobeCache(region.bbox, true);
  const globeDemo = hitsFromGlobeCache(region.bbox, false).filter((h) => h.source === "route-marker");

  let aisHits: ShipHit[] = [];
  try {
    const meta = await loadVesselMeta();
    aisHits = await fetchDigitrafficAis(region.center, region.radiusNm, region.bbox, meta);
    if (!aisHits.length && region.bbox && /סואץ|suez|פרס|persian|ים\s+תיכון|mediterranean/i.test(region.label + query)) {
      aisHits = await fetchDigitrafficAis(null, region.radiusNm, region.bbox, meta);
    }
  } catch {
    aisHits = [];
  }

  const routeHits: ShipHit[] = routeMarkersForRegion(region.bbox).map((r) => ({
    ...r,
    source: "route-marker" as const,
  }));

  const liveHits = dedupeHits([...globeLive, ...aisHits]);
  const demoHits = dedupeHits([
    ...globeDemo.filter((h) => !liveHits.some((l) => hitKey(l) === hitKey(h))),
    ...routeHits.filter((h) => !liveHits.some((l) => hitKey(l) === hitKey(h))),
  ]);

  const countQuery = isCountShipsQuery(query);
  const allHits = countQuery ? [...liveHits, ...demoHits] : dedupeHits([...liveHits, ...demoHits]);

  return {
    liveHits,
    demoHits,
    allHits,
    aisCount: liveHits.filter((h) => h.source === "ais").length,
    globeCount: liveHits.filter((h) => h.source === "globe").length,
    routeCount: demoHits.length,
    fetchedAt,
  };
};

export const formatShipsText = (
  regionLabel: string,
  agg: ShipAggregateResult,
  query: string,
): string => {
  const liveCount = agg.liveHits.length;
  const countQuery = isCountShipsQuery(query);
  const answerCount = countQuery ? liveCount : agg.allHits.length;

  const lines = [
    `אזור: ${regionLabel}`,
    `ANSWER (ships live): ${answerCount}`,
    `דיווח AIS חי + עולם חי: ${liveCount} (${agg.aisCount} AIS · ${agg.globeCount} עולם חי)`,
  ];

  if (agg.demoHits.length) {
    lines.push(`סימוני מסלול (הדגמה — לא AIS חי): ${agg.demoHits.length}`);
  }

  lines.push(`עודכן: ${agg.fetchedAt.replace("T", " ").slice(0, 19)} UTC`);

  if (countQuery && liveCount === 0 && agg.demoHits.length) {
    lines.push(
      "הערה: בתעלת סואץ ובאזורים מחוץ לצפון אירופה אין כיסוי AIS חי מ-Digitraffic — הספירה 0. סימוני המסלול הם הדגמה מ«עולם חי», לא אוניות בזמן אמת. פתח REALITY LIVE לשכבת ספינות.",
    );
  } else {
    lines.push("הערה: Digitraffic מכסה בעיקר צפון אירופה; עולם חי משלב AIS גלובלי + סימוני מסלול.");
  }

  const samplePool = countQuery && liveCount > 0 ? agg.liveHits : agg.allHits;
  lines.push(
    ...samplePool.slice(0, 12).map((s, i) => {
      const spd = s.speed != null ? `${s.speed.toFixed(1)} kn` : "—";
      const tag =
        s.source === "route-marker" ? "מסלול (הדגמה)" : s.source === "globe" ? "עולם חי" : "AIS";
      return `${i + 1}. ${s.name} · ${tag} · ${s.lat.toFixed(2)},${s.lon.toFixed(2)} · ${spd}${s.destination ? ` → ${s.destination}` : ""}`;
    }),
  );

  return lines.join("\n");
};
