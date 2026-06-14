import { resolveShipRegion } from "../../realityData/shipRegion";
import { fetchJson } from "../fetchJson";
import type { SearchSourceResult } from "../types";

type OverpassElement = {
  type?: string;
  id?: number;
  lat?: number;
  lon?: number;
  center?: { lat?: number; lon?: number };
  tags?: Record<string, string>;
};

type OverpassResponse = { elements?: OverpassElement[] };

const OVERPASS_CACHE = new Map<string, { at: number; data: OverpassResponse }>();
const CACHE_TTL_MS = 10 * 60_000;

const EMPTY_COUNTS = { harbour: 0, buoy: 0, lighthouse: 0, pier: 0, other: 0 };

const classifySeamark = (tags: Record<string, string> | undefined): keyof typeof EMPTY_COUNTS | null => {
  if (!tags) return null;
  const t = tags["seamark:type"] || tags.seamark || "";
  if (/harbour|harbor/i.test(t) || tags.harbour === "yes" || tags.landuse === "port") return "harbour";
  if (/buoy/i.test(t)) return "buoy";
  if (/light|beacon|lighthouse/i.test(t) || tags.man_made === "lighthouse" || tags.man_made === "beacon") {
    return "lighthouse";
  }
  if (/pier|wharf|breakwater|floating_dock/i.test(t)) return "pier";
  if (tags.man_made === "pier" || tags.man_made === "breakwater") return "pier";
  return null;
};

const elementLabel = (el: OverpassElement): string =>
  el.tags?.name || el.tags?.["seamark:name"] || el.tags?.["seamark:type"] || `OSM ${el.type}/${el.id}`;

const buildOverpassQuery = (south: number, west: number, north: number, east: number): string =>
  `[out:json][timeout:28];
(
  node["seamark:type"](${south},${west},${north},${east});
  way["seamark:type"](${south},${west},${north},${east});
  node["seamark"](${south},${west},${north},${east});
  way["seamark"](${south},${west},${north},${east});
  node["man_made"="lighthouse"](${south},${west},${north},${east});
  way["man_made"="lighthouse"](${south},${west},${north},${east});
  node["harbour"](${south},${west},${north},${east});
  way["landuse"="port"](${south},${west},${north},${east});
  node["man_made"~"pier|breakwater|beacon"](${south},${west},${north},${east});
  way["man_made"~"pier|breakwater"](${south},${west},${north},${east});
);
out center 120;`;

/** OpenStreetMap Overpass — static marine infrastructure (harbours, buoys, lighthouses). */
export const fetchOverpassMarineSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "osm-overpass-marine" as const;
  const label = "תשתיות ימיות (OpenStreetMap)";

  const region = await resolveShipRegion(query);
  if (!region.bbox) {
    return {
      provider,
      label,
      ok: false,
      text: "",
      error: "ציין אזור (למשל מפרץ חיפה, רוטרדם, יוון) — Overpass דורש bbox",
      latencyMs: Math.round(performance.now() - started),
    };
  }

  const { minLat, minLon, maxLat, maxLon } = region.bbox;
  const cacheKey = `${minLat.toFixed(2)},${minLon.toFixed(2)},${maxLat.toFixed(2)},${maxLon.toFixed(2)}`;
  const cached = OVERPASS_CACHE.get(cacheKey);
  let data: OverpassResponse | undefined;
  if (cached && Date.now() - cached.at < CACHE_TTL_MS) {
    data = cached.data;
  } else {
    const q = buildOverpassQuery(minLat, minLon, maxLat, maxLon);
    const endpoints = [
      "https://overpass-api.de/api/interpreter",
      "https://overpass.kumi.systems/api/interpreter",
    ];
    let lastErr: unknown;
    for (const endpoint of endpoints) {
      try {
        data = await fetchJson<OverpassResponse>(
          endpoint,
          {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8" },
            body: `data=${encodeURIComponent(q)}`,
          },
          { timeoutMs: 28_000 },
        );
        lastErr = null;
        break;
      } catch (err) {
        lastErr = err;
      }
    }
    if (lastErr || !data) throw lastErr instanceof Error ? lastErr : new Error(String(lastErr ?? "Overpass failed"));
    OVERPASS_CACHE.set(cacheKey, { at: Date.now(), data });
  }

  const counts = { ...EMPTY_COUNTS };
  const samples: string[] = [];

  for (const el of data!.elements ?? []) {
    const kind = classifySeamark(el.tags);
    if (!kind) {
      counts.other++;
      continue;
    }
    counts[kind]++;
    if (samples.length < 10) {
      const lat = el.lat ?? el.center?.lat;
      const lon = el.lon ?? el.center?.lon;
      const coords = lat != null && lon != null ? ` · ${lat.toFixed(2)},${lon.toFixed(2)}` : "";
      samples.push(`${samples.length + 1}. ${elementLabel(el)} · ${kind}${coords}`);
    }
  }

  const total = counts.harbour + counts.buoy + counts.lighthouse + counts.pier;
  if (!total) {
    return {
      provider,
      label,
      ok: true,
      text: [
        `אזור: ${region.label} (OSM)`,
        "תשתיות ימיות בטווח: 0 (נמלים/מצופים/מגדלורים)",
        "הערה: נתונים סטטיים מ-OpenStreetMap — לא AIS חי ולא ספירת אוניות בתנועה.",
      ].join("\n"),
      url: "https://www.openstreetmap.org",
      latencyMs: Math.round(performance.now() - started),
    };
  }

  const lines = [
    `אזור: ${region.label} (OpenStreetMap / Overpass)`,
    `תשתיות ימיות בטווח: ${total} (${counts.harbour} נמלים · ${counts.buoy} מצופים · ${counts.lighthouse} מגדלורים · ${counts.pier} רציפים)`,
    "הערה: נתונים סטטיים — לא ספירת כלי שייט בתנועה (AIS).",
    ...samples,
  ];

  return {
    provider,
    label,
    ok: true,
    text: lines.join("\n"),
    url: "https://www.openstreetmap.org",
    latencyMs: Math.round(performance.now() - started),
  };
};
