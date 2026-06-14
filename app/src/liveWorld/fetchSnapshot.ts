import { fetchEarthquakeSearch } from "../webSearch/providers/usgsEarthquake";
import { fetchIssSearch } from "../realityData/providers/iss";
import { fetchShipsSearch } from "../realityData/providers/ships";
import { fetchAviationSearch } from "../realityData/providers/aviation";
import type { SearchSourceResult } from "../webSearch/types";
import {
  getCachedLiveWorldSnapshot,
  getInflightSnapshotFetch,
  mergeLiveWorldSnapshot,
  setInflightSnapshotFetch,
  setLiveWorldSnapshot,
} from "./snapshotStore";
import type { LiveEarthquakeItem, LiveIssPosition, LiveShipItem, LiveWorldSnapshot } from "./types";
import { enrichAviationItem, formatAviationSampleLine } from "./militaryAviation";

const parseEarthquakesFromText = (text: string, feedLabel: string): LiveWorldSnapshot["earthquake"] => {
  const items: LiveEarthquakeItem[] = [];
  for (const line of text.split("\n")) {
    const m = line.match(/M([\d.]+)\s*·\s*(.+?)\s*·\s*(\d{4}-\d{2}-\d{2})/);
    if (m) {
      items.push({
        magnitude: parseFloat(m[1]),
        place: m[2].trim(),
        time: Date.parse(`${m[3]}T00:00:00Z`) || Date.now(),
      });
    }
  }
  if (!items.length) return undefined;
  return { items, feedLabel };
};

const parseIssFromText = (text: string): LiveIssPosition | undefined => {
  const lat = text.match(/קו רוחב:\s*([-\d.]+)/)?.[1];
  const lon = text.match(/קו אורך:\s*([-\d.]+)/)?.[1];
  const alt = text.match(/גובה:\s*([\d.]+)/)?.[1];
  const vel = text.match(/מהירות:\s*([\d.]+)/)?.[1];
  if (!lat || !lon) return undefined;
  return {
    lat: parseFloat(lat),
    lon: parseFloat(lon),
    altitudeKm: alt ? parseFloat(alt) : 408,
    velocityKmh: vel ? parseFloat(vel) : undefined,
  };
};

const parseShipsFromText = (text: string): LiveWorldSnapshot["ships"] | undefined => {
  const region = text.match(/אזור:\s*(.+)/)?.[1]?.trim() ?? "ים תיכון";
  const countMatch = text.match(/ספינות בטווח:\s*(\d+)/);
  const count = countMatch ? parseInt(countMatch[1], 10) : 0;
  const items: LiveShipItem[] = [];
  for (const line of text.split("\n")) {
    const m = line.match(/^\d+\.\s+(.+?)\s*·\s*.+?\s*·\s*([-\d.]+),([-\d.]+)/);
    if (m) {
      items.push({
        name: m[1].trim(),
        lat: parseFloat(m[2]),
        lon: parseFloat(m[3]),
        source: "ais",
      });
    }
  }
  if (!count && !items.length) return undefined;
  return { regionLabel: region, count: count || items.length, items };
};

const buildSnapshotFromResults = (
  results: SearchSourceResult[],
  source: LiveWorldSnapshot["source"],
): LiveWorldSnapshot => {
  const snapshot: LiveWorldSnapshot = { fetchedAt: Date.now(), source };
  for (const r of results) {
    if (!r.ok || !r.text.trim()) continue;
    if (r.provider === "usgs-earthquake") {
      snapshot.earthquake = parseEarthquakesFromText(r.text, r.label) ?? snapshot.earthquake;
    }
    if (r.provider === "iss-tracker") {
      snapshot.iss = parseIssFromText(r.text) ?? snapshot.iss;
    }
    if (r.provider === "ais-ships") {
      snapshot.ships = parseShipsFromText(r.text) ?? snapshot.ships;
    }
    if (r.provider === "adsb-aviation") {
      const lines = r.text.split("\n").filter(Boolean);
      const countLine = lines.find((l) => /מטוסים/.test(l));
      const count = parseInt(countLine?.match(/(\d+)/)?.[1] ?? "0", 10);
      snapshot.aviation = {
        count,
        regionLabel: lines.find((l) => l.startsWith("אזור:"))?.replace("אזור:", "").trim() ?? "",
        sample: lines.filter((l) => /^\d+\./.test(l)).slice(0, 5),
      };
    }
  }
  return snapshot;
};

/** Parallel fetch of live-world layers for cache + fallback. */
export async function fetchLiveWorldSnapshot(force = false): Promise<LiveWorldSnapshot | null> {
  if (!force) {
    const cached = getCachedLiveWorldSnapshot();
    if (cached) return cached;
  }

  const queries = {
    eq: "אילו רעידות אדמה התרחשו ב-24 השעות האחרונות?",
    iss: "איפה תחנת החלל הבינלאומית עכשיו?",
    ships: "כמה ספינות יש בים התיכון?",
    av: "כמה מטוסים נמצאים כרגע מעל ישראל?",
  };

  const [eq, iss, ships, av] = await Promise.all([
    fetchEarthquakeSearch(queries.eq),
    fetchIssSearch(queries.iss),
    fetchShipsSearch(queries.ships),
    fetchAviationSearch(queries.av, []),
  ]);

  const snapshot = buildSnapshotFromResults([eq, iss, ships, av], "fetch");
  setLiveWorldSnapshot(snapshot);
  return snapshot;
}

export async function warmLiveWorldCache(): Promise<void> {
  if (getInflightSnapshotFetch()) return;
  const p = fetchLiveWorldSnapshot(true).finally(() => setInflightSnapshotFetch(null));
  setInflightSnapshotFetch(p);
  await p.catch(() => null);
}

/** Ingest minimal payload from Reality Live iframe postMessage. */
export function ingestGlobeLivePayload(payload: unknown): LiveWorldSnapshot | null {
  if (!payload || typeof payload !== "object") return null;
  const p = payload as Record<string, unknown>;
  const snapshot: LiveWorldSnapshot = { fetchedAt: Date.now(), source: "globe" };

  const eqItems = (p.earthquake as { items?: unknown[] } | undefined)?.items;
  if (Array.isArray(eqItems) && eqItems.length) {
    snapshot.earthquake = {
      feedLabel: "USGS (עולם חי)",
      items: eqItems.slice(0, 12).map((raw) => {
        const e = raw as {
          magnitude?: number;
          mag?: number;
          place?: string;
          time?: number;
          geo?: { lat?: number; lon?: number };
        };
        return {
          magnitude: e.magnitude ?? e.mag ?? null,
          place: String(e.place ?? "unknown"),
          time: e.time ?? Date.now(),
          lat: e.geo?.lat,
          lon: e.geo?.lon,
        };
      }),
    };
  }

  const issGeo = (p.iss as { geo?: { lat?: number; lon?: number }; altitude?: number } | undefined)?.geo;
  if (issGeo?.lat != null && issGeo?.lon != null) {
    snapshot.iss = {
      lat: issGeo.lat,
      lon: issGeo.lon,
      altitudeKm: (p.iss as { altitude?: number }).altitude ?? 408,
    };
  }

  const shipItems = (p.ships as { items?: unknown[] } | undefined)?.items;
  if (Array.isArray(shipItems)) {
    const mapped: LiveShipItem[] = shipItems
      .filter((s) => {
        const g = (s as { geo?: { lat?: number; lon?: number } }).geo;
        return g?.lat != null && g?.lon != null;
      })
      .slice(0, 200)
      .map((s) => {
        const row = s as {
          name?: string;
          geo: { lat: number; lon: number };
          speed?: number;
          destination?: string;
          source?: string;
        };
        return {
          name: row.name || "AIS",
          lat: row.geo.lat,
          lon: row.geo.lon,
          speedKn: row.speed ?? undefined,
          destination: row.destination,
          source: row.source ?? "globe",
        };
      });
    snapshot.ships = {
      regionLabel: "עולם חי (AIS)",
      count: mapped.length,
      items: mapped,
    };
  }

  const avItems = (p.aviation as { items?: unknown[] } | undefined)?.items;
  if (Array.isArray(avItems)) {
    const mapped = avItems
      .slice(0, 900)
      .map((raw) => {
        const a = raw as {
          icao24?: string;
          callsign?: string;
          country?: string;
          category?: string | number;
          isMilitary?: boolean;
          milLabel?: string;
          geo?: { lat?: number; lon?: number; alt?: number };
          altitude?: number;
        };
        return enrichAviationItem(a);
      });
    const militaryCount = mapped.filter((i) => i.isMilitary).length;
    const awacsCount = mapped.filter((i) => i.awacsSuspect).length;
    snapshot.aviation = {
      count: mapped.length,
      militaryCount,
      awacsCount,
      tankerCount: mapped.filter((i) => i.tankerSuspect).length,
      regionLabel: "עולם חי (ADS-B)",
      sample: mapped.slice(0, 5).map(formatAviationSampleLine),
      items: mapped,
    };
  }

  if (!snapshot.earthquake && !snapshot.iss && !snapshot.ships && !snapshot.aviation) {
    return null;
  }
  return mergeLiveWorldSnapshot(snapshot);
}
