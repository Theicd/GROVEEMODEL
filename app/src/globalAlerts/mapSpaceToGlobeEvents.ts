import type { CadRecord } from "./fetchJplCad";
import type { ScoutRecord } from "./fetchJplScout";
import { fetchNeoHorizonsTrack } from "./fetchNeoHorizonsTrack";
import { filterNeoAlerts } from "./alertFilters";
import { formatNeoEta, neoSeverityLine } from "./neoEta";
import { mergeShowcaseWithReal } from "./neoShowcaseCatalog";
import type { GlobeAlertEvent } from "./types";

const HORIZONS_BATCH = 3;
const HORIZONS_ENRICH_LIMIT = 10;
const TRACK_WINDOW_DAYS = 7;

/**
 * Deterministic sky-anchor per object so NEOs without a Horizons track
 * still spread across the celestial sphere instead of stacking on lat/lon 0,0.
 */
export function pseudoAnchorFor(id: string): { lat: number; lon: number } {
  let h = 2166136261;
  for (let i = 0; i < id.length; i++) {
    h ^= id.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  const u = (h >>> 0) / 4294967296;
  const v = ((Math.imul(h, 2654435761) >>> 0) / 4294967296);
  const lat = (u - 0.5) * 150; // -75..75
  const lon = v * 360 - 180; // -180..180
  return { lat, lon };
}

export function cadToGlobeEvent(
  cad: CadRecord,
  lat: number,
  lon: number,
  opts?: { trackPending?: boolean },
): GlobeAlertEvent {
  const name = cad.fullname?.trim() || cad.des;
  const isPha = cad.distLd < 20;
  return {
    id: `neo-cad-${cad.des}-${cad.approachTime}`,
    type: "neo",
    lat,
    lon,
    location: name,
    time: cad.approachTime,
    source: "nasa-jpl",
    designation: cad.des,
    distAu: cad.distAu,
    distLd: cad.distLd,
    distMinLd: cad.distMinAu / (384_400 / 149_597_870.7),
    distMaxLd: cad.distMaxAu / (384_400 / 149_597_870.7),
    vRel: cad.vRel,
    vInf: cad.vInf,
    approachTime: cad.approachTime,
    approachLabel: cad.approachLabel,
    hMagnitude: cad.h,
    diameterKm: cad.diameterKm,
    isPha,
    trackPending: opts?.trackPending,
    severityText: neoSeverityLine(cad),
  };
}

export function scoutToGlobeEvent(
  scout: ScoutRecord,
  lat: number,
  lon: number,
  opts?: { trackPending?: boolean },
): GlobeAlertEvent {
  return {
    id: `neo-scout-${scout.objectName}`,
    type: "neo",
    lat,
    lon,
    location: `Scout · ${scout.objectName}`,
    time: Date.now(),
    source: "nasa-jpl",
    designation: scout.objectName,
    distLd: scout.caDistLd,
    distAu: scout.caDistLd * (384_400 / 149_597_870.7),
    vRel: scout.vInf,
    vInf: scout.vInf,
    approachTime: Date.now(),
    isPha: scout.phaScore > 0,
    trackPending: opts?.trackPending,
    severityText: `חדש · ${scout.caDistLd.toFixed(2)} LD · v∞ ${scout.vInf.toFixed(1)} km/s`,
    scoutNeoScore: scout.neoScore,
  };
}

async function enrichCadWithHorizons(cad: CadRecord): Promise<GlobeAlertEvent | null> {
  const track = await fetchNeoHorizonsTrack(cad.des, cad.approachTime, TRACK_WINDOW_DAYS).catch(
    () => null,
  );
  const anchor = track?.closest ?? track?.points[0];
  if (!anchor) return null;
  return cadToGlobeEvent(cad, anchor.lat, anchor.lon, { trackPending: false });
}

/** Incoming NEO close approaches (CAD) — list immediately, Horizons tracks in background batches. */
export async function fetchSpaceGlobeEvents(): Promise<GlobeAlertEvent[]> {
  const { fetchNeoCloseApproaches } = await import("./fetchJplCad");
  const { fetchScoutSummary } = await import("./fetchJplScout");

  const now = Date.now();
  const [cadList, scoutList] = await Promise.all([
    fetchNeoCloseApproaches({ daysAhead: 14, distMaxAu: 0.12, limit: 20 }).catch(() => [] as CadRecord[]),
    fetchScoutSummary().catch(() => [] as ScoutRecord[]),
  ]);

  const futureCad = cadList
    .filter((c) => c.approachTime > now - 3_600_000)
    .sort((a, b) => a.distLd - b.distLd || a.approachTime - b.approachTime);

  const events: GlobeAlertEvent[] = futureCad.map((cad) => {
    const a = pseudoAnchorFor(cad.des);
    return cadToGlobeEvent(cad, a.lat, a.lon, { trackPending: true });
  });

  const seen = new Set(events.map((e) => e.id));

  for (let i = 0; i < Math.min(futureCad.length, HORIZONS_ENRICH_LIMIT); i += HORIZONS_BATCH) {
    const batch = futureCad.slice(i, i + HORIZONS_BATCH);
    const enriched = await Promise.all(batch.map((cad) => enrichCadWithHorizons(cad)));
    for (let j = 0; j < batch.length; j++) {
      const ev = enriched[j];
      if (!ev) continue;
      const idx = events.findIndex((e) => e.id === ev.id);
      if (idx >= 0) events[idx] = ev;
    }
  }

  for (const scout of scoutList.slice(0, 3)) {
    const id = `neo-scout-${scout.objectName}`;
    if (seen.has(id)) continue;
    const track = await fetchNeoHorizonsTrack(scout.objectName, Date.now(), 3).catch(() => null);
    const anchor = track?.closest ?? track?.points[0];
    if (!anchor) {
      const a = pseudoAnchorFor(id);
      events.push(scoutToGlobeEvent(scout, a.lat, a.lon, { trackPending: true }));
    } else {
      events.push(scoutToGlobeEvent(scout, anchor.lat, anchor.lon, { trackPending: false }));
    }
    seen.add(id);
  }

  return filterNeoAlerts(
    events.sort((a, b) => {
      const da = a.approachTime ?? a.time;
      const db = b.approachTime ?? b.time;
      if (da !== db) return da - db;
      return (a.distLd ?? 99) - (b.distLd ?? 99);
    }),
  );
}

const SPACE_CAD_DAYS = 14;
const SPACE_CAD_DIST_AU = 0.5;
const SPACE_CAD_LIMIT = 40;

/** All NEO close approaches in the next 14 days — raw NASA feed for space tab. */
export async function fetchSpaceGlobeEvents24h(): Promise<GlobeAlertEvent[]> {
  const { fetchNeoCloseApproaches } = await import("./fetchJplCad");
  const { fetchScoutSummary } = await import("./fetchJplScout");
  const { filterSpacePanelNeos } = await import("./alertFilters");

  const now = Date.now();
  const [cadList, scoutList] = await Promise.all([
    fetchNeoCloseApproaches({
      daysAhead: SPACE_CAD_DAYS,
      distMaxAu: SPACE_CAD_DIST_AU,
      limit: SPACE_CAD_LIMIT,
    }).catch(() => [] as CadRecord[]),
    fetchScoutSummary().catch(() => [] as ScoutRecord[]),
  ]);

  const futureCad = cadList
    .filter((c) => c.approachTime > now)
    .sort((a, b) => a.approachTime - b.approachTime || a.distLd - b.distLd);

  const events: GlobeAlertEvent[] = futureCad.map((cad) => {
    const a = pseudoAnchorFor(cad.des);
    return cadToGlobeEvent(cad, a.lat, a.lon, { trackPending: true });
  });

  const seen = new Set(events.map((e) => e.id));

  for (let i = 0; i < Math.min(futureCad.length, HORIZONS_ENRICH_LIMIT); i += HORIZONS_BATCH) {
    const batch = futureCad.slice(i, i + HORIZONS_BATCH);
    const enriched = await Promise.all(batch.map((cad) => enrichCadWithHorizons(cad)));
    for (let j = 0; j < batch.length; j++) {
      const ev = enriched[j];
      if (!ev) continue;
      const idx = events.findIndex((e) => e.id === ev.id);
      if (idx >= 0) events[idx] = ev;
    }
  }

  for (const scout of scoutList.slice(0, 5)) {
    const id = `neo-scout-${scout.objectName}`;
    if (seen.has(id)) continue;
    const track = await fetchNeoHorizonsTrack(scout.objectName, Date.now(), 3).catch(() => null);
    const anchor = track?.closest ?? track?.points[0];
    if (!anchor) {
      const a = pseudoAnchorFor(id);
      events.push(scoutToGlobeEvent(scout, a.lat, a.lon, { trackPending: true }));
    } else {
      events.push(scoutToGlobeEvent(scout, anchor.lat, anchor.lon, { trackPending: false }));
    }
    seen.add(id);
  }

  return filterSpacePanelNeos(
    events.sort((a, b) => {
      const da = a.approachTime ?? a.time;
      const db = b.approachTime ?? b.time;
      if (da !== db) return da - db;
      return (a.distLd ?? 99) - (b.distLd ?? 99);
    }),
  );
}

/** Space tab: NASA CAD (14d) + famous periodic catalog for rich 3D scene. */
export async function fetchSpaceTabEvents(): Promise<GlobeAlertEvent[]> {
  const { filterSpaceAlerts } = await import("./alertFilters");
  const raw = await fetchSpaceGlobeEvents24h();
  const alerts = filterSpaceAlerts(raw);
  return mergeShowcaseWithReal(alerts);
}

export { formatNeoEta };
