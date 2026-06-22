import type { LiveShipItem, LiveWorldSnapshot } from "../liveWorld/types";
import { getLiveWorldSnapshotForPanel } from "../liveWorld/snapshotStore";
import { isMarineInfraQuery, isShipsQuery } from "../webSearch/intents";
import type { UnifiedSearchHit } from "./types";
import { faviconForUrl } from "./sourceBranding";
import { coerceText } from "./coerceHitUrl";

const DIGITRAFFIC_HOME = "https://meri.digitraffic.fi/en/web/ais/vessels";
const OSM_HOME = "https://www.openstreetmap.org";

export const SERP_SHIP_CARD_CAP = 64;
export const SERP_MARINE_INFRA_CARD_CAP = 24;

const slug = (s: string): string =>
  s
    .toLowerCase()
    .replace(/[^\w\u0590-\u05ff]+/g, "-")
    .slice(0, 64);

const osmMapUrl = (lat: number, lon: number): string =>
  `https://www.openstreetmap.org/?mlat=${lat.toFixed(4)}&mlon=${lon.toFixed(4)}#map=11/${lat}/${lon}`;

const parseShipSourceTag = (tag: string): "ais" | "globe" | "route-marker" | "aisstream" | null => {
  const t = tag.trim();
  if (t === "AIS") return "ais";
  if (/aisstream/i.test(t)) return "aisstream";
  if (/עולם חי/i.test(t)) return "globe";
  if (/מסלול|הדגמה|demo/i.test(t)) return "route-marker";
  return null;
};

const shipLineRe =
  /^\d+\.\s+(.+?)\s·\s*(AIS|AISStream|עולם חי|מסלול \(הדגמה\))\s·\s*([-\d.]+),([-\d.]+)\s·\s*([\d.]+|—)\s*kn(?:\s→\s*(.+))?$/;

const marineInfraLineRe =
  /^\d+\.\s+(.+?)\s·\s*(harbour|buoy|lighthouse|pier|other)(?:\s·\s*([-\d.]+),([-\d.]+))?$/i;

const infraLabelHe: Record<string, string> = {
  harbour: "נמל",
  buoy: "מצוף",
  lighthouse: "מגדלור",
  pier: "רציף",
  other: "תשתית ימית",
};

const shipSourceLabelHe: Record<string, string> = {
  ais: "Digitraffic AIS",
  aisstream: "AISStream חי",
  globe: "עולם חי",
  "route-marker": "מסלול (הדגמה)",
};

export type ParsedShipRow = {
  name: string;
  lat: number;
  lon: number;
  speedKn?: number;
  destination?: string;
  source: "ais" | "globe" | "route-marker" | "aisstream";
};

export type ParsedMarineInfraRow = {
  name: string;
  kind: string;
  lat?: number;
  lon?: number;
};

export const parseShipSampleLine = (line: string): ParsedShipRow | null => {
  const m = line.trim().match(shipLineRe);
  if (!m) return null;
  const source = parseShipSourceTag(m[2]);
  if (!source) return null;
  const lat = parseFloat(m[3]);
  const lon = parseFloat(m[4]);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
  const speedRaw = m[5]?.trim();
  const speedKn = speedRaw && speedRaw !== "—" ? parseFloat(speedRaw) : undefined;
  return {
    name: m[1].trim(),
    lat,
    lon,
    speedKn: Number.isFinite(speedKn) ? speedKn : undefined,
    destination: m[6]?.trim() || undefined,
    source,
  };
};

export const parseMarineInfraSampleLine = (line: string): ParsedMarineInfraRow | null => {
  const m = line.trim().match(marineInfraLineRe);
  if (!m) return null;
  const lat = m[3] != null ? parseFloat(m[3]) : undefined;
  const lon = m[4] != null ? parseFloat(m[4]) : undefined;
  return {
    name: m[1].trim(),
    kind: m[2].toLowerCase(),
    lat: Number.isFinite(lat) ? lat : undefined,
    lon: Number.isFinite(lon) ? lon : undefined,
  };
};

export const parseRegionLabelFromProviderText = (text: string): string | undefined => {
  const m = text.match(/^אזור:\s*(.+)$/m);
  return m?.[1]?.trim().replace(/\s*\(OpenStreetMap.*\)$/, "").replace(/\s*\(OSM\)$/, "");
};

export const parseLiveShipCountFromText = (text: string): number | undefined => {
  const answer = text.match(/ANSWER \(ships live\):\s*(\d+)/i)?.[1];
  if (answer != null) return parseInt(answer, 10);
  const liveLine = text.match(/דיווח AIS חי \+ עולם חי:\s*(\d+)/);
  if (liveLine) return parseInt(liveLine[1], 10);
  return undefined;
};

export const shipRowToHit = (row: ParsedShipRow, index: number, regionLabel?: string): UnifiedSearchHit => {
  const sourceHe = shipSourceLabelHe[row.source] ?? row.source;
  const coords = `${row.lat.toFixed(2)}°, ${row.lon.toFixed(2)}°`;
  const speedPart = row.speedKn != null ? `${row.speedKn.toFixed(1)} kn` : "— kn";
  const destPart = row.destination ? ` → ${row.destination}` : "";
  const scoreBase =
    row.source === "aisstream" ? 72 : row.source === "ais" ? 68 : row.source === "globe" ? 62 : 28;

  return {
    id: `ship-${slug(row.name)}-${row.lat.toFixed(2)}-${row.lon.toFixed(2)}-${index}`,
    kind: "ship",
    title: row.name,
    titleOriginal: row.name,
    url: osmMapUrl(row.lat, row.lon),
    snippet: `${coords} · ${speedPart}${destPart}`,
    snippetOriginal: `${coords} · ${speedPart}${destPart}`,
    sourceLabel:
      row.source === "aisstream"
        ? "AISStream"
        : row.source === "globe"
          ? "עולם חי"
          : "Digitraffic AIS",
    provider: "ais-ships",
    faviconUrl: faviconForUrl(DIGITRAFFIC_HOME),
    score: scoreBase + (row.speedKn != null && row.speedKn > 0.5 ? 4 : 0),
    meta: {
      engine: sourceHe,
      shipLat: row.lat,
      shipLon: row.lon,
      speedKn: row.speedKn,
      destination: row.destination,
      shipSource: row.source,
      regionLabel,
    },
    summarizable: false,
  };
};

export const marineInfraRowToHit = (
  row: ParsedMarineInfraRow,
  index: number,
  regionLabel?: string,
): UnifiedSearchHit => {
  const kindHe = infraLabelHe[row.kind] ?? row.kind;
  const hasCoords = row.lat != null && row.lon != null;
  const coords = hasCoords ? `${row.lat!.toFixed(2)}°, ${row.lon!.toFixed(2)}°` : "";
  const url = hasCoords ? osmMapUrl(row.lat!, row.lon!) : OSM_HOME;

  return {
    id: `marine-infra-${slug(row.name)}-${index}`,
    kind: "marine",
    title: row.name,
    titleOriginal: row.name,
    url,
    snippet: coords ? `${kindHe} · ${coords}` : kindHe,
    snippetOriginal: coords ? `${kindHe} · ${coords}` : kindHe,
    sourceLabel: "OpenStreetMap",
    provider: "osm-overpass-marine",
    faviconUrl: faviconForUrl(OSM_HOME),
    score: 54,
    meta: {
      engine: kindHe,
      marineInfraKind: row.kind,
      shipLat: row.lat,
      shipLon: row.lon,
      regionLabel,
    },
    summarizable: false,
  };
};

/** Parse ais-ships provider text — live vessels only (no demo route markers). */
export const parseAisShipsText = (text: string, cap = SERP_SHIP_CARD_CAP): UnifiedSearchHit[] => {
  const regionLabel = parseRegionLabelFromProviderText(text);
  const out: UnifiedSearchHit[] = [];
  for (const line of text.split("\n")) {
    const row = parseShipSampleLine(line);
    if (!row || row.source === "route-marker") continue;
    out.push(shipRowToHit(row, out.length, regionLabel));
    if (out.length >= cap) break;
  }
  return out;
};

/** Parse osm-overpass-marine provider samples into infra cards. */
export const parseMarineInfraText = (
  text: string,
  cap = SERP_MARINE_INFRA_CARD_CAP,
): UnifiedSearchHit[] => {
  const regionLabel = parseRegionLabelFromProviderText(text);
  const out: UnifiedSearchHit[] = [];
  for (const line of text.split("\n")) {
    const row = parseMarineInfraSampleLine(line);
    if (!row) continue;
    out.push(marineInfraRowToHit(row, out.length, regionLabel));
    if (out.length >= cap) break;
  }
  return out;
};

export const liveShipItemToHit = (
  item: LiveShipItem,
  index: number,
  regionLabel?: string,
): UnifiedSearchHit | null => {
  if (!Number.isFinite(item.lat) || !Number.isFinite(item.lon)) return null;
  const source =
    item.source === "route-marker" ||
    item.source === "med-fallback" ||
    /demo|הדגמה/i.test(item.source ?? "")
      ? ("route-marker" as const)
      : item.source === "aisstream"
        ? ("aisstream" as const)
        : item.source === "globe" || item.source === "digitraffic"
        ? ("globe" as const)
        : ("ais" as const);
  if (source === "route-marker") return null;
  return shipRowToHit(
    {
      name: coerceText(item.name, "AIS"),
      lat: item.lat,
      lon: item.lon,
      speedKn: item.speedKn,
      destination: item.destination,
      source,
    },
    index,
    regionLabel,
  );
};

export const buildLiveShipHitsFromSnapshot = (
  snapshot: LiveWorldSnapshot | null,
  cap = SERP_SHIP_CARD_CAP,
): UnifiedSearchHit[] => {
  if (!snapshot?.ships?.items?.length) return [];
  const regionLabel = snapshot.ships.regionLabel;
  const sorted = [...snapshot.ships.items].sort((a, b) => {
    const rank = (s: LiveShipItem) =>
      s.source === "aisstream" ? 0 : s.source === "globe" ? 1 : s.source === "ais" || s.source === "digitraffic" ? 2 : 9;
    return rank(a) - rank(b);
  });
  const out: UnifiedSearchHit[] = [];
  for (const item of sorted) {
    const hit = liveShipItemToHit(item, out.length, regionLabel);
    if (!hit) continue;
    out.push(hit);
    if (out.length >= cap) break;
  }
  return out;
};

export const buildLiveMarineInfraHitsFromSnapshot = (
  snapshot: LiveWorldSnapshot | null,
  cap = SERP_MARINE_INFRA_CARD_CAP,
): UnifiedSearchHit[] => {
  if (!snapshot?.marineInfra?.items?.length) return [];
  const regionLabel = snapshot.marineInfra.regionLabel;
  const out: UnifiedSearchHit[] = [];
  for (const item of snapshot.marineInfra.items) {
    out.push(
      marineInfraRowToHit(
        {
          name: item.name,
          kind: item.kind,
          lat: item.lat,
          lon: item.lon,
        },
        out.length,
        regionLabel,
      ),
    );
    if (out.length >= cap) break;
  }
  return out;
};

export const getLiveShipHits = (): UnifiedSearchHit[] =>
  buildLiveShipHitsFromSnapshot(getLiveWorldSnapshotForPanel());

const isShipPanelHit = (h: UnifiedSearchHit): boolean => h.kind === "ship" || h.kind === "marine";

export const mergeLiveShipHits = (hits: UnifiedSearchHit[], query = ""): UnifiedSearchHit[] => {
  const live = filterShipHitsForQuery(getLiveShipHits(), query);
  if (!live.length) return hits;
  const existing = new Set(
    hits.filter((h) => h.kind === "ship").map((h) => `${h.title}:${h.meta?.shipLat}:${h.meta?.shipLon}`),
  );
  const merged = [...hits];
  for (const h of live) {
    const key = `${h.title}:${h.meta?.shipLat}:${h.meta?.shipLon}`;
    if (existing.has(key)) continue;
    merged.push(h);
    existing.add(key);
  }
  return merged.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
};

/** Merge cached live ships + marine infra for empty SERP panel open. */
export const buildLivePanelShipHits = (): UnifiedSearchHit[] => {
  const snap = getLiveWorldSnapshotForPanel();
  return [
    ...buildLiveShipHitsFromSnapshot(snap),
    ...buildLiveMarineInfraHitsFromSnapshot(snap),
  ];
};

export const refreshLiveShipsInPayload = (payload: SearchResultsPayload): SearchResultsPayload => {
  const nonShip = payload.hits.filter((h) => h.kind !== "ship" && h.kind !== "marine");
  const shipHits = buildLivePanelShipHits();
  const hits = [...nonShip, ...shipHits].sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
  const shipsCount = shipsPanelTotal(hits);
  return {
    ...payload,
    generatedAt: Date.now(),
    hits,
    facets: {
      ...payload.facets,
      ships: shipsCount,
    },
    liveShipsNote: formatLiveShipsNote(getLiveWorldSnapshotForPanel(), "he", payload.query),
  };
};

export const shipsFacetCounts = (
  hits: UnifiedSearchHit[],
): { ships: number; marineInfra: number } => ({
  ships: hits.filter((h) => h.kind === "ship").length,
  marineInfra: hits.filter((h) => h.kind === "marine").length,
});

export const shipsPanelTotal = (hits: UnifiedSearchHit[]): number => {
  const { ships, marineInfra } = shipsFacetCounts(hits);
  return ships + marineInfra;
};

export const formatLiveShipsNote = (
  snapshot: LiveWorldSnapshot | null,
  uiLang: "he" | "en",
  query = "",
): string | undefined => {
  const region = snapshot?.ships?.regionLabel;
  const count = snapshot?.ships?.count ?? snapshot?.ships?.items?.length ?? 0;
  const wantsShips = isShipsQuery(query);
  const wantsInfra = isMarineInfraQuery(query);
  if (!count && !wantsShips && !wantsInfra) return undefined;
  const when = snapshot?.fetchedAt
    ? new Date(snapshot.fetchedAt).toLocaleTimeString(uiLang === "he" ? "he-IL" : "en-GB", {
        hour: "2-digit",
        minute: "2-digit",
      })
    : null;
  const cardCap = SERP_SHIP_CARD_CAP + SERP_MARINE_INFRA_CARD_CAP;
  const totalNote = count > cardCap ? ` · מציג ${cardCap} כרטיסים מתוך ${count}` : "";
  if (uiLang === "he") {
    if (wantsInfra && !wantsShips) {
      return `תשתיות ימיות (OSM) · אזור ${region ?? "—"} · עודכן ${when ?? "—"}`;
    }
    return `AIS חי: ${count} כלי שייט${totalNote} · ${region ?? "עולם חי"} · AISStream + Digitraffic · עודכן ${when ?? "—"}`;
  }
  return `Live AIS: ${count} vessels${totalNote} · ${region ?? "globe"} · updated ${when ?? "—"}`;
};

export const filterShipHitsForQuery = (
  hits: UnifiedSearchHit[],
  query: string,
): UnifiedSearchHit[] => {
  const q = query.trim();
  if (!q) return hits.filter(isShipPanelHit);
  if (isMarineInfraQuery(q) && !isShipsQuery(q)) {
    return hits.filter((h) => h.kind === "marine");
  }
  if (isShipsQuery(q) && !isMarineInfraQuery(q)) {
    return hits.filter((h) => h.kind === "ship");
  }
  return hits.filter(isShipPanelHit);
};
