import { fetchJson } from "../../webSearch/fetchJson";
import type { SearchSourceResult } from "../../webSearch/types";
import type { LiveDisasterItem } from "../../liveWorld/types";
import { EARTH_LIVE_WINDOW_MS } from "../../globalAlerts/types";
import { coerceHttpUrl, coerceText } from "../../searchResults/coerceHitUrl";

type GdacsFeature = {
  geometry?: {
    type?: string;
    coordinates?: number[] | number[][] | number[][][] | number[][][][];
  };
  properties?: {
    eventname?: string;
    name?: string;
    alertlevel?: string;
    episodealertlevel?: string;
    country?: string;
    eventtype?: string;
    fromdate?: string;
    todate?: string;
    datemodified?: string;
    iscurrent?: string | boolean;
    url?: string | { geometry?: string; report?: string; details?: string };
    eventid?: number;
    episodeid?: number;
    severitydata?: { severitytext?: string };
  };
};

const GDACS_LIVE_URL =
  "https://www.gdacs.org/gdacsapi/api/events/geteventlist/events4app";

function gdacsSearchUrl(fromDate: Date, toDate: Date, eventlist = "EQ;TC;FL;VO;WF;TS"): string {
  const fmt = (d: Date) => d.toISOString().slice(0, 10);
  const params = new URLSearchParams({
    eventlist,
    fromdate: fmt(fromDate),
    todate: fmt(toDate),
    alertlevel: "Green;Orange;Red",
  });
  return `https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH?${params}`;
}

function gdacsItemKey(item: LiveDisasterItem): string {
  if (item.eventId != null && item.episodeId != null) return `${item.eventId}-${item.episodeId}`;
  return `${item.eventName}-${item.startTime ?? 0}`;
}

function gdacsItemInWindow(item: LiveDisasterItem, sinceMs: number, now = Date.now()): boolean {
  const start = item.startTime ?? item.dateModified ?? 0;
  const end = item.endTime ?? now;
  const modified = item.dateModified ?? start;
  if (start <= now && end >= sinceMs) return true;
  if (start >= sinceMs && start <= now + 86_400_000) return true;
  if (modified >= sinceMs && modified <= now + 86_400_000) return true;
  return false;
}

function mergeGdacsItems(lists: LiveDisasterItem[][]): LiveDisasterItem[] {
  const merged = new Map<string, LiveDisasterItem>();
  for (const list of lists) {
    for (const item of list) {
      merged.set(gdacsItemKey(item), item);
    }
  }
  return [...merged.values()];
}

const ONGOING_GDACS = new Set(["TC", "FL", "VO", "WF", "TS"]);

function parseGdacsTime(raw?: string): number | undefined {
  if (!raw) return undefined;
  const t = Date.parse(raw.includes("T") && !raw.endsWith("Z") ? `${raw}Z` : raw);
  return Number.isFinite(t) ? t : undefined;
}

const extractGdacsLonLat = (geom: GdacsFeature["geometry"]): { lat?: number; lon?: number } => {
  if (!geom?.coordinates) return {};
  const c = geom.coordinates;
  if (geom.type === "Point" && Array.isArray(c) && c.length >= 2) {
    const lon = Number(c[0]);
    const lat = Number(c[1]);
    if (Number.isFinite(lat) && Number.isFinite(lon)) return { lat, lon };
  }
  const flat: number[] = [];
  const walk = (node: unknown): void => {
    if (Array.isArray(node)) {
      if (node.length >= 2 && typeof node[0] === "number" && typeof node[1] === "number") {
        flat.push(node[0], node[1]);
      } else {
        node.forEach(walk);
      }
    }
  };
  walk(c);
  if (flat.length >= 2) {
    let sx = 0;
    let sy = 0;
    let n = 0;
    for (let i = 0; i + 1 < flat.length; i += 2) {
      sx += flat[i];
      sy += flat[i + 1];
      n++;
    }
    if (n > 0) return { lon: sx / n, lat: sy / n };
  }
  return {};
};

export const parseGdacsFeatures = (feats: GdacsFeature[]): LiveDisasterItem[] =>
  feats.map((f) => {
    const p = f.properties ?? {};
    const { lat, lon } = extractGdacsLonLat(f.geometry);
    const alertLevel = coerceText(p.episodealertlevel || p.alertlevel);
    return {
      eventName: coerceText(p.eventname, "—") || coerceText(p.name, "—"),
      country: coerceText(p.country),
      alertLevel,
      eventType: coerceText(p.eventtype) || undefined,
      url: coerceHttpUrl(p.url, "https://www.gdacs.org"),
      lat,
      lon,
      eventId: Number.isFinite(Number(p.eventid)) ? Number(p.eventid) : undefined,
      episodeId: Number.isFinite(Number(p.episodeid)) ? Number(p.episodeid) : undefined,
      geometryUrl:
        p.url && typeof p.url === "object" && typeof p.url.geometry === "string"
          ? p.url.geometry
          : undefined,
      severityText: coerceText(p.severitydata?.severitytext) || undefined,
      startTime: parseGdacsTime(p.fromdate),
      endTime: parseGdacsTime(p.todate),
      dateModified: parseGdacsTime(p.datemodified),
      isCurrent: p.iscurrent === true || p.iscurrent === "true",
    };
  });

/** Keep only what GDACS marks as current and still active — no stale EQ/DR history. */
export function isGdacsEventLive(item: LiveDisasterItem, now = Date.now()): boolean {
  if (!/orange|red/i.test(item.alertLevel)) return false;
  if (item.eventType === "EQ") return false;
  if (item.eventType === "DR") return false;
  if (!item.isCurrent) return false;
  if (item.endTime != null && item.endTime < now) return false;

  const modified = item.dateModified ?? item.startTime ?? 0;
  if (ONGOING_GDACS.has(item.eventType ?? "")) return true;
  return now - modified <= EARTH_LIVE_WINDOW_MS;
}

export const formatGdacsDisasterText = (items: LiveDisasterItem[]): string => {
  if (!items.length) return "";
  return [
    "אירועי טבע (GDACS) · פעילים עכשיו:",
    ...items.map(
      (p, i) =>
        `${i + 1}. ${p.eventName} · ${p.country} · ${p.alertLevel}${p.eventType ? ` · ${p.eventType}` : ""}`,
    ),
  ].join("\n");
};

export const fetchDisasterSearch = async (_query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "gdacs-disasters" as const;
  const label = "אסונות (GDACS)";
  try {
    const data = await fetchJson<{ features?: GdacsFeature[] }>(GDACS_LIVE_URL);
    const list = parseGdacsFeatures(data.features ?? []).filter(isGdacsEventLive).slice(0, 16);
    if (!list.length) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: "אין אירועים פעילים",
        latencyMs: Math.round(performance.now() - started),
      };
    }
    return {
      provider,
      label,
      ok: true,
      text: formatGdacsDisasterText(list),
      url: "https://www.gdacs.org",
      latencyMs: Math.round(performance.now() - started),
    };
  } catch (err) {
    return {
      provider,
      label,
      ok: false,
      text: "",
      error: err instanceof Error ? err.message : "שגיאה",
      latencyMs: Math.round(performance.now() - started),
    };
  }
};

/** GDACS latest feed — current orange/red TC/FL/VO/WF only (matches gdacs.org active alerts). */
export const fetchGdacsDisastersForCache = async (): Promise<{
  items: LiveDisasterItem[];
  feedLabel: string;
} | null> => {
  try {
    const data = await fetchJson<{ features?: GdacsFeature[] }>(GDACS_LIVE_URL);
    const items = parseGdacsFeatures(data.features ?? []).filter(isGdacsEventLive).slice(0, 20);
    if (!items.length) return null;
    return { items, feedLabel: "GDACS · פעיל עכשיו" };
  } catch {
    return null;
  }
};

/** GDACS SEARCH — historical window (weather + disasters, merged sources). */
export const fetchGdacsDisastersHistorical = async (
  sinceMs: number,
): Promise<{ items: LiveDisasterItem[]; feedLabel: string } | null> => {
  const to = new Date();
  const from = new Date(sinceMs);
  const now = Date.now();

  try {
    const weatherUrl = gdacsSearchUrl(from, to, "TC;FL;VO;WF;TS");
    const fullUrl = gdacsSearchUrl(from, to);
    const [weatherData, fullData, liveData] = await Promise.all([
      fetchJson<{ features?: GdacsFeature[] }>(weatherUrl, undefined, { timeoutMs: 22_000 }).catch(
        () => null,
      ),
      fetchJson<{ features?: GdacsFeature[] }>(fullUrl, undefined, { timeoutMs: 22_000 }).catch(
        () => null,
      ),
      sinceMs > now - 4 * 86_400_000
        ? fetchJson<{ features?: GdacsFeature[] }>(GDACS_LIVE_URL, undefined, {
            timeoutMs: 14_000,
          }).catch(() => null)
        : Promise.resolve(null),
    ]);

    const parsed = mergeGdacsItems([
      parseGdacsFeatures(weatherData?.features ?? []),
      parseGdacsFeatures(fullData?.features ?? []),
      parseGdacsFeatures(liveData?.features ?? []),
    ]).filter((item) => gdacsItemInWindow(item, sinceMs, now));

    if (!parsed.length) return null;
    return { items: parsed.slice(0, 150), feedLabel: "GDACS · ארכיון" };
  } catch {
    try {
      const data = await fetchJson<{ features?: GdacsFeature[] }>(GDACS_LIVE_URL, undefined, {
        timeoutMs: 14_000,
      });
      const items = parseGdacsFeatures(data.features ?? [])
        .filter((item) => gdacsItemInWindow(item, sinceMs, now))
        .slice(0, 100);
      if (!items.length) return null;
      return { items, feedLabel: "GDACS · ארכיון" };
    } catch {
      return null;
    }
  }
};
