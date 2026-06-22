import { fetchJson } from "../../webSearch/fetchJson";
import type { SearchSourceResult } from "../../webSearch/types";
import type { LiveDisasterItem } from "../../liveWorld/types";
import { coerceHttpUrl, coerceText } from "../../searchResults/coerceHitUrl";

type GdacsFeature = {
  properties?: {
    eventname?: string;
    alertlevel?: string;
    country?: string;
    eventtype?: string;
    url?: string;
  };
};

export const parseGdacsFeatures = (feats: GdacsFeature[]): LiveDisasterItem[] =>
  feats.map((f) => {
    const p = f.properties ?? {};
    return {
      eventName: coerceText(p.eventname, "—"),
      country: coerceText(p.country),
      alertLevel: coerceText(p.alertlevel),
      eventType: coerceText(p.eventtype) || undefined,
      url: coerceHttpUrl(p.url, "https://www.gdacs.org"),
    };
  });

export const formatGdacsDisasterText = (items: LiveDisasterItem[]): string => {
  if (!items.length) return "";
  return [
    "אירועי טבע (GDACS):",
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
    const year = new Date().getFullYear();
    const data = await fetchJson<{ features?: GdacsFeature[] }>(
      `https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH?eventlist=EQ;TC;FL;VO;WF&fromDate=${year - 1}-01-01&toDate=${year}-12-31&alertlevel=Green;Orange;Red`,
    );
    const feats = data.features ?? [];
    const active = feats.filter((f) => /orange|red/i.test(f.properties?.alertlevel ?? ""));
    const list = parseGdacsFeatures(active.length ? active : feats).slice(0, 12);
    if (!list.length) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: "אין אירועים",
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

/** Background cache fetch — returns structured items (no query needed). */
export const fetchGdacsDisastersForCache = async (): Promise<{
  items: LiveDisasterItem[];
  feedLabel: string;
} | null> => {
  try {
    const year = new Date().getFullYear();
    const data = await fetchJson<{ features?: GdacsFeature[] }>(
      `https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH?eventlist=EQ;TC;FL;VO;WF&fromDate=${year - 1}-01-01&toDate=${year}-12-31&alertlevel=Green;Orange;Red`,
    );
    const feats = data.features ?? [];
    const active = feats.filter((f) => /orange|red/i.test(f.properties?.alertlevel ?? ""));
    const items = parseGdacsFeatures(active.length ? active : feats).slice(0, 12);
    if (!items.length) return null;
    return { items, feedLabel: "אסונות (GDACS)" };
  } catch {
    return null;
  }
};
