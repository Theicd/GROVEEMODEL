import { fetchJson } from "../fetchJson";
import type { SearchSourceResult } from "../types";

type UsgsFeature = {
  properties: {
    mag: number | null;
    place: string | null;
    time: number;
    url: string | null;
    type: string | null;
    tsunami?: number;
  };
};

type UsgsFeed = {
  features?: UsgsFeature[];
};

const formatQuake = (f: UsgsFeature): string => {
  const p = f.properties;
  const when = new Date(p.time).toISOString().replace("T", " ").slice(0, 19);
  const mag = p.mag != null ? p.mag.toFixed(1) : "?";
  const tsunami = p.tsunami === 1 ? " · אזהרת צונאmi" : "";
  return `- M${mag} · ${p.place ?? "unknown"} · ${when} UTC${tsunami}\n  ${p.url ?? ""}`;
};

export const fetchEarthquakeSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "usgs-earthquake" as const;
  const label = "רעידות אדמה (USGS)";
  try {
    const feed = await fetchJson<UsgsFeed>(
      "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson",
    );
    const features = feed.features ?? [];
    if (!features.length) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: "אין נתונים מהפיד",
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const qLower = query.toLowerCase();
    const regionTokens = qLower
      .split(/[\s,.!?]+/)
      .filter((w) => w.length >= 4 && !/רעיד|אדמה|earthquake|recent|latest|היום|עכשיו/.test(w));

    let filtered = features;
    if (regionTokens.length) {
      const regional = features.filter((f) => {
        const place = (f.properties.place ?? "").toLowerCase();
        return regionTokens.some((t) => place.includes(t));
      });
      if (regional.length) filtered = regional;
    }

    const sorted = [...filtered].sort((a, b) => (b.properties.mag ?? 0) - (a.properties.mag ?? 0));
    const top = sorted.slice(0, 8);
    const text = [
      `רעידות אדמה (24 שעות אחרונות, ${top.length} רשומות):`,
      ...top.map(formatQuake),
    ].join("\n");

    return {
      provider,
      label,
      ok: true,
      text,
      url: "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson",
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
