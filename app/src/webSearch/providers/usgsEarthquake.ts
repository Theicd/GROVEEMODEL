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

/** Hebrew / English region hints → USGS place substrings. */
const REGION_ALIASES: Record<string, string[]> = {
  יפן: ["japan", "honshu", "hokkaido", "kyushu", "okinawa", "ryukyu", "nippon"],
  japan: ["japan", "honshu", "hokkaido", "kyushu", "okinawa", "ryukyu", "nippon"],
  ישראל: ["israel", "dead sea", "sinai"],
  israel: ["israel", "dead sea", "sinai"],
  טורקיה: ["turkey", "turkiye"],
  turkey: ["turkey", "turkiye"],
  יוון: ["greece", "aegean", "crete"],
  greece: ["greece", "aegean", "crete"],
  קליפורניה: ["california", "san francisco", "los angeles"],
  california: ["california", "san francisco", "los angeles"],
  צילה: ["chile"],
  chile: ["chile"],
  אינדונזיה: ["indonesia", "sumatra", "java", "sulawesi"],
  indonesia: ["indonesia", "sumatra", "java", "sulawesi"],
  פיליפינים: ["philippines", "mindanao", "luzon"],
  philippines: ["philippines", "mindanao", "luzon"],
};

const formatQuake = (f: UsgsFeature): string => {
  const p = f.properties;
  const when = new Date(p.time).toISOString().replace("T", " ").slice(0, 19);
  const mag = p.mag != null ? p.mag.toFixed(1) : "?";
  const tsunami = p.tsunami === 1 ? " · אזהרת צונאmi" : "";
  return `- M${mag} · ${p.place ?? "unknown"} · ${when} UTC${tsunami}\n  ${p.url ?? ""}`;
};

const extractRegionMatchers = (query: string): string[] => {
  const qLower = query.toLowerCase();
  const matchers = new Set<string>();

  for (const [key, aliases] of Object.entries(REGION_ALIASES)) {
    if (qLower.includes(key.toLowerCase())) {
      aliases.forEach((a) => matchers.add(a));
    }
  }

  qLower
    .split(/[\s,.!?]+/)
    .filter((w) => w.length >= 4 && !/רעיד|אדמה|earthquake|recent|latest|היום|עכשיו|שעות|hours|האם|הייתה|שבוע|week/.test(w))
    .forEach((w) => matchers.add(w));

  return [...matchers];
};

export const fetchEarthquakeSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "usgs-earthquake" as const;
  const label = "רעידות אדמה (USGS)";
  try {
    const useWeek = /(?:ה)?שבוע|this\s+week|48\s*(?:שעות|hours)|(?:7|שבע)\s*(?:ימים|days)|\bweek\b/i.test(query);
    const feedUrl = useWeek
      ? "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_week.geojson"
      : "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson";
    const windowLabel = useWeek ? "7 ימים" : "24 שעות";

    const feed = await fetchJson<UsgsFeed>(feedUrl);
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

    const regionMatchers = extractRegionMatchers(query);

    let filtered = features;
    if (regionMatchers.length) {
      filtered = features.filter((f) => {
        const place = (f.properties.place ?? "").toLowerCase();
        return regionMatchers.some((t) => place.includes(t));
      });
    }

    const sorted = [...filtered].sort((a, b) => (b.properties.mag ?? 0) - (a.properties.mag ?? 0));
    const top = sorted.slice(0, 8);
    const regionNote =
      regionMatchers.length && filtered.length === 0
        ? `לא נמצאו רעידות באזור (${regionMatchers.slice(0, 3).join(", ")}) ב-${windowLabel}.`
        : regionMatchers.length && filtered.length > 0
          ? `מסונן לפי: ${regionMatchers.slice(0, 4).join(", ")}.`
          : "";

    const text =
      regionMatchers.length && filtered.length === 0
        ? [`אין רעידות אדמה מדווחות ב-${windowLabel} באזור ${regionMatchers.slice(0, 3).join(", ")} (USGS).`, regionNote]
            .filter(Boolean)
            .join("\n")
        : [
            `סה"כ ${filtered.length} רעידות ב-${windowLabel} (USGS). ${top.length} הגדולות:`,
            regionNote,
            ...top.map(formatQuake),
          ]
            .filter(Boolean)
            .join("\n");

    return {
      provider,
      label,
      ok: true,
      text,
      url: feedUrl,
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
