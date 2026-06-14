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

const REGION_STOP_WORDS =
  /^(?:רעיד|רעידת|רעידות|אדמה|האדמה|earthquake|recent|latest|היום|עכשיו|שעות|hours|האם|הייתה|שבוע|week|איפה|היכן|where|what|which|חזקה|החזקה|strongest|largest|בעולם|עולם|world|worldwide|global|ברחבי|האחרונות|אחרונות|האחרונה|אחרונה|last|above|over|מעל|כמה|מידע|למשל|התרחש|התרחשו|show|display|most)$/i;

const hasExplicitRegion = (query: string): boolean => {
  const qLower = query.toLowerCase();
  return Object.keys(REGION_ALIASES).some((key) => qLower.includes(key.toLowerCase()));
};

/** Parse minimum magnitude from queries like «מעל 5.0», «M5+», «above 5». */
export const extractMinMagnitude = (query: string): number | null => {
  const patterns = [
    /מעל\s*(?:M\s*)?([\d.]+)/i,
    /above\s*(?:M\s*)?([\d.]+)/i,
    /over\s*(?:M\s*)?([\d.]+)/i,
    /(?:>=|≥|>)\s*(?:M\s*)?([\d.]+)/i,
    /M\s*([\d.]+)\s*\+/i,
  ];
  for (const re of patterns) {
    const m = query.match(re);
    if (m?.[1]) {
      const v = parseFloat(m[1]);
      if (!Number.isNaN(v) && v >= 0 && v <= 10) return v;
    }
  }
  return null;
};

const isGlobalEarthquakeQuery = (query: string): boolean => {
  if (
    /(?:בעולם|ברחבי\s+העולם|worldwide|global|around\s+the\s+world|החזקה|הכי\s+חזק|strongest|largest\s+mag)/i.test(
      query,
    )
  ) {
    return true;
  }
  const minMag = extractMinMagnitude(query);
  const wantsLast = /(?:האחרונה|אחרונה|latest|last|most\s+recent)/i.test(query);
  if (wantsLast && minMag != null && !hasExplicitRegion(query)) return true;
  if (wantsLast && !hasExplicitRegion(query) && !/(?:ב|ליד|near|around|close\s+to)\s+\S/i.test(query)) {
    return true;
  }
  return false;
};

const extractRegionMatchers = (query: string): string[] => {
  if (isGlobalEarthquakeQuery(query)) return [];

  const qLower = query.toLowerCase();
  const matchers = new Set<string>();

  for (const [key, aliases] of Object.entries(REGION_ALIASES)) {
    if (qLower.includes(key.toLowerCase())) {
      aliases.forEach((a) => matchers.add(a));
    }
  }

  qLower
    .split(/[\s,.!?]+/)
    .filter((w) => w.length >= 4 && !REGION_STOP_WORDS.test(w) && !/^[\d.]+$/.test(w))
    .forEach((w) => matchers.add(w));

  return [...matchers];
};

export const fetchEarthquakeSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "usgs-earthquake" as const;
  const label = "רעידות אדמה (USGS)";
  try {
    const minMag = extractMinMagnitude(query);
    const wantsLast = /(?:האחרונה|אחרונה|latest|last|most\s+recent)/i.test(query);
    const useWeek =
      /(?:ה)?שבוע|this\s+week|48\s*(?:שעות|hours)|(?:7|שבע)\s*(?:ימים|days)|\bweek\b/i.test(query) ||
      (minMag != null && minMag >= 4.5) ||
      (wantsLast && minMag != null);
    const feedUrl = useWeek
      ? "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_week.geojson"
      : "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson";
    const windowLabel = useWeek ? "7 ימים" : "24 שעות";

    const feed = await fetchJson<UsgsFeed>(feedUrl);
    let features = feed.features ?? [];
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

    if (minMag != null) {
      filtered = filtered.filter((f) => (f.properties.mag ?? 0) >= minMag);
    }

    const sortByTime = wantsLast && !/(?:חזק|strongest|largest|הגדול)/i.test(query);
    const sorted = [...filtered].sort((a, b) =>
      sortByTime
        ? b.properties.time - a.properties.time
        : (b.properties.mag ?? 0) - (a.properties.mag ?? 0),
    );
    const top = sorted.slice(0, 8);
    const strongest = sorted[0];
    const strongestLine =
      strongest && /(?:חזק|strongest|largest|הגדול)/i.test(query)
        ? `הרעידה החזקה ביותר: M${(strongest.properties.mag ?? 0).toFixed(1)} · ${strongest.properties.place ?? "unknown"}`
        : "";
    const lastLine =
      strongest && wantsLast && minMag != null
        ? `הרעידה האחרונה מעל M${minMag}: M${(strongest.properties.mag ?? 0).toFixed(1)} · ${strongest.properties.place ?? "unknown"} · ${new Date(strongest.properties.time).toISOString().replace("T", " ").slice(0, 19)} UTC`
        : strongest && wantsLast
          ? `הרעידה האחרונה: M${(strongest.properties.mag ?? 0).toFixed(1)} · ${strongest.properties.place ?? "unknown"} · ${new Date(strongest.properties.time).toISOString().replace("T", " ").slice(0, 19)} UTC`
          : "";
    const regionNote =
      regionMatchers.length && filtered.length === 0
        ? `לא נמצאו רעידות באזור (${regionMatchers.slice(0, 3).join(", ")}) ב-${windowLabel}.`
        : regionMatchers.length && filtered.length > 0
          ? `מסונן לפי: ${regionMatchers.slice(0, 4).join(", ")}.`
          : "";
    const magNote =
      minMag != null && filtered.length === 0 && !regionMatchers.length
        ? `אין רעידות מעל M${minMag} ב-${windowLabel} (USGS).`
        : minMag != null && filtered.length > 0
          ? `מסונן: magnitude ≥ M${minMag}.`
          : "";

    const text =
      filtered.length === 0
        ? [
            regionMatchers.length
              ? `אין רעידות אדמה מדווחות ב-${windowLabel} באזור ${regionMatchers.slice(0, 3).join(", ")} (USGS).`
              : magNote || `אין רעידות אדמה מדווחות ב-${windowLabel} (USGS).`,
            regionNote,
            magNote && regionMatchers.length ? magNote : "",
          ]
            .filter(Boolean)
            .join("\n")
        : [
            minMag != null
              ? `סה"כ ${filtered.length} רעידות מעל M${minMag} ב-${windowLabel} (USGS).`
              : `סה"כ ${filtered.length} רעידות ב-${windowLabel} (USGS). ${top.length} הגדולות:`,
            lastLine || strongestLine,
            regionNote,
            magNote && filtered.length > 0 ? magNote : "",
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
