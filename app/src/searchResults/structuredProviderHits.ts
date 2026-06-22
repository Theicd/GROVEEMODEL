import type { SearchProviderId, SearchSourceResult } from "../webSearch/types";
import type { UnifiedSearchHit } from "./types";

const linesOf = (text: string): string[] => text.split("\n").map((l) => l.trim()).filter(Boolean);

const firstMatching = (lines: string[], re: RegExp): string | undefined =>
  lines.find((l) => re.test(l));

export const parseWeatherStructuredHits = (s: SearchSourceResult): UnifiedSearchHit[] => {
  if (!s.text.trim()) return [];
  const lines = linesOf(s.text);
  const place = firstMatching(lines, /^מיקום:/)?.replace(/^מיקום:\s*/, "") ?? "מזג אוויר";
  const temp = firstMatching(lines, /^טמפרטורה:/)?.replace(/^טמפרטורה:\s*/, "");
  const condition = firstMatching(lines, /^מצב:/)?.replace(/^מצב:\s*/, "");
  const forecast = lines.filter((l) => l.startsWith("- ") || l.startsWith("תחזית"));
  const title = temp ? `${place}: ${temp}` : `מזג אוויר · ${place}`;
  const snippet = [condition, ...forecast.slice(0, 4)].filter(Boolean).join(" · ");
  return [
    {
      id: `weather-${place.slice(0, 24)}`,
      kind: "weather",
      title,
      url: s.url ?? "https://open-meteo.com/",
      snippet: snippet || lines.slice(0, 6).join(" · "),
      sourceLabel: s.label,
      provider: s.provider as SearchProviderId,
      score: 78,
      summarizable: false,
      meta: { engine: "Open-Meteo" },
    },
  ];
};

export const parseMarineStructuredHits = (s: SearchSourceResult): UnifiedSearchHit[] => {
  if (!s.text.trim()) return [];
  const lines = linesOf(s.text);
  const place = firstMatching(lines, /^מיקום:/)?.replace(/^מיקום:\s*/, "") ?? "ים";
  const wave = firstMatching(lines, /^גובה גל:/)?.replace(/^גובה גל:\s*/, "");
  const wind = firstMatching(lines, /^רוח:/)?.replace(/^רוח:\s*/, "");
  const title = wave ? `גלים · ${place}: ${wave}` : `ים · ${place}`;
  return [
    {
      id: `marine-${place.slice(0, 24)}`,
      kind: "marine",
      title,
      url: s.url ?? "https://open-meteo.com/",
      snippet: [wave, wind].filter(Boolean).join(" · ") || lines.slice(0, 5).join(" · "),
      sourceLabel: s.label,
      provider: s.provider as SearchProviderId,
      score: 76,
      summarizable: false,
      meta: { engine: "Open-Meteo Marine" },
    },
  ];
};

export const parsePlacesStructuredHits = (s: SearchSourceResult): UnifiedSearchHit[] => {
  if (!s.text.trim()) return [];
  const lines = linesOf(s.text);
  const hits: UnifiedSearchHit[] = [];
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(/^(\d+)\.\s+(.+)$/);
    if (!m) continue;
    const idx = m[1];
    const name = m[2].trim();
    const next = lines[i + 1] ?? "";
    const coords = next.match(/(\d+\.\d+),\s*(\d+\.\d+)/);
    hits.push({
      id: `place-${idx}-${name.slice(0, 20)}`,
      kind: "place",
      title: name.split(" · ")[0]?.trim() || name,
      url:
        s.url ??
        `https://www.openstreetmap.org/search?query=${encodeURIComponent(name)}${
          coords ? `#map=15/${coords[1]}/${coords[2]}&layers=P` : ""
        }`,
      snippet: coords ? `${coords[1]}, ${coords[2]}` : "OpenStreetMap / Nominatim",
      sourceLabel: "OpenStreetMap",
      provider: "nominatim-places",
      score: 74 - Number(idx),
      summarizable: false,
      meta: { engine: "OSM" },
    });
    if (hits.length >= 5) break;
  }
  if (!hits.length && s.geo?.lat != null && s.geo.lon != null) {
    hits.push({
      id: "place-geo",
      kind: "place",
      title: s.geo.label ?? "מיקום",
      url: s.url ?? `https://www.openstreetmap.org/#map=15/${s.geo.lat}/${s.geo.lon}`,
      snippet: `${s.geo.lat}, ${s.geo.lon}`,
      sourceLabel: "OpenStreetMap",
      provider: "nominatim-places",
      score: 75,
      summarizable: false,
      meta: { engine: "OSM" },
    });
  }
  return hits;
};

export const parseRouteStructuredHits = (s: SearchSourceResult): UnifiedSearchHit[] => {
  if (!s.text.trim()) return [];
  const lines = linesOf(s.text);
  const from = firstMatching(lines, /^מ:/)?.replace(/^מ:\s*/, "").split("(")[0]?.trim();
  const to = firstMatching(lines, /^אל:/)?.replace(/^אל:\s*/, "").split("(")[0]?.trim();
  const km = firstMatching(lines, /ק"מ/)?.match(/([\d.]+)\s*ק"מ/)?.[1];
  const time = firstMatching(lines, /^זמן/)?.replace(/^זמן נסיעה משוער:\s*/, "");
  const title = from && to ? `${from} → ${to}` : "מסלול נסיעה";
  const snippet = [km ? `${km} ק"מ` : "", time ? `~${time}` : ""].filter(Boolean).join(" · ");
  return [
    {
      id: `route-${title.slice(0, 16)}`,
      kind: "route",
      title,
      url: s.url ?? "https://www.openstreetmap.org/directions",
      snippet: snippet || lines.join(" · "),
      sourceLabel: "OSRM · OpenStreetMap",
      provider: "osrm-distance",
      score: 77,
      summarizable: false,
      meta: { engine: "OSRM" },
    },
  ];
};

export const parseStructuredProviderHits = (s: SearchSourceResult): UnifiedSearchHit[] => {
  switch (s.provider) {
    case "open-meteo":
    case "open-meteo-air-quality":
      return parseWeatherStructuredHits(s);
    case "open-meteo-marine":
      return parseMarineStructuredHits(s);
    case "nominatim-places":
      return parsePlacesStructuredHits(s);
    case "osrm-distance":
      return parseRouteStructuredHits(s);
    default:
      return [];
  }
};
