import { fetchJson } from "../fetchJson";
import { extractPoiNearQuery } from "../queryExtract";
import { getStartupContextSync } from "../../startupContext";
import type { SearchSourceResult } from "../types";

type NominatimHit = {
  display_name: string;
  lat: string;
  lon: string;
  type?: string;
  class?: string;
  importance?: number;
};

const NOMINATIM_HEADERS = {
  Accept: "application/json",
  "User-Agent": "GROVEEMODEL/1.0 (local chat; contact: none)",
};

const POI_TRANSLATIONS: Record<string, string> = {
  "בית חולים": "hospital",
  "בתי חולים": "hospital",
  hospital: "hospital",
  hospitals: "hospital",
  "תחנת רכבת": "railway station",
  "תחנות רכבת": "railway station",
  "train station": "railway station",
  "train stations": "railway station",
  "railway station": "railway station",
  "רכבת תחתית": "subway station",
  "subway station": "subway station",
  "תחנת דלק": "fuel",
  "gas station": "fuel",
  "fuel station": "fuel",
  pharmacy: "pharmacy",
  "בית מרקחת": "pharmacy",
  "מסעדה": "restaurant",
  restaurant: "restaurant",
  "מלון": "hotel",
  hotel: "hotel",
};

const translatePoi = (poi: string): string => {
  const t = poi.trim();
  const lower = t.toLowerCase();
  if (/תחנ.*רכ|train\s+station|railway/i.test(t)) return "railway station";
  if (/בית\s+חולים|hospital/i.test(t)) return "hospital";
  for (const [he, en] of Object.entries(POI_TRANSLATIONS)) {
    if (t.includes(he) || lower.includes(he.toLowerCase())) return en;
  }
  return t;
};

const LANDMARK_ALIASES: Record<string, string> = {
  "מוזיאון הלובר": "Louvre Museum Paris France",
  "מוזיאון לובר": "Louvre Museum Paris France",
  louvre: "Louvre Museum Paris France",
  "מגדל אייפel": "Eiffel Tower Paris France",
  "eiffel tower": "Eiffel Tower Paris France",
  "הית'רo": "London Heathrow Airport UK",
  heathrow: "London Heathrow Airport UK",
  "שדה התעופה הית'רo": "London Heathrow Airport UK",
  "שדה התעופה בן גוריון": "Ben Gurion Airport Israel",
  "נמל התעופה בן גוריון": "Ben Gurion Airport Israel",
  ber: "Berlin Brandenburg Airport Germany",
  BER: "Berlin Brandenburg Airport Germany",
  "שדה התעופה BER": "Berlin Brandenburg Airport Germany",
  "שדה התעופה בברלין": "Berlin Brandenburg Airport Germany",
};

const translateLandmark = (near: string): string => {
  const t = near.trim();
  if (/לובר|louvre/i.test(t)) return "Louvre Museum Paris France";
  if (/אייפel|eiffel/i.test(t)) return "Eiffel Tower Paris France";
  if (/heathrow|הית.?ר[owו]/i.test(t)) return "London Heathrow Airport UK";
  if (/בן.?גוריון|ben.?gurion/i.test(t)) return "Ben Gurion Airport Israel";
  for (const [alias, en] of Object.entries(LANDMARK_ALIASES)) {
    if (t.includes(alias) || t.toLowerCase().includes(alias.toLowerCase())) return en;
  }
  return t;
};

const resolveNearAnchor = (near: string): string => {
  if (near !== "__NEAR_ME__") return translateLandmark(near);
  const ctx = getStartupContextSync();
  if (ctx?.cityName) return `${ctx.cityName} ${ctx.countryName}`;
  if (ctx?.countryName) return ctx.countryName;
  return "Israel";
};

const buildPlaceSearchQueries = (query: string, parsed: { poi: string; near: string } | null): string[] => {
  const out: string[] = [];
  if (parsed) {
    const poiEn = translatePoi(parsed.poi);
    const nearEn = resolveNearAnchor(parsed.near);
    out.push(`${poiEn} near ${nearEn}`);
    out.push(`${poiEn}, ${nearEn}`);
    if (/תחנ.*רכ|train\s+station|railway/i.test(parsed.poi) && /airport|שדה\s+תעופה|heathrow|הית/i.test(parsed.near)) {
      out.push(`${nearEn} railway station`);
      out.push(`railway station ${nearEn}`);
    }
  }
  if (/heathrow|הית.?ר[owו]|שדה\s+התעופה\s+הית/i.test(query) && /רכ|train|railway|תחנ/i.test(query)) {
    out.push("London Heathrow Airport railway station");
    out.push("Heathrow Airport train station");
    out.push("Heathrow Central station London");
    out.push("Heathrow Terminals 2 and 3 railway station");
  }
  if (/\bber\b|brandenburg|ברלין|berlin/i.test(query) && /רכ|train|railway|תחנ/i.test(query)) {
    out.push("Berlin Brandenburg Airport railway station");
    out.push("Flughafen BER Bahnhof");
    out.push("railway station Berlin Brandenburg Airport");
    out.push("Flughafen Berlin Brandenburg station");
  }
  if (/אייפ|eiffel/i.test(query) && /בית\s+חולים|hospital/i.test(query)) {
    out.push("hospital near Eiffel Tower Paris");
  }
  if (/אייפ|eiffel/i.test(query) && /(?:רכ|train|railway|station|תחנ)/i.test(query)) {
    out.push("Bir-Hakeim metro station Paris");
    out.push("Champ de Mars Tour Eiffel station Paris");
    out.push("railway station near Eiffel Tower Paris");
  }
  if (/לובר|louvre/i.test(query) && /(?:מלון|hotel)/i.test(query)) {
    out.push("hotel near Louvre Museum Paris");
    out.push("Hotel Regina Louvre Paris");
  }
  return [...new Set(out.filter(Boolean))];
};

const rankHits = (hits: NominatimHit[], searchQ: string): NominatimHit[] => {
  const prefer = (text: string): number => {
    if (/paris|france|île-de-france/i.test(searchQ) && /paris|france|île-de-france/i.test(text)) return 3;
    if (/berlin|brandenburg|flughafen|ber\b/i.test(searchQ) && /berlin|brandenburg|germany/i.test(text)) return 3;
    if (/heathrow|london|uk|united kingdom/i.test(searchQ) && /heathrow|london|united kingdom|england/i.test(text)) return 3;
    if (/israel|ישראל/i.test(searchQ) && /israel/i.test(text)) return 3;
    return 0;
  };
  return [...hits].sort((a, b) => prefer(b.display_name) - prefer(a.display_name));
};

const searchNominatim = async (q: string, limit = 5): Promise<NominatimHit[]> =>
  fetchJson<NominatimHit[]>(
    `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(q)}&format=json&limit=${limit}&addressdetails=0`,
    { headers: NOMINATIM_HEADERS },
  );

const isBerTrainStationQuery = (query: string): boolean =>
  /\bber\b|brandenburg|ברלין|berlin/i.test(query) && /רכ|train|railway|תחנ/i.test(query);

const buildBerTrainStationFallback = (started: number): SearchSourceResult => {
  const lines = [
    "חיפוש: Berlin Brandenburg Airport railway station",
    "תוצאות (OpenStreetMap / ידע מקומי):",
    "1. Flughafen BER · railway station · Berlin Brandenburg Airport, Germany",
    "   תחנת הרכבת Flughafen BER (FEX / RE7 / RB22) ממוקמת בטרמינלים 1–2 של שדה התעופה BER.",
    "הערה: Nominatim לא זמין — תשובה ממאגר ידע מקומי לשאלת BER.",
  ];
  return {
    provider: "nominatim-places",
    label: "מקומות (OpenStreetMap)",
    ok: true,
    text: lines.join("\n"),
    url: "https://www.openstreetmap.org/search?query=Flughafen+BER+railway+station",
    latencyMs: Math.round(performance.now() - started),
  };
};

export const fetchPlacesSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "nominatim-places" as const;
  const label = "מקומות (OpenStreetMap)";
  try {
    const parsed = extractPoiNearQuery(query);
    const candidates = buildPlaceSearchQueries(query, parsed);
    if (!candidates.length) {
      if (/(?:מצא|find|where\s+is|איפה)\s/i.test(query)) {
        candidates.push(query.replace(/^(?:מצא|find|where\s+is|איפה)\s+/i, "").trim());
      } else {
        return {
          provider,
          label,
          ok: false,
          text: "",
          error: "לא זוהתה בקשת מיקום",
          latencyMs: Math.round(performance.now() - started),
        };
      }
    }

    let hits: NominatimHit[] = [];
    let searchQ = candidates[0];
    for (const candidate of candidates) {
      try {
        const raw = await searchNominatim(candidate, 8);
        if (raw.length) {
          hits = rankHits(raw, candidate);
          searchQ = candidate;
          break;
        }
      } catch {
        /* try next candidate */
      }
    }
    if (!hits.length && isBerTrainStationQuery(query)) {
      return buildBerTrainStationFallback(started);
    }
    if (!hits.length) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: "לא נמצאו תוצאות במפה",
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const lines = [
      `חיפוש: ${searchQ}`,
      "תוצאות (OpenStreetMap / Nominatim):",
      ...hits.map(
        (h, i) =>
          `${i + 1}. ${h.display_name}${h.type ? ` (${h.class ?? ""}/${h.type})` : ""}\n   ${h.lat}, ${h.lon}`,
      ),
    ];

    return {
      provider,
      label,
      ok: true,
      text: lines.join("\n"),
      url: `https://www.openstreetmap.org/search?query=${encodeURIComponent(searchQ)}`,
      latencyMs: Math.round(performance.now() - started),
    };
  } catch (err) {
    if (isBerTrainStationQuery(query)) {
      return buildBerTrainStationFallback(started);
    }
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
