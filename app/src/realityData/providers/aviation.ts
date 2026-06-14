import { fetchJson } from "../../webSearch/fetchJson";
import type { SearchSourceResult } from "../../webSearch/types";
import { classifyAircraft, isAwacsSuspect, isTankerSuspect } from "../aviationClassifier";
import { getCachedLiveWorldSnapshot } from "../../liveWorld/snapshotStore";
import { buildMilitaryAviationText } from "../../liveWorld/militaryAviation";

type AdsbResponse = {
  ac?: Array<{
    flight?: string;
    alt_baro?: number | string;
    gs?: number;
    track?: number;
    hex?: string;
    r?: string;
    t?: string;
    category?: string;
    dbFlags?: number;
    lat?: number;
    lon?: number;
  }>;
};

type OpenSkyResponse = { states?: Array<(string | number | null)[]> | null };

const ISRAEL_RE = /ישראל|israel|tel\s*aviv|תל\s*אביב|נתב"?ג|ben\s*gurion/i;
const LONDON_RE = /לונדון|london/i;
const MED_RE = /(?:ה)?ים\s*תיכון|mediterranean|med\s+sea/i;
const EUROPE_RE = /אירופה|europe/i;
const GLOBAL_RE = /בעולם|worldwide|global|ברחבי\s+העולם|around\s+the\s+world|in\s+the\s+air|כרגע/i;

type RegionPoint = { lat: number; lon: number; label: string };

const pickRegion = (context: string): RegionPoint => {
  if (ISRAEL_RE.test(context)) return { lat: 32.08, lon: 34.78, label: "ישראל (מרכז)" };
  if (LONDON_RE.test(context)) return { lat: 51.47, lon: -0.45, label: "לונדון" };
  if (MED_RE.test(context)) return { lat: 35.0, lon: 20.0, label: "ים תיכון" };
  if (EUROPE_RE.test(context)) return { lat: 50.0, lon: 10.0, label: "מרכז אירופה" };
  return { lat: 40.7, lon: -74.0, label: "גלובלי (OpenSky)" };
};

const isMilitaryAviationQuery = (query: string): boolean =>
  /\bawacs\b|צבאי|military|תדלוק|tanker|מודיעין/i.test(query);

const mapOpenSky = (states: OpenSkyResponse["states"]) =>
  (states ?? [])
    .filter((s) => s?.[5] != null && s?.[6] != null && !s[8])
    .map((s) => {
      const cls = classifyAircraft(String(s[0] ?? ""), String(s[1] ?? ""), String(s[2] ?? ""), s[17]);
      return {
        icao24: String(s[0] ?? ""),
        callsign: String(s[1] ?? "").trim(),
        country: String(s[2] ?? ""),
        alt: Number(s[13] ?? s[7] ?? 0),
        cls,
      };
    });

const mapAdsb = (ac: NonNullable<AdsbResponse["ac"]>) =>
  ac.map((a) => {
    const cls = classifyAircraft(a.hex, a.flight, a.r, a.t ?? a.category, a.dbFlags);
    return {
      icao24: a.hex ?? "",
      callsign: (a.flight ?? "").trim(),
      country: a.r ?? "",
      alt: typeof a.alt_baro === "number" ? a.alt_baro : 0,
      cls,
    };
  });

/** Live aircraft — airplanes.live (regional) or OpenSky (global count). */
export const fetchAviationSearch = async (
  query: string,
  recentUserText: string[] = [],
): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "adsb-aviation" as const;
  const label = "תעופה (ADS-B)";
  const context = [query, ...recentUserText].join(" ");

  const snap = getCachedLiveWorldSnapshot(120_000);
  if (snap?.aviation?.items?.length) {
    const text = buildMilitaryAviationText(query, snap);
    if (text) {
      return {
        provider,
        label: "תעופה (עולם חי / ADS-B)",
        ok: true,
        text,
        url: "https://api.airplanes.live",
        latencyMs: Math.round(performance.now() - started),
      };
    }
  }

  const wantGlobal = GLOBAL_RE.test(context) || isMilitaryAviationQuery(query);

  const buildRegionalResult = async (): Promise<SearchSourceResult> => {
    const region = pickRegion(context);
    const data = await fetchJson<AdsbResponse>(
      `https://api.airplanes.live/v2/point/${region.lat}/${region.lon}/250`,
    );
    const mapped = mapAdsb(data.ac ?? []);
    if (!mapped.length) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: "אין מטוסים בטווח",
        latencyMs: Math.round(performance.now() - started),
      };
    }
    const milCount = mapped.filter((a) => a.cls.mil).length;
    const lines = [
      `אזור: ${region.label} · רדיוס 250km`,
      `מטוסים בטווח: ${mapped.length}`,
      ...(milCount ? [`מטוסים צבאיים (heuristic): ${milCount}`] : []),
      ...mapped.slice(0, 12).map((a, i) => {
        const tag = a.cls.mil ? ` · ${a.cls.label || "צבאי"}` : "";
        return `${i + 1}. ${a.callsign || "לא ידוע"} · ${a.alt}ft${tag}`;
      }),
    ];
    if (/צבאי|military|מהם.*מטוס/i.test(query)) {
      lines.push("הערה: זיהוי צבאי heuristic — כמו שכבת תעופה בעולם חי.");
    }
    return {
      provider,
      label,
      ok: true,
      text: lines.join("\n"),
      url: "https://api.airplanes.live",
      latencyMs: Math.round(performance.now() - started),
    };
  };

  try {
    if (wantGlobal) {
      let mapped: ReturnType<typeof mapOpenSky> = [];
      try {
        const data = await fetchJson<OpenSkyResponse>("https://opensky-network.org/api/states/all", undefined, {
          timeoutMs: 18_000,
        });
        mapped = mapOpenSky(data.states);
      } catch {
        /* OpenSky often blocked on static hosts — fall through to cache/regional */
      }

      if (!mapped.length) {
        const cached = getCachedLiveWorldSnapshot(120_000);
        if (cached?.aviation?.items?.length) {
          const text = buildMilitaryAviationText(query, cached);
          if (text) {
            return {
              provider,
              label: "תעופה (עולם חי / ADS-B)",
              ok: true,
              text,
              url: "https://api.airplanes.live",
              latencyMs: Math.round(performance.now() - started),
            };
          }
        }
        return buildRegionalResult();
      }

      if (/\bawacs\b/i.test(query)) {
        const awacs = mapped.filter(
          (a) => a.cls.awacsSuspect || isAwacsSuspect(a.callsign, a.cls.label, undefined),
        );
        const lines = [
          "אזור: גלובלי (OpenSky ADS-B)",
          `מטוסים באוויר: ${mapped.length}`,
          `מועמדים ל-AWACS (heuristic): ${awacs.length}`,
          awacs.length
            ? `ANSWER (AWACS): ${awacs.length} · ${awacs[0].callsign || awacs[0].icao24} · ${awacs[0].cls.label || "heuristic"}`
            : "ANSWER (AWACS): 0 מטוסים מזוהים כ-AWACS (heuristic — לא כל AWACS משדר ADS-B).",
          ...awacs.slice(0, 8).map((a, i) => `${i + 1}. ${a.callsign || a.icao24} · ${a.cls.label || "AWACS?"}`),
          "הערה: זיהוי כמו עולם חי — ICAO hex / callsign / NATO. פתח עולם חי לצפייה על המפה.",
        ];
        return {
          provider,
          label,
          ok: true,
          text: lines.join("\n"),
          url: "https://opensky-network.org",
          latencyMs: Math.round(performance.now() - started),
        };
      }

      if (/צבאי|military/i.test(query)) {
        const mil = mapped.filter((a) => a.cls.mil);
        const lines = [
          "אזור: גלובלי (OpenSky ADS-B)",
          `מטוסים באוויר: ${mapped.length}`,
          `מטוסים צבאיים (heuristic): ${mil.length}`,
          ...mil.slice(0, 10).map((a, i) => `${i + 1}. ${a.callsign || a.icao24} · ${a.cls.label || "צבאי"}`),
          "הערה: ADS-B ציבורי — זיהוי heuristic כמו שכבת תעופה בעולם חי.",
        ];
        return {
          provider,
          label,
          ok: true,
          text: lines.join("\n"),
          url: "https://opensky-network.org",
          latencyMs: Math.round(performance.now() - started),
        };
      }

      if (/תדלוק|tanker/i.test(query)) {
        const tankers = mapped.filter((a) => isTankerSuspect(a.callsign, undefined));
        const lines = [
          "אזור: גלובלי (OpenSky ADS-B)",
          `מטוסים באוויר: ${mapped.length}`,
          `מועמדים לתדלוק (heuristic): ${tankers.length}`,
          ...tankers.slice(0, 8).map((a, i) => `${i + 1}. ${a.callsign || a.icao24}`),
        ];
        return {
          provider,
          label,
          ok: true,
          text: lines.join("\n"),
          url: "https://opensky-network.org",
          latencyMs: Math.round(performance.now() - started),
        };
      }

      const lines = [
        "אזור: גלובלי (OpenSky ADS-B)",
        `מטוסים באוויר (דיווח אחרון): ${mapped.length}`,
        `מטוסים צבאיים (heuristic): ${mapped.filter((a) => a.cls.mil).length}`,
        "הערה: עולם חי (🌐) מציג מטוסים על המפה — «הצג על המפה» לצפייה.",
        ...mapped.slice(0, 10).map((a, i) => {
          const tag = a.cls.mil ? " · צבאי" : "";
          return `${i + 1}. ${a.callsign || "—"} · ${a.alt}m${tag}`;
        }),
      ];
      return {
        provider,
        label,
        ok: true,
        text: lines.join("\n"),
        url: "https://opensky-network.org",
        latencyMs: Math.round(performance.now() - started),
      };
    }

    return buildRegionalResult();
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
