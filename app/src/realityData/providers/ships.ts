import { fetchJson } from "../../webSearch/fetchJson";
import type { SearchSourceResult } from "../../webSearch/types";

type AisFeature = {
  properties?: { mmsi?: number; sog?: number; cog?: number; navStat?: number; timestampExternal?: string };
  geometry?: { coordinates?: [number, number] };
};

type AisGeo = { features?: AisFeature[] };

type ShipHit = {
  name: string;
  lat: number;
  lon: number;
  speed?: number;
  destination?: string;
  source: "ais" | "route-marker";
};

const SUEZ_BBOX = { minLat: 29.8, maxLat: 31.55, minLon: 32.15, maxLon: 33.05 };
const MED_BBOX = { minLat: 27, maxLat: 42, minLon: 18, maxLon: 38 };

const SUEZ_MARKERS: ShipHit[] = [
  { name: "Suez Transit (מסלול)", lat: 31.25, lon: 32.31, destination: "EGPSD", source: "route-marker" },
  { name: "Suez South (מסלול)", lat: 30.0, lon: 32.58, destination: "EGPSD", source: "route-marker" },
];

const inBbox = (lat: number, lon: number, b: typeof SUEZ_BBOX) =>
  lat >= b.minLat && lat <= b.maxLat && lon >= b.minLon && lon <= b.maxLon;

const detectRegion = (query: string): "suez" | "med" | "global" => {
  if (/סואץ|suez|canal/i.test(query)) return "suez";
  if (/ים\s+תיכון|mediterranean|haifa|חיפה|אשדוד|eilat|אילat/i.test(query)) return "med";
  return "global";
};

/** Live AIS ships — Digitraffic (same feed as Live World) + route markers near Suez. */
export const fetchShipsSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "ais-ships" as const;
  const label = "ספינות (AIS / עולם חי)";
  const region = detectRegion(query);

  try {
    const geo = await fetchJson<AisGeo>("https://meri.digitraffic.fi/api/ais/v1/locations", undefined, {
      timeoutMs: 14_000,
    });
    const features = geo.features ?? [];
    const bbox = region === "suez" ? SUEZ_BBOX : region === "med" ? MED_BBOX : null;

    const aisHits: ShipHit[] = [];
    for (const f of features) {
      const c = f.geometry?.coordinates;
      const p = f.properties;
      if (!c?.length || c[1] == null || c[0] == null) continue;
      const lat = c[1];
      const lon = c[0];
      if (bbox && !inBbox(lat, lon, bbox)) continue;
      aisHits.push({
        name: `MMSI ${p?.mmsi ?? "?"}`,
        lat,
        lon,
        speed: p?.sog != null ? p.sog / 10 : undefined,
        destination: "",
        source: "ais",
      });
    }

    let hits = aisHits;
    if (region === "suez") {
      hits = [...aisHits, ...SUEZ_MARKERS];
    }

    if (!hits.length) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error:
          region === "suez"
            ? "אין כרגע דיווחי AIS חיים בתעלת סואץ — Digitraffic מכסה בעיקר צפון אירופה; ב«עולם חי» יש גם סימוני מסלול"
            : "לא נמצאו ספינות בטווח",
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const regionLabel =
      region === "suez" ? "תעלת סואץ" : region === "med" ? "ים תיכון" : "גלובלי (Digitraffic)";
    const lines = [
      `אזור: ${regionLabel}`,
      `ספינות בטווח: ${hits.length} (${aisHits.length} AIS חי + ${hits.length - aisHits.length} סימוני מסלול)`,
      "הערה: שכבת «ספינות» בעולם החי (🌐) מציגה את אותם נתונים על המפה — «הצג על המפה» לצפייה.",
      ...hits.slice(0, 10).map((s, i) => {
        const spd = s.speed != null ? `${s.speed.toFixed(1)} kn` : "—";
        const tag = s.source === "route-marker" ? "מסלול" : "AIS";
        return `${i + 1}. ${s.name} · ${tag} · ${s.lat.toFixed(2)},${s.lon.toFixed(2)} · ${spd}${s.destination ? ` → ${s.destination}` : ""}`;
      }),
    ];

    return {
      provider,
      label,
      ok: true,
      text: lines.join("\n"),
      url: "https://meri.digitraffic.fi/en/web/ais/vessels",
      latencyMs: Math.round(performance.now() - started),
    };
  } catch (err) {
    if (region === "suez") {
      const lines = [
        `אזור: תעלת סואץ`,
        `סימוני מסלול (עולם חי): ${SUEZ_MARKERS.length}`,
        ...SUEZ_MARKERS.map((s, i) => `${i + 1}. ${s.name} · ${s.lat.toFixed(2)},${s.lon.toFixed(2)} → ${s.destination}`),
        "הערה: AIS חי בתעלת סואץ דליל ב-API החינמי; לצפייה מלאה פתח «עולם חי» → שכבת ספינות.",
      ];
      return {
        provider,
        label,
        ok: true,
        text: lines.join("\n"),
        url: "https://meri.digitraffic.fi/en/web/ais/vessels",
        latencyMs: Math.round(performance.now() - started),
      };
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
