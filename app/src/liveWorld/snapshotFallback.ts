import type { SearchIntent } from "../webSearch/types";
import { getCachedLiveWorldSnapshot } from "./snapshotStore";
import type { LiveWorldSnapshot, SnapshotSearchFallback } from "./types";
import { buildMilitaryAviationText, isMilitaryAviationQuery } from "./militaryAviation";
import { isLiveShipSource } from "../realityData/shipAggregate";
import { formatIssSnapshotText, LIVE_WORLD_ISS_MAX_AGE_MS } from "./issSnapshot";

const ISRAEL_EQ =
  /ישראל|israel|dead\s+sea|sinai|ירדן|jordan|lebanon|לבנון|syria|סוריה|mediterranean|ים\s+תיכון/i;

const strongestEarthquakeText = (snap: LiveWorldSnapshot, query: string): string | null => {
  const items = snap.earthquake?.items;
  if (!items?.length) return null;
  let filtered = [...items].sort((a, b) => (b.magnitude ?? 0) - (a.magnitude ?? 0));
  if (ISRAEL_EQ.test(query)) {
    filtered = filtered.filter((e) => ISRAEL_EQ.test(e.place));
  }
  if (!filtered.length) {
    return `אין רעידות אדמה מדווחות (USGS / עולם חי) באזור המבוקש ב-24 השעות האחרונות.`;
  }
  const top = filtered[0];
  const mag = top.magnitude != null ? top.magnitude.toFixed(1) : "?";
  const when = new Date(top.time).toISOString().replace("T", " ").slice(0, 19);
  const lines = [
    `סה"כ ${filtered.length} רעידות רלוונטיות (מקור: ${snap.earthquake?.feedLabel ?? "USGS"}).`,
    `החזקה ביותר: M${mag} · ${top.place} · ${when} UTC`,
    ...filtered.slice(0, 6).map((e, i) => {
      const m = e.magnitude != null ? e.magnitude.toFixed(1) : "?";
      return `${i + 1}. M${m} · ${e.place}`;
    }),
  ];
  return lines.join("\n");
};

const issText = (snap: LiveWorldSnapshot): string | null => formatIssSnapshotText(snap);

const HAIFA_BAY = { minLat: 32.72, maxLat: 32.92, minLon: 34.92, maxLon: 35.12 };
const SUEZ_BBOX = { minLat: 29.8, maxLat: 31.55, minLon: 32.15, maxLon: 33.05 };

const inBbox = (lat: number, lon: number, b: typeof HAIFA_BAY) =>
  lat >= b.minLat && lat <= b.maxLat && lon >= b.minLon && lon <= b.maxLon;

const shipsText = (snap: LiveWorldSnapshot, query: string): string | null => {
  const ships = snap.ships;
  if (!ships?.items.length && !ships?.count) return null;

  const wantHaifa = /מפרץ\s*חיפה|חיפה|haifa/i.test(query);
  const wantSuez = /סואץ|suez|תעלת/i.test(query);
  let hits = ships.items.filter((s) => {
    if (!isLiveShipSource(s.source)) return false;
    if (wantHaifa) return inBbox(s.lat, s.lon, HAIFA_BAY);
    if (wantSuez) return inBbox(s.lat, s.lon, SUEZ_BBOX);
    return true;
  });
  let regionLabel = ships.regionLabel;

  if (wantHaifa) {
    regionLabel = "מפרץ חיפה (bbox)";
    if (!hits.length) {
      return [
        `אזור: ${regionLabel}`,
        "ANSWER (ships live): 0",
        "דיווח AIS חי + עולם חי: 0",
        "הערה: Digitraffic מכסה בעיקר צפון אירופה; במפרץ חיפה ייתכן שאין AIS חי.",
      ].join("\n");
    }
  }

  if (wantSuez) {
    regionLabel = "תעלת סואץ";
  }

  const count = hits.length;
  return [
    `אזור: ${regionLabel}`,
    `ANSWER (ships live): ${count}`,
    `דיווח AIS חי + עולם חי: ${count}`,
    `עודכן: ${new Date(snap.fetchedAt).toISOString().replace("T", " ").slice(0, 19)} UTC`,
    ...hits.slice(0, 10).map((s, i) => {
      const spd = s.speedKn != null ? `${s.speedKn.toFixed(1)} kn` : "—";
      return `${i + 1}. ${s.name} · עולם חי · ${s.lat.toFixed(2)},${s.lon.toFixed(2)} · ${spd}${s.destination ? ` → ${s.destination}` : ""}`;
    }),
  ].join("\n");
};

const aviationText = (snap: LiveWorldSnapshot, query: string): string | null => {
  if (isMilitaryAviationQuery(query)) {
    const mil = buildMilitaryAviationText(query, snap);
    if (mil) return mil;
  }
  const av = snap.aviation;
  if (!av?.count) return null;
  return [
    `אזור: ${av.regionLabel}`,
    `מטוסים בטווח: ${av.count}`,
    ...(av.militaryCount != null ? [`מטוסים צבאיים (heuristic): ${av.militaryCount}`] : []),
    ...av.sample.map((s, i) => `${i + 1}. ${s}`),
  ].join("\n");
};

/** Build SearchSourceResult from cached snapshot when live fetch failed. */
export function fallbackFromLiveWorldSnapshot(
  query: string,
  intents: SearchIntent[],
): SnapshotSearchFallback | null {
  if (
    intents.includes("satellite") &&
    /\biss\b|תחנת\s+(?:ה)?חלל|space\s+station|החלל\s+הבינלאומ/i.test(query)
  ) {
    const issSnap = getCachedLiveWorldSnapshot(LIVE_WORLD_ISS_MAX_AGE_MS);
    const text = issSnap ? issText(issSnap) : null;
    if (text) {
      return {
        provider: "iss-tracker",
        label: "תחנת חלל (עולם חי / ISS)",
        ok: true,
        text,
        url: "https://api.wheretheiss.at",
        latencyMs: 0,
      };
    }
  }

  const snap = getCachedLiveWorldSnapshot(120_000);
  if (!snap) return null;

  if (intents.includes("earthquake")) {
    const text = strongestEarthquakeText(snap, query);
    if (!text) return null;
    return {
      provider: "usgs-earthquake",
      label: "רעידות אדמה (עולם חי / USGS)",
      ok: true,
      text,
      url: "https://earthquake.usgs.gov",
      latencyMs: 0,
    };
  }

  if (intents.includes("ships")) {
    const text = shipsText(snap, query);
    if (!text) return null;
    return {
      provider: "ais-ships",
      label: "ספינות (עולם חי / AIS)",
      ok: true,
      text,
      url: "https://meri.digitraffic.fi",
      latencyMs: 0,
    };
  }

  if (intents.includes("aviation")) {
    const text = aviationText(snap, query);
    if (!text) return null;
    return {
      provider: "adsb-aviation",
      label: "תעופה (עולם חי / ADS-B)",
      ok: true,
      text,
      url: "https://api.airplanes.live",
      latencyMs: 0,
    };
  }

  return null;
}

export function applySnapshotFallbacks(
  query: string,
  intents: SearchIntent[],
  sources: SnapshotSearchFallback[],
): SnapshotSearchFallback[] {
  const liveIntents: SearchIntent[] = ["earthquake", "ships", "aviation", "satellite"];
  const needsFallback = liveIntents.some((i) => intents.includes(i));
  if (!needsFallback) return sources;

  const snap = getCachedLiveWorldSnapshot(120_000);
  if (!snap) return sources;

  const out = [...sources];
  const hasOk = (provider: string) => out.some((s) => s.provider === provider && s.ok && s.text.trim());

  if (intents.includes("earthquake") && !hasOk("usgs-earthquake")) {
    const fb = fallbackFromLiveWorldSnapshot(query, ["earthquake"]);
    if (fb) out.push(fb);
  }
  if (intents.includes("ships") && !hasOk("ais-ships")) {
    const fb = fallbackFromLiveWorldSnapshot(query, ["ships"]);
    if (fb) out.push(fb);
  }
  if (intents.includes("aviation") && !hasOk("adsb-aviation")) {
    const fb = fallbackFromLiveWorldSnapshot(query, ["aviation"]);
    if (fb) out.push(fb);
  }
  if (
    intents.includes("satellite") &&
    /\biss\b|תחנת\s+(?:ה)?חלל|space\s+station|החלל\s+הבינלאומ/i.test(query) &&
    !hasOk("iss-tracker")
  ) {
    const fb = fallbackFromLiveWorldSnapshot(query, ["satellite"]);
    if (fb) out.push(fb);
  }

  return out;
}
