import { pingGlobeForLiveSnapshot, waitForGlobeSnapshotUpdate } from "../../liveWorld/bridge";
import {
  issSearchResultFromLiveWorld,
  LIVE_WORLD_ISS_MAX_AGE_MS,
} from "../../liveWorld/issSnapshot";
import { fallbackFromLiveWorldSnapshot } from "../../liveWorld/snapshotFallback";
import { fetchJson } from "../../webSearch/fetchJson";
import type { SearchSourceResult } from "../../webSearch/types";

type IssPos = {
  latitude: number;
  longitude: number;
  altitude: number;
  velocity: number;
  timestamp: number;
};

type OpenNotifyIss = {
  iss_position?: { latitude: string; longitude: string };
};

type IssPass = {
  passes?: Array<{ risetime: number; duration: number }>;
};

const ISS_TIMEOUT_MS = 10_000;

const ISRAEL_PASS_RE =
  /(?:תעבור|יעבור|pass(?:es)?).*(?:מעל|above)\s+(?:ישראל|israel)|(?:מעל|above)\s+(?:ישראל|israel)|מתי.*(?:מעל|above)/i;

const formatIssLines = (lat: number, lon: number, altKm: number, velocityKmh?: number, query = ""): string[] => {
  const lines = [
    "מיקום ISS (זמן אמת):",
    `קו רוחב: ${lat.toFixed(2)}°`,
    `קו אורך: ${lon.toFixed(2)}°`,
    `גובה: ${altKm.toFixed(0)} km`,
    ...(velocityKmh != null ? [`מהירות: ${velocityKmh.toFixed(0)} km/h`] : []),
    `ANSWER (ISS position): ${lat.toFixed(2)}°N, ${lon.toFixed(2)}°E · ${altKm.toFixed(0)} km`,
    `עודכן: ${new Date().toISOString().replace("T", " ").slice(0, 19)} UTC`,
  ];
  if (/יבשה|ים|land|sea|ocean/i.test(query)) {
    const overPacific = (lon >= 140 || lon <= -70) && Math.abs(lat) < 55;
    const overAtlantic = lon >= -70 && lon <= 20 && Math.abs(lat) < 55;
    const estimate = overPacific || overAtlantic ? "לרוב מעל ים (הערכה)" : "ייתכן מעל יבשה או ים — «הצג על הגלובוס»";
    lines.push(`ANSWER (land/sea): ${estimate}`);
  }
  return lines;
};

const formatPassLines = (passes: IssPass["passes"]): string[] => {
  const next = passes?.[0];
  if (!next) {
    return ["מעבר ISS מעל ישראל:", "לא נמצא מעבר קרוב ב-10 הימים הקרובים (לפי wheretheiss.at)."];
  }
  const when = new Date(next.risetime * 1000);
  const heTime = when.toLocaleString("he-IL", { timeZone: "Asia/Jerusalem" });
  return [
    "מעבר ISS מעל ישראל (מרכז — תל אביב):",
    `המעבר הבא: ${heTime} (שעון ישראל)`,
    `משך נראות משוער: ${Math.round(next.duration / 60)} דקות`,
    "הערה: זמנים משוערים — «הצג על הגלובוס» לצפייה במסלול.",
  ];
};

const fetchIssPosition = async (): Promise<{ lat: number; lon: number; alt: number; vel?: number; url: string }> => {
  const sources: Array<{ url: string; run: () => Promise<{ lat: number; lon: number; alt: number; vel?: number; url: string }> }> = [
    {
      url: "https://api.wheretheiss.at/v1/satellites/25544",
      run: () =>
        fetchJson<IssPos>("https://api.wheretheiss.at/v1/satellites/25544", undefined, {
          timeoutMs: ISS_TIMEOUT_MS,
        }).then((data) => ({
          lat: data.latitude,
          lon: data.longitude,
          alt: data.altitude,
          vel: data.velocity,
          url: "https://api.wheretheiss.at",
        })),
    },
    {
      url: "https://api.open-notify.org/iss-now.json",
      run: () =>
        fetchJson<OpenNotifyIss>("https://api.open-notify.org/iss-now.json", undefined, {
          timeoutMs: ISS_TIMEOUT_MS,
        }).then((data) => {
          const pos = data.iss_position;
          if (!pos?.latitude || !pos?.longitude) throw new Error("no position");
          const lat = parseFloat(pos.latitude);
          const lon = parseFloat(pos.longitude);
          if (!Number.isFinite(lat) || !Number.isFinite(lon)) throw new Error("invalid position");
          return { lat, lon, alt: 408, url: "https://api.open-notify.org" };
        }),
    },
  ];

  const errors: unknown[] = [];
  for (const src of sources) {
    try {
      return await src.run();
    } catch (err) {
      errors.push(err);
    }
  }
  throw errors[0] instanceof Error ? errors[0] : new Error("ISS fetch failed");
};

const tryLiveWorldIss = (query: string): SearchSourceResult | null =>
  issSearchResultFromLiveWorld(query, LIVE_WORLD_ISS_MAX_AGE_MS);

export const fetchIssSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "iss-tracker" as const;
  const label = "תחנת חלל (ISS / עולם חי)";

  const cached = tryLiveWorldIss(query);
  if (cached) {
    return { ...cached, latencyMs: Math.round(performance.now() - started) };
  }

  pingGlobeForLiveSnapshot();

  if (ISRAEL_PASS_RE.test(query)) {
    try {
      const passData = await fetchJson<IssPass>(
        "https://api.wheretheiss.at/v1/satellites/25544/pass/32.0853/34.7818",
        undefined,
        { timeoutMs: ISS_TIMEOUT_MS },
      );
      return {
        provider,
        label,
        ok: true,
        text: formatPassLines(passData.passes).join("\n"),
        url: "https://api.wheretheiss.at",
        latencyMs: Math.round(performance.now() - started),
      };
    } catch {
      /* fall through */
    }
  }

  try {
    const pos = await fetchIssPosition();
    return {
      provider,
      label,
      ok: true,
      text: formatIssLines(pos.lat, pos.lon, pos.alt, pos.vel, query).join("\n"),
      url: pos.url,
      latencyMs: Math.round(performance.now() - started),
    };
  } catch {
    await waitForGlobeSnapshotUpdate(2200);
    const afterGlobe = tryLiveWorldIss(query) ?? fallbackFromLiveWorldSnapshot(query, ["satellite"]);
    if (afterGlobe) {
      return {
        ...afterGlobe,
        latencyMs: Math.round(performance.now() - started),
      };
    }
    return {
      provider,
      label,
      ok: false,
      text: "",
      error: "ISS timeout — פתח REALITY LIVE (עולם חי) לטעינת cache, או רענן",
      latencyMs: Math.round(performance.now() - started),
    };
  }
};
