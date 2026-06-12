import { fetchJson } from "../fetchJson";
import { geocodePlace, formatPlaceLabel } from "../geoResolve";
import { extractPlacePair } from "../queryExtract";
import type { SearchSourceResult } from "../types";

type OsrmRoute = {
  routes?: Array<{ distance: number; duration: number }>;
  code?: string;
};

export const fetchDistanceSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "osrm-distance" as const;
  const label = "מרחק (OSRM)";
  try {
    const pair = extractPlacePair(query);
    if (!pair) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: "לא זוהו שני מקומות למרחק",
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const [aName, bName] = pair;
    const [a, b] = await Promise.all([geocodePlace(aName), geocodePlace(bName)]);
    if (!a || !b) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: `לא נמצאו קואורדינטות: ${!a ? aName : bName}`,
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const coords = `${a.longitude},${a.latitude};${b.longitude},${b.latitude}`;
    const route = await fetchJson<OsrmRoute>(
      `https://router.project-osrm.org/route/v1/driving/${coords}?overview=false`,
    );
    const leg = route.routes?.[0];
    if (!leg) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: "לא נמצא מסלול",
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const km = (leg.distance / 1000).toFixed(1);
    const hours = Math.floor(leg.duration / 3600);
    const mins = Math.round((leg.duration % 3600) / 60);
    const driveTime = hours > 0 ? `${hours} שע' ${mins} דק'` : `${mins} דק'`;

    const lines = [
      `מ: ${formatPlaceLabel(a)} (${aName})`,
      `אל: ${formatPlaceLabel(b)} (${bName})`,
      `מרחק נסיעה (OSRM): ${km} ק"מ`,
      `זמן נסיעה משוער: ${driveTime}`,
    ];

    return {
      provider,
      label,
      ok: true,
      text: lines.join("\n"),
      url: `https://router.project-osrm.org/route/v1/driving/${coords}`,
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
