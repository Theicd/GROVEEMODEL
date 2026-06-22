import { fetchJson } from "../fetchJson";
import { geocodePlace, formatPlaceLabel } from "../geoResolve";
import { extractPlacePair } from "../queryExtract";
import type { SearchSourceResult } from "../types";

type OsrmRoute = {
  routes?: Array<{
    distance: number;
    duration: number;
    geometry?: { coordinates?: [number, number][] };
  }>;
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
      `https://router.project-osrm.org/route/v1/driving/${coords}?overview=full&geometries=geojson`,
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

    const routePoints =
      leg.geometry?.coordinates?.map(([lon, lat]) => ({ lat, lon })) ??
      [
        { lat: a.latitude, lon: a.longitude },
        { lat: b.latitude, lon: b.longitude },
      ];

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
      url: `https://www.openstreetmap.org/directions?engine=fossgis_osrm_car&route=${a.latitude}%2C${a.longitude}%3B${b.latitude}%2C${b.longitude}`,
      geo: {
        from: { lat: a.latitude, lon: a.longitude, label: formatPlaceLabel(a) },
        to: { lat: b.latitude, lon: b.longitude, label: formatPlaceLabel(b) },
        route: routePoints,
        label: `${formatPlaceLabel(a)} → ${formatPlaceLabel(b)}`,
      },
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
