import { fetchJson } from "../fetchJson";
import type { SearchSourceResult } from "../types";
import { extractLocationPhrase } from "../queryExtract";
import { geocodePlace, formatPlaceLabel } from "../geoResolve";

type MarineResult = {
  current?: {
    time?: string;
    wave_height?: number;
    wave_direction?: number;
    wave_period?: number;
    swell_wave_height?: number;
    ocean_current_velocity?: number;
  };
};

export const fetchMarineSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "open-meteo-marine" as const;
  const label = "ים וגלים (Open-Meteo Marine)";
  try {
    const location = extractLocationPhrase(query) ?? query.replace(/wave\s*height|גלים|marine|ocean/gi, "").trim();
    const place = await geocodePlace(location);
    if (!place) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: `לא נמצא מיקום: ${location}`,
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const marine = await fetchJson<MarineResult>(
      `https://marine-api.open-meteo.com/v1/marine?latitude=${place.latitude}&longitude=${place.longitude}` +
        `&current=wave_height,wave_direction,wave_period,swell_wave_height,ocean_current_velocity&timezone=auto`,
    );

    const c = marine.current;
    const lines = [
      `מיקום: ${formatPlaceLabel(place)}`,
      `זמן: ${c?.time ?? "—"}`,
      `גובה גל: ${c?.wave_height ?? "—"} m`,
      `כיוון גל: ${c?.wave_direction ?? "—"}°`,
      `מחזור גל: ${c?.wave_period ?? "—"} s`,
      `גובה swell: ${c?.swell_wave_height ?? "—"} m`,
      `זרם ים: ${c?.ocean_current_velocity ?? "—"} km/h`,
    ];

    return {
      provider,
      label,
      ok: true,
      text: lines.join("\n"),
      url: `https://open-meteo.com/en/docs/marine-weather-api`,
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
