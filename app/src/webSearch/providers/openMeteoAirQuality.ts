import { fetchJson } from "../fetchJson";
import type { SearchSourceResult } from "../types";
import { extractLocationPhrase } from "../queryExtract";
import { geocodePlace, formatPlaceLabel, type GeoPlace } from "../geoResolve";
import { getStartupContextSync } from "../../startupContext";

type AirQualityResponse = {
  current?: {
    time?: string;
    us_aqi?: number;
    pm2_5?: number;
    pm10?: number;
    nitrogen_dioxide?: number;
    ozone?: number;
  };
};

const aqiLabelHe = (aqi: number): string => {
  if (aqi <= 50) return "טוב";
  if (aqi <= 100) return "בינוני";
  if (aqi <= 150) return "לא בריא לקבוצות רגישות";
  if (aqi <= 200) return "לא בריא";
  if (aqi <= 300) return "מזיק מאוד";
  return "מסוכן";
};

export const fetchAirQualitySearch = async (
  query: string,
  sharedPlace?: GeoPlace | null,
): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "open-meteo-air-quality" as const;
  const label = "איכות אוויר (Open-Meteo)";

  try {
    let location = extractLocationPhrase(query);
    if (!location || location.length < 2) {
      location = query
        .replace(/(?:איכות\s+(?:ה)?אוויר|air\s+quality|pm2\.?5|pm10|\baqi\b|זיהום\s+אוויר)/gi, " ")
        .trim();
    }
    if (!location || location.length < 2) {
      const ctx = getStartupContextSync();
      if (ctx?.cityName) location = ctx.cityName;
    }
    if (!location || location.length < 2) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: "לא זוהה מיקום בשאלה",
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const place = sharedPlace ?? (await geocodePlace(location));
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

    const url =
      `https://air-quality-api.open-meteo.com/v1/air-quality?latitude=${place.latitude}&longitude=${place.longitude}` +
      `&current=us_aqi,pm2_5,pm10,nitrogen_dioxide,ozone&timezone=auto`;

    const data = await fetchJson<AirQualityResponse>(url, undefined, { timeoutMs: 12_000 });
    const cur = data.current;
    if (cur?.us_aqi == null && cur?.pm2_5 == null) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: "אין נתוני איכות אוויר למיקום",
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const placeLabel = formatPlaceLabel(place);
    const aqi = cur?.us_aqi ?? 0;
    const lines = [
      `מיקום: ${placeLabel}`,
      `זמן: ${cur?.time ?? "—"}`,
      `US AQI: ${aqi} (${aqiLabelHe(aqi)})`,
      ...(cur?.pm2_5 != null ? [`PM2.5: ${cur.pm2_5} µg/m³`] : []),
      ...(cur?.pm10 != null ? [`PM10: ${cur.pm10} µg/m³`] : []),
      ...(cur?.nitrogen_dioxide != null ? [`NO₂: ${cur.nitrogen_dioxide} µg/m³`] : []),
      ...(cur?.ozone != null ? [`O₃: ${cur.ozone} µg/m³`] : []),
      `ANSWER (air quality): US AQI ${aqi} · PM2.5 ${cur?.pm2_5 ?? "—"} · ${placeLabel}`,
    ];

    return {
      provider,
      label,
      ok: true,
      text: lines.join("\n"),
      url: `https://open-meteo.com/en/docs/air-quality-api`,
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
