import { fetchJson } from "../fetchJson";
import type { SearchSourceResult } from "../types";
import { extractLocationPhrase, normalizePlaceSearchName } from "../queryExtract";
import { geocodePlace, formatPlaceLabel, type GeoPlace } from "../geoResolve";
import { getStartupContextSync } from "../../startupContext";
import { buildWeatherWidgetFromForecast } from "../../weatherWidget/buildWeatherWidget";
import { isWeatherForecastQuery } from "../../weatherWidget/inChatWeather";
import { WMO_HE } from "../../weatherWidget/wmoLabels";

const stripWeatherNoise = (raw: string, keepWeekly = false): string =>
  raw
    .replace(
      new RegExp(
        `(?:weather|forecast|temperature|temperatur|מזג\\s*האוויר|תחזית|טמפרטור(?:ה)?|tomorrow|מחר|today|היום|now|עכשיו|כרגע${keepWeekly ? "" : "|שבוע|weekly|שבועי"})`,
        "gi",
      ),
      " ",
    )
    .replace(/\s{2,}/g, " ")
    .trim();

type ForecastResult = {
  current?: {
    time?: string;
    temperature_2m?: number;
    apparent_temperature?: number;
    relative_humidity_2m?: number;
    weather_code?: number;
    wind_speed_10m?: number;
    wind_direction_10m?: number;
    surface_pressure?: number;
  };
  daily?: {
    time?: string[];
    temperature_2m_max?: number[];
    temperature_2m_min?: number[];
    precipitation_sum?: number[];
    precipitation_probability_max?: number[];
    weather_code?: number[];
    wind_speed_10m_max?: number[];
  };
};

export const fetchWeatherSearch = async (
  query: string,
  sharedPlace?: GeoPlace | null,
): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "open-meteo" as const;
  const label = "מזג אוויר (Open-Meteo)";
  try {
    const wantsTomorrow = /(?:tomorrow|מחר)/i.test(query);
    const wantsWeekly = /(?:שבוע|weekly|7\s*days?|שבועי)/i.test(query);
    const includeForecastInWidget = isWeatherForecastQuery(query);
    let location = extractLocationPhrase(query);
    if (!location || location.length < 2) {
      location = stripWeatherNoise(query, wantsWeekly);
    }
    if ((!location || location.length < 2) && wantsWeekly) {
      const ctx = getStartupContextSync();
      if (ctx?.cityName) location = ctx.cityName;
      else if (ctx?.countryName) location = ctx.countryName;
    }
    if (!location || location.length < 2) {
      const ctx = getStartupContextSync();
      if (ctx?.cityName) location = ctx.cityName;
      else if (ctx?.countryName) location = ctx.countryName;
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

    location = normalizePlaceSearchName(location);

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

    const { latitude, longitude } = place;
    const forecastDays = wantsWeekly ? 7 : wantsTomorrow ? 2 : includeForecastInWidget ? 3 : 1;
    const forecastUrl =
      `https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}` +
      `&current=temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m,wind_direction_10m,surface_pressure` +
      `&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,weather_code,wind_speed_10m_max` +
      `&timezone=auto&forecast_days=${forecastDays}`;

    let forecast: ForecastResult;
    try {
      forecast = await fetchJson<ForecastResult>(forecastUrl, undefined, { timeoutMs: 12_000 });
    } catch {
      forecast = await fetchJson<ForecastResult>(forecastUrl, undefined, { timeoutMs: 18_000 });
    }

    const cur = forecast.current;
    const daily = forecast.daily;
    if (cur?.temperature_2m == null && !(daily?.time?.length ?? 0)) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: "אין נתוני מזג אוויר עדכניים למיקום",
        latencyMs: Math.round(performance.now() - started),
      };
    }
    const placeLabel = formatPlaceLabel(place);
    const desc =
      cur?.weather_code != null ? (WMO_HE[cur.weather_code] ?? `קוד ${cur.weather_code}`) : "—";

    const lines = [
      `מיקום: ${placeLabel}`,
      ...(place.elevation != null ? [`גובה: ${Math.round(place.elevation)} m`] : []),
      `זמן (מקומי): ${cur?.time ?? "—"}`,
      `מצב: ${desc}`,
      `טמפרטורה: ${cur?.temperature_2m ?? "—"}°C (מרגיש ${cur?.apparent_temperature ?? "—"}°C)`,
      `לחות: ${cur?.relative_humidity_2m ?? "—"}%`,
      `רוח: ${cur?.wind_speed_10m ?? "—"} km/h, כיוון ${cur?.wind_direction_10m ?? "—"}°`,
      `לחץ: ${cur?.surface_pressure ?? "—"} hPa`,
    ];

    if (daily?.time?.length) {
      const wantsRain = /(?:גשם|rain|precipitation|ממטר)/i.test(query);
      const forecastLabel = wantsTomorrow
        ? "תחזית למחר"
        : wantsWeekly
          ? "תחזית שבועית"
          : wantsRain
            ? "תחזית גשם (3 ימים)"
            : "תחזית 3 ימים";
      lines.push(`${forecastLabel}:`);
      const startIdx = wantsTomorrow && daily.time.length > 1 ? 1 : 0;
      const count = wantsTomorrow ? 1 : wantsWeekly ? Math.min(7, daily.time.length) : Math.min(3, daily.time.length);
      for (let i = startIdx; i < startIdx + count && i < daily.time.length; i++) {
        const dayTag = wantsTomorrow ? "מחר" : daily.time[i];
        const rainMm = daily.precipitation_sum?.[i] ?? 0;
        const rainProb = daily.precipitation_probability_max?.[i];
        const rainCode = daily.weather_code?.[i];
        const rainDesc = rainCode != null ? (WMO_HE[rainCode] ?? "") : "";
        lines.push(
          `- ${dayTag} (${daily.time[i]}): ${daily.temperature_2m_min?.[i] ?? "?"}–${daily.temperature_2m_max?.[i] ?? "?"}°C, ` +
            `גשם ${rainMm} mm${rainProb != null ? ` (סיכוי ${rainProb}%)` : ""}${rainDesc ? ` · ${rainDesc}` : ""}, ` +
            `רוח עד ${daily.wind_speed_10m_max?.[i] ?? "?"} km/h`,
        );
      }
    }

    const url = `https://open-meteo.com/en/docs#latitude=${latitude}&longitude=${longitude}`;
    const weatherWidget = buildWeatherWidgetFromForecast({
      place,
      placeLabel,
      current: cur,
      daily,
      wantsTomorrow,
      wantsWeekly,
      includeForecast: includeForecastInWidget,
      sourceLabel: label,
    });
    return {
      provider,
      label,
      ok: true,
      text: lines.join("\n"),
      url,
      weatherWidget: weatherWidget ?? undefined,
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
