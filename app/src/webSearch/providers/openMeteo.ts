import { fetchJson } from "../fetchJson";
import type { SearchSourceResult } from "../types";
import { extractLocationPhrase } from "../queryExtract";
import { geocodePlace, formatPlaceLabel } from "../geoResolve";

const WMO_HE: Record<number, string> = {
  0: "שמיים בהירים",
  1: "בהיר ברובו",
  2: "מעונן חלקית",
  3: "מעונן",
  45: "ערפל",
  48: "ערפל קפוא",
  51: "טפטוף קל",
  53: "טפטוף",
  55: "טפטוף כבד",
  61: "גשם קל",
  63: "גשם",
  65: "גשם כבד",
  71: "שלג קל",
  73: "שלג",
  75: "שלג כבד",
  80: "ממטרים",
  95: "סופת רעמים",
};

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
    wind_speed_10m_max?: number[];
  };
};

export const fetchWeatherSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "open-meteo" as const;
  const label = "מזג אוויר (Open-Meteo)";
  try {
    const location = extractLocationPhrase(query) ?? query.trim();
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

    const { latitude, longitude } = place;
    const forecast = await fetchJson<ForecastResult>(
      `https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}` +
        `&current=temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m,wind_direction_10m,surface_pressure` +
        `&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max` +
        `&timezone=auto&forecast_days=3`,
    );

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
      lines.push("תחזית 3 ימים:");
      for (let i = 0; i < Math.min(3, daily.time.length); i++) {
        lines.push(
          `- ${daily.time[i]}: ${daily.temperature_2m_min?.[i] ?? "?"}–${daily.temperature_2m_max?.[i] ?? "?"}°C, ` +
            `גשם ${daily.precipitation_sum?.[i] ?? 0} mm, רוח עד ${daily.wind_speed_10m_max?.[i] ?? "?"} km/h`,
        );
      }
    }

    const url = `https://open-meteo.com/en/docs#latitude=${latitude}&longitude=${longitude}`;
    return {
      provider,
      label,
      ok: true,
      text: lines.join("\n"),
      url,
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
