import type { GeoPlace } from "../webSearch/geoResolve";
import type { WeatherForecastDay, WeatherWidgetData } from "./types";
import { wmoIconKind, wmoLabel } from "./wmoLabels";

type ForecastCurrent = {
  time?: string;
  temperature_2m?: number;
  apparent_temperature?: number;
  relative_humidity_2m?: number;
  weather_code?: number;
  wind_speed_10m?: number;
  wind_direction_10m?: number;
  surface_pressure?: number;
};

type ForecastDaily = {
  time?: string[];
  temperature_2m_max?: number[];
  temperature_2m_min?: number[];
  precipitation_sum?: number[];
  precipitation_probability_max?: number[];
  weather_code?: number[];
  wind_speed_10m_max?: number[];
};

export type BuildWeatherWidgetInput = {
  place: GeoPlace;
  placeLabel: string;
  current?: ForecastCurrent;
  daily?: ForecastDaily;
  wantsTomorrow?: boolean;
  wantsWeekly?: boolean;
  /** When false, widget shows current conditions only (no multi-day forecast). */
  includeForecast?: boolean;
  sourceLabel: string;
};

function formatDayLabel(dateIso: string, tomorrowTag: boolean): string {
  if (tomorrowTag) return "מחר";
  try {
    const d = new Date(`${dateIso}T12:00:00`);
    return new Intl.DateTimeFormat("he-IL", { weekday: "short" }).format(d);
  } catch {
    return dateIso;
  }
}

function buildRegionLabel(place: GeoPlace): string {
  const parts = [place.admin1, place.country_code].filter(Boolean);
  return parts.join(", ") || place.name;
}

function buildForecastDays(
  daily: ForecastDaily | undefined,
  wantsTomorrow: boolean,
  wantsWeekly: boolean,
): WeatherForecastDay[] {
  if (!daily?.time?.length) return [];

  const startIdx = wantsTomorrow && daily.time.length > 1 ? 1 : 0;
  const count = wantsTomorrow ? 1 : wantsWeekly ? Math.min(7, daily.time.length) : Math.min(3, daily.time.length);

  const slice: WeatherForecastDay[] = [];
  for (let i = startIdx; i < startIdx + count && i < daily.time.length; i++) {
    const minC = daily.temperature_2m_min?.[i];
    const maxC = daily.temperature_2m_max?.[i];
    if (minC == null && maxC == null) continue;
    const code = daily.weather_code?.[i];
    slice.push({
      dateIso: daily.time[i],
      dayLabel: formatDayLabel(daily.time[i], wantsTomorrow && i === startIdx),
      minC: minC ?? maxC ?? 0,
      maxC: maxC ?? minC ?? 0,
      precipMm: daily.precipitation_sum?.[i],
      precipChancePct: daily.precipitation_probability_max?.[i],
      condition: wmoLabel(code),
      iconKind: wmoIconKind(code),
      windMaxKmh: daily.wind_speed_10m_max?.[i],
      tempBarPct: 0,
    });
  }

  if (!slice.length) return slice;

  const globalMin = Math.min(...slice.map((d) => d.minC));
  const globalMax = Math.max(...slice.map((d) => d.maxC));
  const span = Math.max(globalMax - globalMin, 1);

  return slice.map((d) => ({
    ...d,
    tempBarPct: Math.round(((d.maxC - globalMin) / span) * 100),
  }));
}

export function buildWeatherWidgetFromForecast(input: BuildWeatherWidgetInput): WeatherWidgetData | null {
  const cur = input.current;
  const daily = input.daily;
  if (cur?.temperature_2m == null && !(daily?.time?.length ?? 0)) return null;

  const code = cur?.weather_code;
  const forecast = input.includeForecast
    ? buildForecastDays(daily, !!input.wantsTomorrow, !!input.wantsWeekly)
    : [];

  return {
    placeLabel: input.placeLabel,
    cityName: input.place.name,
    regionLabel: buildRegionLabel(input.place),
    observedAt: cur?.time,
    condition: wmoLabel(code),
    iconKind: wmoIconKind(code),
    temperatureC: cur?.temperature_2m ?? forecast[0]?.maxC ?? 0,
    feelsLikeC: cur?.apparent_temperature,
    humidityPct: cur?.relative_humidity_2m,
    windKmh: cur?.wind_speed_10m,
    windDirectionDeg: cur?.wind_direction_10m,
    pressureHpa: cur?.surface_pressure,
    forecast,
    sourceLabel: input.sourceLabel,
  };
}

export function buildShortWeatherReply(widget: WeatherWidgetData): string {
  const place = widget.cityName || widget.placeLabel;
  const temp =
    widget.temperatureC % 1 === 0
      ? String(Math.round(widget.temperatureC))
      : widget.temperatureC.toFixed(1);
  return `כרגע ב${place}: ${temp}°C, ${widget.condition}.`;
}
