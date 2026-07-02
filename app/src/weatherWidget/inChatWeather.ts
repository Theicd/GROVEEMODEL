import { isWeatherQuery } from "../webSearch/intents";
import type { SearchSourceResult } from "../webSearch/types";
import { buildShortWeatherReply } from "./buildWeatherWidget";
import type { WeatherWidgetData } from "./types";

export { buildShortWeatherReply };

/** User explicitly asked for a multi-day / tomorrow / rain forecast — not plain current temp. */
export function isWeatherForecastQuery(text: string): boolean {
  if (!isWeatherQuery(text)) return false;
  if (/(?:תחזית|forecast)/i.test(text)) return true;
  if (/(?:שבוע|weekly|שבועי|7\s*days?)/i.test(text)) return true;
  if (/(?:מחר|tomorrow)/i.test(text) && /(?:מזג|weather|גשם|rain|טמפרטור|temperatur)/i.test(text)) {
    return true;
  }
  if (/(?:גשם|rain|ממטר|precipitation)/i.test(text) && /(?:תחזית|forecast|צפוי)/i.test(text)) {
    return true;
  }
  return false;
}

export function isWeatherWidgetQuery(text: string): boolean {
  return isWeatherQuery(text);
}

/** Chat should show only the weather card — no bullet text or search panel. */
export function isWeatherWidgetOnlyTurn(
  text: string,
  widget: WeatherWidgetData | null | undefined,
): boolean {
  return !!widget && isWeatherWidgetQuery(text);
}

export function attachWeatherWidgetFromSources(
  ref: { current: WeatherWidgetData | null },
  sources: SearchSourceResult[],
): void {
  const src = sources.find((s) => s.provider === "open-meteo" && s.ok && s.weatherWidget);
  ref.current = src?.weatherWidget ?? null;
}

export function resolveWeatherWidgetFromSource(
  source: SearchSourceResult,
): WeatherWidgetData | null {
  if (source.provider !== "open-meteo" || !source.ok) return null;
  return source.weatherWidget ?? null;
}
