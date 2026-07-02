import { describe, expect, it } from "vitest";
import { buildShortWeatherReply, buildWeatherWidgetFromForecast } from "./buildWeatherWidget";
import { isWeatherForecastQuery } from "./inChatWeather";

describe("weatherWidget", () => {
  it("builds widget from forecast payload", () => {
    const widget = buildWeatherWidgetFromForecast({
      place: { name: "Berlin", latitude: 52.52, longitude: 13.41, country_code: "DE", admin1: "Berlin" },
      placeLabel: "Berlin, Berlin, DE",
      current: {
        time: "2026-07-02T14:00",
        temperature_2m: 25.4,
        apparent_temperature: 24,
        relative_humidity_2m: 40,
        weather_code: 2,
        wind_speed_10m: 14.3,
        wind_direction_10m: 245,
      },
      daily: {
        time: ["2026-07-02", "2026-07-03", "2026-07-04"],
        temperature_2m_min: [15.2, 15.1, 13.7],
        temperature_2m_max: [27.8, 22.8, 19.7],
        precipitation_probability_max: [50, 5, 45],
        weather_code: [80, 2, 61],
        wind_speed_10m_max: [23.5, 28.3, 18.8],
      },
      includeForecast: true,
      sourceLabel: "מזג אוויר (Open-Meteo)",
    });

    expect(widget?.temperatureC).toBe(25.4);
    expect(widget?.condition).toContain("מעונן");
    expect(widget?.forecast).toHaveLength(3);
    expect(widget?.forecast[0].tempBarPct).toBeGreaterThan(0);
  });

  it("omits forecast when includeForecast is false", () => {
    const widget = buildWeatherWidgetFromForecast({
      place: { name: "Berlin", latitude: 52.52, longitude: 13.41, country_code: "DE", admin1: "Berlin" },
      placeLabel: "Berlin, Berlin, DE",
      current: {
        time: "2026-07-02T14:00",
        temperature_2m: 25.4,
        weather_code: 2,
      },
      daily: {
        time: ["2026-07-02", "2026-07-03", "2026-07-04"],
        temperature_2m_min: [15.2, 15.1, 13.7],
        temperature_2m_max: [27.8, 22.8, 19.7],
      },
      includeForecast: false,
      sourceLabel: "מזג אוויר (Open-Meteo)",
    });

    expect(widget?.temperatureC).toBe(25.4);
    expect(widget?.forecast).toHaveLength(0);
  });

  it("builds short Hebrew reply", () => {
    const reply = buildShortWeatherReply({
      placeLabel: "Berlin, Berlin, DE",
      cityName: "Berlin",
      regionLabel: "Berlin, DE",
      condition: "מעונן חלקית",
      iconKind: "partly-cloudy",
      temperatureC: 25.4,
      forecast: [],
      sourceLabel: "Open-Meteo",
    });
    expect(reply).toContain("25.4");
    expect(reply).toContain("Berlin");
  });

  it("detects forecast vs current-only weather queries", () => {
    expect(isWeatherForecastQuery("תחזית מזג האוויר בלונדון")).toBe(true);
    expect(isWeatherForecastQuery("מה הטמפרטורה בברזיל")).toBe(false);
    expect(isWeatherForecastQuery("מה מזג האוויר בתל אביב")).toBe(false);
  });
});
