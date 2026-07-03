import { fetchJson } from "../webSearch/fetchJson";
import {
  buildWeatherSampleGrid,
  chunkArray,
  classifyWeatherCell,
  type WeatherCell,
} from "./weatherGrid";

type OpenMeteoCurrent = {
  weather_code?: number;
  precipitation?: number;
  cloud_cover?: number;
};

type OpenMeteoOne = {
  latitude?: number;
  longitude?: number;
  current?: OpenMeteoCurrent;
};

type OpenMeteoMulti = OpenMeteoOne[];

const BATCH = 72;

export async function fetchGlobalWeatherCells(): Promise<WeatherCell[]> {
  const grid = buildWeatherSampleGrid(20);
  const batches = chunkArray(grid, BATCH);
  const cells: WeatherCell[] = [];

  for (const batch of batches) {
    const lat = batch.map((p) => p.lat).join(",");
    const lon = batch.map((p) => p.lon).join(",");
    const url =
      `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}` +
      "&current=weather_code,precipitation,cloud_cover&timezone=UTC";

    try {
      const data = await fetchJson<OpenMeteoOne | OpenMeteoMulti>(url, undefined, {
        timeoutMs: 18_000,
      });
      const rows = Array.isArray(data) ? data : [data];
      for (const row of rows) {
        const latV = row.latitude;
        const lonV = row.longitude;
        const cur = row.current;
        if (!Number.isFinite(latV) || !Number.isFinite(lonV) || !cur) continue;
        const weatherCode = cur.weather_code ?? 0;
        const precipitation = cur.precipitation ?? 0;
        const cloudCover = cur.cloud_cover ?? 0;
        const kind = classifyWeatherCell(weatherCode, precipitation, cloudCover);
        if (kind === "clear") continue;
        cells.push({
          lat: latV!,
          lon: lonV!,
          kind,
          precipitation,
          cloudCover,
          weatherCode,
        });
      }
    } catch {
      /* skip failed batch */
    }
  }

  return cells;
}
