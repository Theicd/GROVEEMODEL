export type WeatherCellKind = "clear" | "cloudy" | "rain" | "snow" | "thunder" | "fog";

export type WeatherCell = {
  lat: number;
  lon: number;
  kind: WeatherCellKind;
  precipitation: number;
  cloudCover: number;
  weatherCode: number;
};

/** Coarse global sampling grid (Open-Meteo, real-time). */
export function buildWeatherSampleGrid(stepDeg = 20): Array<{ lat: number; lon: number }> {
  const out: Array<{ lat: number; lon: number }> = [];
  for (let lat = -60; lat <= 60; lat += stepDeg) {
    for (let lon = -180; lon < 180; lon += stepDeg) {
      out.push({ lat, lon });
    }
  }
  return out;
}

export function classifyWeatherCell(
  weatherCode: number,
  precipitation: number,
  cloudCover: number,
): WeatherCellKind {
  if (weatherCode === 95 || weatherCode === 96 || weatherCode === 99) return "thunder";
  if (weatherCode === 45 || weatherCode === 48) return "fog";
  if (weatherCode >= 71 && weatherCode <= 77) return "snow";
  if (precipitation >= 0.15 || (weatherCode >= 61 && weatherCode <= 67) || weatherCode === 80) {
    return "rain";
  }
  if (cloudCover >= 55 || weatherCode === 3) return "cloudy";
  return "clear";
}

export function chunkArray<T>(arr: T[], size: number): T[][] {
  const out: T[][] = [];
  for (let i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size));
  return out;
}
