import { fetchJson } from "./fetchJson";

export type GeoPlace = {
  name: string;
  latitude: number;
  longitude: number;
  elevation?: number;
  country_code?: string;
  admin1?: string;
  timezone?: string;
};

type GeoResult = {
  results?: GeoPlace[];
};

/** Open-Meteo geocoding — shared by weather, marine, world time. */
export const geocodePlace = async (name: string): Promise<GeoPlace | null> => {
  const q = name.trim();
  if (q.length < 2) return null;
  const encoded = encodeURIComponent(q);
  const data = await fetchJson<GeoResult>(
    `https://geocoding-api.open-meteo.com/v1/search?name=${encoded}&count=3&language=he&format=json`,
  );
  return data.results?.[0] ?? null;
};

export const formatPlaceLabel = (place: GeoPlace): string =>
  [place.name, place.admin1, place.country_code].filter(Boolean).join(", ");
