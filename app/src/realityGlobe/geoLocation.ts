export type UserGeoRegion = {
  countryCode: string;
  countryName: string;
  lat: number;
  lon: number;
  cityName?: string;
};

/** Approximate region from IP — delegates to unified StartupContext when available. */
export async function fetchUserGeoRegion(): Promise<UserGeoRegion> {
  const { fetchStartupContext } = await import("../startupContext");
  const ctx = await fetchStartupContext();
  return {
    countryCode: ctx.countryCode,
    countryName: ctx.countryName,
    lat: ctx.lat,
    lon: ctx.lon,
    cityName: ctx.cityName,
  };
}
