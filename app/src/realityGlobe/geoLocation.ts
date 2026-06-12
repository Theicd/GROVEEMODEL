export type UserGeoRegion = {
  countryCode: string;
  countryName: string;
  lat: number;
  lon: number;
};

const DEFAULT_REGION: UserGeoRegion = {
  countryCode: "IL",
  countryName: "Israel",
  lat: 31.7683,
  lon: 35.2137,
};

/** Approximate country center from IP — no API key required. */
export async function fetchUserGeoRegion(): Promise<UserGeoRegion> {
  const cached = readCachedRegion();
  if (cached) return cached;

  const providers = [
    async (): Promise<UserGeoRegion | null> => {
      const j = await fetchJson<{ country_code?: string; country_name?: string; latitude?: number; longitude?: number }>(
        "https://ipapi.co/json/",
      );
      if (!j?.country_code) return null;
      return {
        countryCode: j.country_code.toUpperCase(),
        countryName: j.country_name || j.country_code,
        lat: Number(j.latitude) || DEFAULT_REGION.lat,
        lon: Number(j.longitude) || DEFAULT_REGION.lon,
      };
    },
    async (): Promise<UserGeoRegion | null> => {
      const j = await fetchJson<{ status?: string; countryCode?: string; country?: string; lat?: number; lon?: number }>(
        "http://ip-api.com/json/?fields=status,country,countryCode,lat,lon",
      );
      if (j?.status !== "success" || !j.countryCode) return null;
      return {
        countryCode: j.countryCode.toUpperCase(),
        countryName: j.country || j.countryCode,
        lat: Number(j.lat) || DEFAULT_REGION.lat,
        lon: Number(j.lon) || DEFAULT_REGION.lon,
      };
    },
  ];

  for (const provider of providers) {
    try {
      const region = await provider();
      if (region) {
        writeCachedRegion(region);
        return region;
      }
    } catch {
      /* try next */
    }
  }

  return DEFAULT_REGION;
}

async function fetchJson<T>(url: string): Promise<T> {
  const r = await fetch(url, { signal: AbortSignal.timeout(6000) });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json() as Promise<T>;
}

const CACHE_KEY = "grovee-user-geo-v1";

function readCachedRegion(): UserGeoRegion | null {
  try {
    const raw = sessionStorage.getItem(CACHE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as UserGeoRegion;
    if (!parsed.countryCode) return null;
    return parsed;
  } catch {
    return null;
  }
}

function writeCachedRegion(region: UserGeoRegion): void {
  try {
    sessionStorage.setItem(CACHE_KEY, JSON.stringify(region));
  } catch {
    /* ignore */
  }
}
