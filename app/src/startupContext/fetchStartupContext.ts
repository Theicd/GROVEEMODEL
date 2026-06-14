import { fetchJson } from "../webSearch/fetchJson";
import type { StartupContext } from "./types";

const CACHE_KEY = "grovee-startup-context-v1";
const CACHE_TTL_MS = 30 * 60 * 1000;

type TimeNowIp = {
  timezone?: string;
  datetime?: string;
  utc_offset?: string;
  abbreviation?: string;
  dst?: boolean;
  day_of_week?: number;
  week_number?: number;
  unixtime?: number;
  client_ip?: string;
};

type IpApiCo = {
  country_code?: string;
  country_name?: string;
  city?: string;
  region?: string;
  latitude?: number;
  longitude?: number;
};

const DEFAULT: StartupContext = {
  fetchedAt: Date.now(),
  datetime: new Date().toISOString(),
  timezone: "Asia/Jerusalem",
  utcOffset: "+03:00",
  dst: false,
  dayOfWeek: new Date().getDay(),
  countryCode: "IL",
  countryName: "Israel",
  cityName: "Jerusalem",
  lat: 31.7683,
  lon: 35.2137,
};

let memoryCache: StartupContext | null = null;

const readSessionCache = (): StartupContext | null => {
  try {
    const raw = sessionStorage.getItem(CACHE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as StartupContext;
    if (Date.now() - parsed.fetchedAt > CACHE_TTL_MS) return null;
    return parsed;
  } catch {
    return null;
  }
};

const writeSessionCache = (ctx: StartupContext): void => {
  try {
    sessionStorage.setItem(CACHE_KEY, JSON.stringify(ctx));
  } catch {
    /* ignore */
  }
};

const browserTimezone = (): string => {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || DEFAULT.timezone;
  } catch {
    return DEFAULT.timezone;
  }
};

const fetchTimeNowIp = async (): Promise<Partial<StartupContext>> => {
  const j = await fetchJson<TimeNowIp>("https://time.now/developer/api/ip", undefined, {
    timeoutMs: 8000,
  });
  return {
    datetime: j.datetime ?? new Date().toISOString(),
    timezone: j.timezone ?? browserTimezone(),
    utcOffset: j.utc_offset ?? "",
    abbreviation: j.abbreviation,
    dst: Boolean(j.dst),
    dayOfWeek: typeof j.day_of_week === "number" ? j.day_of_week : new Date().getDay(),
    weekNumber: j.week_number,
    unixtime: j.unixtime,
    clientIp: j.client_ip,
  };
};

const fetchIpGeo = async (): Promise<Partial<StartupContext>> => {
  const j = await fetchJson<IpApiCo>("https://ipapi.co/json/", undefined, { timeoutMs: 8000 });
  if (!j.country_code) throw new Error("no country");
  return {
    countryCode: j.country_code.toUpperCase(),
    countryName: j.country_name || j.country_code,
    cityName: j.city || undefined,
    regionName: j.region || undefined,
    lat: Number(j.latitude) || DEFAULT.lat,
    lon: Number(j.longitude) || DEFAULT.lon,
  };
};

const mergeContext = (
  time: Partial<StartupContext>,
  geo: Partial<StartupContext>,
): StartupContext => ({
  ...DEFAULT,
  ...geo,
  ...time,
  fetchedAt: Date.now(),
  timezone: time.timezone ?? geo.timezone ?? browserTimezone(),
  datetime: time.datetime ?? new Date().toISOString(),
  utcOffset: time.utcOffset ?? DEFAULT.utcOffset,
  dst: time.dst ?? false,
  dayOfWeek: time.dayOfWeek ?? new Date().getDay(),
  countryCode: geo.countryCode ?? DEFAULT.countryCode,
  countryName: geo.countryName ?? DEFAULT.countryName,
  lat: geo.lat ?? DEFAULT.lat,
  lon: geo.lon ?? DEFAULT.lon,
});

/** Fetch once per session — Time.Now time + ipapi geo. */
export async function fetchStartupContext(force = false): Promise<StartupContext> {
  if (!force) {
    if (memoryCache) return memoryCache;
    const cached = readSessionCache();
    if (cached) {
      memoryCache = cached;
      return cached;
    }
  }

  let timePart: Partial<StartupContext> = {};
  let geoPart: Partial<StartupContext> = {};

  const [timeRes, geoRes] = await Promise.allSettled([fetchTimeNowIp(), fetchIpGeo()]);

  if (timeRes.status === "fulfilled") timePart = timeRes.value;
  else {
    timePart = {
      datetime: new Date().toISOString(),
      timezone: browserTimezone(),
      dayOfWeek: new Date().getDay(),
    };
  }

  if (geoRes.status === "fulfilled") geoPart = geoRes.value;
  else geoPart = { countryCode: DEFAULT.countryCode, countryName: DEFAULT.countryName, lat: DEFAULT.lat, lon: DEFAULT.lon };

  const ctx = mergeContext(timePart, geoPart);
  memoryCache = ctx;
  writeSessionCache(ctx);
  return ctx;
}

export function getStartupContextSync(): StartupContext | null {
  return memoryCache ?? readSessionCache();
}

export function clearStartupContextCache(): void {
  memoryCache = null;
  try {
    sessionStorage.removeItem(CACHE_KEY);
  } catch {
    /* ignore */
  }
}

/** Refresh Open-Meteo current temp for header widget. */
export async function refreshLocalWeather(ctx: StartupContext): Promise<StartupContext> {
  try {
    const url =
      `https://api.open-meteo.com/v1/forecast?latitude=${ctx.lat}&longitude=${ctx.lon}` +
      `&current=temperature_2m,weather_code&timezone=auto&forecast_days=1`;
    const data = await fetchJson<{ current?: { temperature_2m?: number; weather_code?: number } }>(
      url,
      undefined,
      { timeoutMs: 10_000 },
    );
    const next = {
      ...ctx,
      localTempC: data.current?.temperature_2m,
      localWeatherCode: data.current?.weather_code,
    };
    memoryCache = next;
    writeSessionCache(next);
    return next;
  } catch {
    return ctx;
  }
}
