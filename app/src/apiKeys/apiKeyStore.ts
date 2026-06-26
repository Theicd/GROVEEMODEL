import { isProviderEnabled } from "./apiProviderUsage";

const AISSTREAM_KEY = "grovee-aisstream-api-key";
const TAVILY_KEY = "grovee-tavily-api-key";
const SCAVIO_KEY = "grovee-scavio-api-key";
const TMDB_KEY = "grovee-tmdb-api-key";
const TMDB_V4_KEY = "grovee-tmdb-v4-token";

/** Built-in TMDB credentials — shown and used from «מפתחות API» by default. */
export const DEFAULT_TMDB_API_KEY = "c1b70f76d13dcb2268678fc347fb0f68";
export const DEFAULT_TMDB_V4_TOKEN =
  "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJjMWI3MGY3NmQxM2RjYjIyNjg2NzhmYzM0N2ZiMGY2OCIsIm5iZiI6MTc1NTU1MzI2MS4yNjIsInN1YiI6IjY4YTM5ZGVkNzk2MTFjNDRiNTNlZTZkMyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.PWSY-Yx-NBfme1ID8vqTXCPTWeZ7tD___dn_ZD03qYE";

export const AISSTREAM_KEY_SAVED_EVENT = "grovee-aisstream-key-saved";
export const TAVILY_KEY_SAVED_EVENT = "grovee-tavily-key-saved";
export const SCAVIO_KEY_SAVED_EVENT = "grovee-scavio-key-saved";
export const TMDB_KEY_SAVED_EVENT = "grovee-tmdb-key-saved";

export type ApiKeyProviderId = "aisstream" | "tavily" | "scavio" | "tmdb";

export type ApiKeyEntry = {
  id: ApiKeyProviderId;
  labelHe: string;
  labelEn: string;
  hintHe: string;
  docsUrl: string;
  configured: boolean;
  masked?: string;
};

export const API_KEY_CATALOG: Record<
  ApiKeyProviderId,
  Omit<ApiKeyEntry, "configured" | "masked">
> = {
  aisstream: {
    id: "aisstream",
    labelHe: "AISStream — ספינות AIS חי",
    labelEn: "AISStream — live AIS ships",
    hintHe: "מפתח חינמי מ-aisstream.io — ים תיכון, סואץ, חיפה (WebSocket דרך proxy מקומי)",
    docsUrl: "https://aisstream.io/documentation",
  },
  tavily: {
    id: "tavily",
    labelHe: "Tavily — חיפוש אתרים כללי",
    labelEn: "Tavily — general web search",
    hintHe: "מפתח מ-tavily.com — חיפוש web מתקדם ל-SERP ו-fallback",
    docsUrl: "https://docs.tavily.com/",
  },
  scavio: {
    id: "scavio",
    labelHe: "Scavio — Google Search API",
    labelEn: "Scavio — Google web search",
    hintHe: "מפתח מ-scavio.dev — תוצאות Google מובנות (Bearer sk_…)",
    docsUrl: "https://scavio.dev/docs/search-api",
  },
  tmdb: {
    id: "tmdb",
    labelHe: "TMDB — מידע סרטים וסדרות (EPG)",
    labelEn: "TMDB — movie & TV metadata (EPG)",
    hintHe: "מפתח API v3 מ-themoviedb.org — משך סרט, תקציר, פוסטר לערוצי Live TV",
    docsUrl: "https://developer.themoviedb.org/docs/getting-started",
  },
};

export const getAisStreamApiKey = (): string | undefined => {
  const fromEnv = (import.meta.env.VITE_AISSTREAM_API_KEY as string | undefined)?.trim();
  if (fromEnv) return fromEnv;
  if (typeof localStorage === "undefined") return undefined;
  const stored = localStorage.getItem(AISSTREAM_KEY)?.trim();
  return stored || undefined;
};

export const setAisStreamApiKey = (key: string): void => {
  if (typeof localStorage === "undefined") return;
  const t = key.trim();
  if (t) localStorage.setItem(AISSTREAM_KEY, t);
  else localStorage.removeItem(AISSTREAM_KEY);
};

export const getTavilyApiKey = (): string | undefined => {
  const fromEnv = (import.meta.env.VITE_TAVILY_API_KEY as string | undefined)?.trim();
  if (fromEnv) return fromEnv;
  if (typeof localStorage === "undefined") return undefined;
  const stored = localStorage.getItem(TAVILY_KEY)?.trim();
  return stored || undefined;
};

export const setTavilyApiKey = (key: string): void => {
  if (typeof localStorage === "undefined") return;
  const t = key.trim();
  if (t) localStorage.setItem(TAVILY_KEY, t);
  else localStorage.removeItem(TAVILY_KEY);
};

export const getScavioApiKey = (): string | undefined => {
  const fromEnv = (import.meta.env.VITE_SCAVIO_API_KEY as string | undefined)?.trim();
  if (fromEnv) return fromEnv;
  if (typeof localStorage === "undefined") return undefined;
  const stored = localStorage.getItem(SCAVIO_KEY)?.trim();
  return stored || undefined;
};

export const setScavioApiKey = (key: string): void => {
  if (typeof localStorage === "undefined") return;
  const t = key.trim();
  if (t) localStorage.setItem(SCAVIO_KEY, t);
  else localStorage.removeItem(SCAVIO_KEY);
};

export const getTmdbApiKey = (): string | undefined => {
  const fromEnv = (import.meta.env.VITE_TMDB_API_KEY as string | undefined)?.trim();
  if (fromEnv) return fromEnv;
  if (typeof localStorage !== "undefined") {
    const stored = localStorage.getItem(TMDB_KEY)?.trim();
    if (stored) return stored;
  }
  return DEFAULT_TMDB_API_KEY;
};

export const getTmdbV4Token = (): string | undefined => {
  const fromEnv = (import.meta.env.VITE_TMDB_V4_TOKEN as string | undefined)?.trim();
  if (fromEnv) return fromEnv;
  if (typeof localStorage !== "undefined") {
    const stored = localStorage.getItem(TMDB_V4_KEY)?.trim();
    if (stored) return stored;
  }
  return DEFAULT_TMDB_V4_TOKEN;
};

export const setTmdbApiKey = (key: string): void => {
  if (typeof localStorage === "undefined") return;
  const t = key.trim();
  if (t) localStorage.setItem(TMDB_KEY, t);
  else localStorage.removeItem(TMDB_KEY);
};

export const setTmdbV4Token = (token: string): void => {
  if (typeof localStorage === "undefined") return;
  const t = token.trim();
  if (t) localStorage.setItem(TMDB_V4_KEY, t);
  else localStorage.removeItem(TMDB_V4_KEY);
};

/** Pre-fill TMDB fields in localStorage so the API keys UI shows them on first open. */
export const ensureTmdbDefaultsInstalled = (): void => {
  if (typeof localStorage === "undefined") return;
  if (!localStorage.getItem(TMDB_KEY)?.trim()) {
    localStorage.setItem(TMDB_KEY, DEFAULT_TMDB_API_KEY);
  }
  if (!localStorage.getItem(TMDB_V4_KEY)?.trim()) {
    localStorage.setItem(TMDB_V4_KEY, DEFAULT_TMDB_V4_TOKEN);
  }
};

if (typeof window !== "undefined") {
  ensureTmdbDefaultsInstalled();
}

const maskKey = (key: string): string => {
  if (key.length <= 8) return "••••";
  return `${key.slice(0, 4)}…${key.slice(-4)}`;
};

const getKeyForProvider = (id: ApiKeyProviderId): string | undefined => {
  if (id === "aisstream") return getAisStreamApiKey();
  if (id === "tavily") return getTavilyApiKey();
  if (id === "tmdb") return getTmdbApiKey();
  return getScavioApiKey();
};

export const listApiKeyEntries = (): ApiKeyEntry[] =>
  Object.values(API_KEY_CATALOG).map((meta) => {
    const key = getKeyForProvider(meta.id);
    return {
      ...meta,
      configured: Boolean(key),
      masked: key ? maskKey(key) : undefined,
    };
  });

export const isAisStreamConfigured = (): boolean =>
  isProviderEnabled("aisstream") && Boolean(getAisStreamApiKey());
export const isTavilyConfigured = (): boolean =>
  isProviderEnabled("tavily") && Boolean(getTavilyApiKey());
export const isScavioConfigured = (): boolean =>
  isProviderEnabled("scavio") && Boolean(getScavioApiKey());
export const isTmdbConfigured = (): boolean =>
  isProviderEnabled("tmdb") && Boolean(getTmdbApiKey());

export {
  isProviderEnabled,
  setProviderEnabled,
  getProviderUsage,
  resetProviderUsage,
  formatUsageSummaryHe,
  formatBytesKb,
  PROVIDER_ENABLED_EVENT,
  PROVIDER_USAGE_EVENT,
} from "./apiProviderUsage";
