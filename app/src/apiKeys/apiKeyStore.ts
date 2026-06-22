import { isProviderEnabled } from "./apiProviderUsage";

const AISSTREAM_KEY = "grovee-aisstream-api-key";
const TAVILY_KEY = "grovee-tavily-api-key";
const SCAVIO_KEY = "grovee-scavio-api-key";

export const AISSTREAM_KEY_SAVED_EVENT = "grovee-aisstream-key-saved";
export const TAVILY_KEY_SAVED_EVENT = "grovee-tavily-key-saved";
export const SCAVIO_KEY_SAVED_EVENT = "grovee-scavio-key-saved";

export type ApiKeyProviderId = "aisstream" | "tavily" | "scavio";

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

const maskKey = (key: string): string => {
  if (key.length <= 8) return "••••";
  return `${key.slice(0, 4)}…${key.slice(-4)}`;
};

const getKeyForProvider = (id: ApiKeyProviderId): string | undefined => {
  if (id === "aisstream") return getAisStreamApiKey();
  if (id === "tavily") return getTavilyApiKey();
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
