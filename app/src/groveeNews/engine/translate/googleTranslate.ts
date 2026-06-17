// @ts-nocheck
import { decodeHtmlEntities } from "./decodeHtmlEntities";
import { currentFetchContext, localProxy } from "../fetch/remoteFetch";

const CACHE_KEY = "gn-google-translate-cache-v2";
const MAX_CACHE_ENTRIES = 800;

export type TranslateProvider = "cloud" | "gtx" | "cache";

export type TranslateBatchResult = {
  texts: string[];
  provider: TranslateProvider;
};

type CacheStore = Record<string, string>;

let memoryCache: CacheStore | null = null;

function loadCache(): CacheStore {
  if (memoryCache) return memoryCache;
  try {
    const raw = localStorage.getItem(CACHE_KEY);
    memoryCache = raw ? (JSON.parse(raw) as CacheStore) : {};
  } catch {
    memoryCache = {};
  }
  return memoryCache;
}

function saveCache(store: CacheStore): void {
  const keys = Object.keys(store);
  if (keys.length > MAX_CACHE_ENTRIES) {
    for (const k of keys.slice(0, keys.length - MAX_CACHE_ENTRIES)) {
      delete store[k];
    }
  }
  memoryCache = store;
  try {
    localStorage.setItem(CACHE_KEY, JSON.stringify(store));
  } catch {
    /* quota */
  }
}

function cacheKey(text: string, target: string): string {
  return `${target}::${text.trim()}`;
}

function normalizeTranslated(text: string): string {
  return decodeHtmlEntities(text).replace(/\s+/g, " ").trim();
}

function translateProxyUrl(): string {
  return (import.meta.env.VITE_TRANSLATE_PROXY_URL as string | undefined)?.trim() || "";
}

function cloudApiKey(): string {
  return (import.meta.env.VITE_GOOGLE_TRANSLATE_API_KEY as string | undefined)?.trim() || "";
}

function canUseLocalTranslateProxy(): boolean {
  const ctx = currentFetchContext();
  return ctx.dev && (ctx.hostname === "localhost" || ctx.hostname === "127.0.0.1");
}

async function postJson(url: string, body: unknown, timeoutMs = 25_000): Promise<Response> {
  const ctrl = new AbortController();
  const timer = window.setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    return await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
  } finally {
    window.clearTimeout(timer);
  }
}

async function translateViaCloudApi(texts: string[], target = "he", source = "auto"): Promise<string[]> {
  const proxy = translateProxyUrl();
  const key = cloudApiKey();

  const body: Record<string, unknown> = { q: texts, target, format: "text" };
  if (source && source !== "auto") body.source = source;

  if (proxy) {
    const res = await postJson(proxy, { texts, target, source });
    if (!res.ok) throw new Error(`Translate proxy ${res.status}`);
    const data = (await res.json()) as { translations?: string[] };
    if (!data.translations?.length) throw new Error("Empty translate proxy response");
    return data.translations.map(normalizeTranslated);
  }

  if (canUseLocalTranslateProxy()) {
    const res = await postJson("/api/translate", { texts, target, source });
    if (!res.ok) throw new Error(`Local translate proxy ${res.status}`);
    const data = (await res.json()) as { translations?: string[] };
    if (!data.translations?.length) throw new Error("Empty local translate response");
    return data.translations.map(normalizeTranslated);
  }

  if (!key) throw new Error("No Google Translate API key");

  const res = await postJson(
    `https://translation.googleapis.com/language/translate/v2?key=${encodeURIComponent(key)}`,
    body,
  );
  if (!res.ok) throw new Error(`Google Cloud Translate ${res.status}`);
  const data = (await res.json()) as {
    data?: { translations?: { translatedText: string }[] };
  };
  const out = data.data?.translations?.map((t) => normalizeTranslated(t.translatedText)) ?? [];
  if (out.length !== texts.length) throw new Error("Translation count mismatch");
  return out;
}

async function fetchGtxJson(url: string): Promise<unknown> {
  const ctx = currentFetchContext();
  const attempts: string[] = [];

  if (canUseLocalTranslateProxy()) {
    attempts.push(localProxy(url));
  } else if (ctx.proxyUrl) {
    attempts.push(`${ctx.proxyUrl.replace(/\/$/, "")}?url=${encodeURIComponent(url)}`);
  }
  attempts.push(url);

  let lastErr: unknown;
  for (const attempt of [...new Set(attempts)]) {
    try {
      const res = await fetch(attempt, { headers: { Accept: "application/json" } });
      if (!res.ok) throw new Error(`gtx ${res.status}`);
      return await res.json();
    } catch (err) {
      lastErr = err;
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error("gtx fetch failed");
}

async function translateViaGtx(text: string, target = "he"): Promise<string> {
  const q = encodeURIComponent(text);
  const url = `https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=${target}&dt=t&q=${q}`;
  const data = (await fetchGtxJson(url)) as [Array<[string, string, ...unknown[]]>, ...unknown[]];
  const parts = data[0] ?? [];
  const joined = parts.map((chunk) => chunk[0]).join("");
  return normalizeTranslated(joined || text);
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => window.setTimeout(r, ms));
}

/** Translate many strings — cache → Cloud API → Google gtx relay. */
export async function translateTexts(
  texts: string[],
  targetLang: string,
  sourceLang = "auto",
): Promise<TranslateBatchResult> {
  const target = targetLang.trim().toLowerCase() || "en";
  const unique = [...new Set(texts.map((t) => t.trim()).filter(Boolean))];
  const store = loadCache();
  const outMap = new Map<string, string>();
  const missing: string[] = [];

  for (const text of unique) {
    const hit = store[cacheKey(text, target)];
    if (hit) outMap.set(text, hit);
    else missing.push(text);
  }

  if (!missing.length) {
    return {
      texts: texts.map((t) => outMap.get(t.trim()) ?? t),
      provider: "cache",
    };
  }

  let provider: TranslateProvider = "cloud";
  let translatedMissing: string[] = [];
  const src = sourceLang === "auto" ? "auto" : sourceLang;

  try {
    const CHUNK = 40;
    for (let i = 0; i < missing.length; i += CHUNK) {
      const chunk = missing.slice(i, i + CHUNK);
      const part = await translateViaCloudApi(chunk, target, src);
      translatedMissing.push(...part);
    }
  } catch {
    provider = "gtx";
    translatedMissing = [];
    const GTX_CHUNK = 4;
    for (let i = 0; i < missing.length; i += GTX_CHUNK) {
      const chunk = missing.slice(i, i + GTX_CHUNK);
      const part = await Promise.all(chunk.map((t) => translateViaGtx(t, target)));
      translatedMissing.push(...part);
      if (i + GTX_CHUNK < missing.length) await sleep(120);
    }
  }

  missing.forEach((text, i) => {
    const translated = translatedMissing[i] || text;
    store[cacheKey(text, target)] = translated;
    outMap.set(text, translated);
  });
  saveCache(store);

  return {
    texts: texts.map((t) => {
      const key = t.trim();
      return key ? (outMap.get(key) ?? t) : t;
    }),
    provider,
  };
}

/** @deprecated use translateTexts(texts, "he") */
export async function translateTextsToHebrew(texts: string[]): Promise<TranslateBatchResult> {
  return translateTexts(texts, "he", "en");
}

export function clearTranslateCache(): void {
  memoryCache = {};
  try {
    localStorage.removeItem(CACHE_KEY);
  } catch {
    /* ignore */
  }
}
