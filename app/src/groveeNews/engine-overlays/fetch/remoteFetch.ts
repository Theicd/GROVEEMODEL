// @ts-nocheck
import { extractArticleImageFromHtml } from "../extract/articleImage";
import { normalizeArticleBody } from "../extract/normalizeArticleBody";

const DEFAULT_TIMEOUT_MS = 22_000;

export type FetchRouteContext = {
  dev: boolean;
  hostname: string;
  proxyUrl: string;
};

/** Local Vite proxy — no CORS in dev/preview. */
export const localProxy = (url: string) => `/api/fetch?url=${encodeURIComponent(url)}`;

export const configuredProxy = (base: string, url: string) =>
  `${base.replace(/\/$/, "")}?url=${encodeURIComponent(url)}`;

const EXTERNAL_RELAYS = [
  (url: string) => `https://api.allorigins.win/raw?url=${encodeURIComponent(url)}`,
  (url: string) => `https://api.allorigins.win/get?url=${encodeURIComponent(url)}`,
  (url: string) => `https://api.cors.lol/?url=${encodeURIComponent(url)}`,
  (url: string) => `https://cors.eu.org/${url}`,
  (url: string) => `https://r.jina.ai/${url}`,
];

export function hasLocalFetchProxy(dev: boolean, hostname: string): boolean {
  if (dev) return true;
  return hostname === "localhost" || hostname === "127.0.0.1";
}

/** Static hosts (GitHub Pages) have no local proxy — avoid hammering public CORS relays. */
export function isStaticWebHost(): boolean {
  if (typeof window === "undefined") return false;
  const { hostname, port } = window.location;
  if (hostname.endsWith("github.io")) return true;
  if (import.meta.env.DEV) return false;
  if (port === "5173" || port === "4173" || port === "5180") return false;
  return hostname !== "localhost" && hostname !== "127.0.0.1";
}

export function currentFetchContext(): FetchRouteContext {
  const hostname = typeof window !== "undefined" ? window.location.hostname : "";
  return {
    dev: import.meta.env.DEV,
    hostname,
    proxyUrl: (import.meta.env.VITE_FETCH_PROXY_URL as string | undefined)?.trim() || "",
  };
}

export function buildFetchAttempts(targetUrl: string, ctx: FetchRouteContext): string[] {
  if (hasLocalFetchProxy(ctx.dev, ctx.hostname)) {
    return [localProxy(targetUrl)];
  }

  const attempts: string[] = [];

  if (ctx.proxyUrl) {
    attempts.push(configuredProxy(ctx.proxyUrl, targetUrl));
  }

  if (isSameOrigin(targetUrl)) {
    attempts.unshift(targetUrl);
  }

  const staticHost = !ctx.dev && (ctx.hostname.endsWith("github.io") || isStaticWebHost());
  if (!staticHost || ctx.proxyUrl) {
    for (const relay of EXTERNAL_RELAYS) {
      attempts.push(relay(targetUrl));
    }
  }

  return [...new Set(attempts)];
}

function isSameOrigin(url: string): boolean {
  try {
    return new URL(url).origin === window.location.origin;
  } catch {
    return false;
  }
}

export type FetchBackend = "local-proxy" | "configured-proxy" | "rss-cache" | "browser-relays";

export type FetchProbeResult = {
  ok: boolean;
  mode: FetchBackend;
  detail: string;
};

export function resolveFetchBackend(ctx: FetchRouteContext): FetchBackend {
  if (hasLocalFetchProxy(ctx.dev, ctx.hostname)) return "local-proxy";
  if (ctx.proxyUrl) return "configured-proxy";
  return "browser-relays";
}

/** Quick connectivity check — logs whether RSS fetch path works in this environment. */
export async function probeFetchBackend(
  testUrl = "https://feeds.bbci.co.uk/news/rss.xml",
): Promise<FetchProbeResult> {
  const ctx = currentFetchContext();
  if (isStaticWebHost() && !ctx.proxyUrl) {
    const { isRssCacheAvailable, getRssCacheSummary } = await import("./rssCache");
    if (await isRssCacheAvailable()) {
      const summary = await getRssCacheSummary();
      return {
        ok: true,
        mode: "rss-cache",
        detail: summary
          ? `Pre-built RSS cache (${summary.okCount}/${summary.feedCount} feeds)`
          : "Pre-built RSS cache loaded",
      };
    }
  }
  const mode = resolveFetchBackend(ctx);
  try {
    const xml = await fetchRemoteText(testUrl, 18_000);
    const ok = /<rss|<feed/i.test(xml);
    return {
      ok,
      mode,
      detail: ok ? "BBC RSS reachable" : "Response was not RSS XML",
    };
  } catch (err) {
    const msg = err instanceof Error ? err.message : "Fetch failed";
    if (mode === "browser-relays") {
      return {
        ok: false,
        mode,
        detail: `${msg} — use npm run dev at http://127.0.0.1:5180 for local proxy`,
      };
    }
    return { ok: false, mode, detail: msg };
  }
}

export async function fetchRemoteText(url: string, timeoutMs = DEFAULT_TIMEOUT_MS): Promise<string> {
  const attempts = buildFetchAttempts(url, currentFetchContext());

  let lastErr: unknown;
  for (const attemptUrl of attempts) {
    try {
      return await fetchText(attemptUrl, timeoutMs, url);
    } catch (err) {
      lastErr = err;
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error("All fetch attempts failed");
}

async function fetchText(fetchUrl: string, timeoutMs: number, originalUrl: string): Promise<string> {
  const res = await fetchWithTimeout(fetchUrl, timeoutMs);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  let text = await res.text();
  if (!text.trim()) throw new Error("Empty response");

  if (fetchUrl.includes("allorigins.win/get")) {
    try {
      const parsed = JSON.parse(text) as { contents?: string };
      if (parsed.contents) text = parsed.contents;
    } catch {
      /* use raw */
    }
  }

  return normalizeRelayText(text, originalUrl);
}

function normalizeRelayText(text: string, sourceUrl: string): string {
  const { body, title } = normalizeArticleBody(text);
  if (title && /markdown content:/i.test(text)) {
    return `Title: ${title}\n\n${body}`;
  }
  if (body !== text.trim()) return body;
  return text;
}

async function fetchWithTimeout(url: string, timeoutMs: number): Promise<Response> {
  const ctrl = new AbortController();
  const id = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    return await fetch(url, {
      signal: ctrl.signal,
      headers: { Accept: "text/html,application/xhtml+xml,application/xml,text/plain,*/*" },
    });
  } finally {
    clearTimeout(id);
  }
}

export function extractOgImage(html: string, pageUrl = ""): string {
  return extractArticleImageFromHtml(html, pageUrl || "https://localhost/");
}
