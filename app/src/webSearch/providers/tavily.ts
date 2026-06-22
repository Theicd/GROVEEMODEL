import { getTavilyApiKey, isTavilyConfigured } from "../../apiKeys/apiKeyStore";
import { recordProviderUsage } from "../../apiKeys/apiProviderUsage";
import type { SearchSourceResult, WebSerpHit } from "../types";
import { promoteArchiveWebHitsToMedia } from "./archiveOrgVideo";

export type TavilyProxyResponse = {
  ok: boolean;
  hits?: WebSerpHit[];
  count?: number;
  answer?: string;
  error?: string;
  fetchedAt?: string;
};

const devProxyAvailable = (): boolean =>
  import.meta.env.DEV ||
  (typeof window !== "undefined" &&
    (window.location.hostname === "127.0.0.1" || window.location.hostname === "localhost"));

/** Open-domain web search via Tavily (POST /api/tavily/search — dev proxy only). */
export const fetchTavilySearch = async (
  query: string,
  options?: { topic?: "general" | "news" | "finance"; maxResults?: number },
): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "tavily" as const;
  const label = "Tavily (web)";
  const q = query.trim();

  if (!q) {
    return {
      provider,
      label,
      ok: false,
      text: "",
      error: "שאילתה ריקה",
      latencyMs: 0,
    };
  }

  const apiKey = getTavilyApiKey();
  if (!apiKey || !isTavilyConfigured()) {
    return {
      provider,
      label,
      ok: false,
      text: "",
      error: apiKey ? "Tavily כבוי — הפעל במסך 🔑" : "Tavily לא מוגדר — הוסף מפתח במסך 🔑",
      latencyMs: Math.round(performance.now() - started),
    };
  }

  if (!devProxyAvailable()) {
    return {
      provider,
      label,
      ok: false,
      text: "",
      error: "Tavily proxy זמין רק ב-npm run dev (127.0.0.1:5180)",
      latencyMs: Math.round(performance.now() - started),
    };
  }

  try {
    const res = await fetch("/api/tavily/search", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({
        apiKey,
        query: q,
        searchDepth: "advanced",
        maxResults: options?.maxResults ?? 12,
        topic: options?.topic ?? (/חדשות|news|מה קורה/i.test(q) ? "news" : "general"),
      }),
      signal: AbortSignal.timeout(30_000),
    });
    const data = (await res.json()) as TavilyProxyResponse;
    const rawJson = JSON.stringify(data);
    if (!res.ok || !data.ok) {
      recordProviderUsage("tavily", { ok: false, bytesApprox: rawJson.length });
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: data.error ?? `HTTP ${res.status}`,
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const rawHits = data.hits ?? [];
    const hitCount = data.count ?? rawHits.length;
    recordProviderUsage("tavily", { ok: true, hitCount, bytesApprox: rawJson.length });
    const { webHits: hits, mediaHits } = await promoteArchiveWebHitsToMedia(rawHits);
    if (!hits.length) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: "אין תוצאות Tavily",
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const lines = hits.map((r, i) => {
      const snippet = r.snippet.slice(0, 140);
      return `${i + 1}. ${r.title} · ${r.url}${snippet ? `\n   ${snippet}` : ""}`;
    });
    const header = data.answer
      ? [`תשובת Tavily: ${data.answer}`, "תוצאות אתרים (Tavily):"]
      : ["תוצאות חיפוש כללי (Tavily):"];

    return {
      provider,
      label,
      ok: true,
      text: [...header, ...lines].join("\n"),
      url: "https://tavily.com",
      webHits: hits,
      mediaHits: mediaHits.length ? mediaHits : undefined,
      latencyMs: Math.round(performance.now() - started),
    };
  } catch (err) {
    return {
      provider,
      label,
      ok: false,
      text: "",
      error: err instanceof Error ? err.message : String(err),
      latencyMs: Math.round(performance.now() - started),
    };
  }
};

export const probeTavilyConnection = async (
  apiKey: string,
  query = "artificial intelligence news",
): Promise<{ ok: boolean; count: number; message: string }> => {
  if (!devProxyAvailable()) {
    return { ok: false, count: 0, message: "Proxy זמין רק ב-npm run dev (127.0.0.1:5180)" };
  }
  try {
    const res = await fetch("/api/tavily/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ apiKey, query, searchDepth: "basic", maxResults: 3 }),
      signal: AbortSignal.timeout(35_000),
    });
    const data = (await res.json()) as TavilyProxyResponse;
    if (!res.ok || !data.ok) {
      recordProviderUsage("tavily", { ok: false, bytesApprox: JSON.stringify(data).length });
      return { ok: false, count: 0, message: data.error ?? `HTTP ${res.status}` };
    }
    const count = data.count ?? data.hits?.length ?? 0;
    recordProviderUsage("tavily", {
      ok: true,
      hitCount: count,
      bytesApprox: JSON.stringify(data).length,
    });
    return {
      ok: count > 0,
      count,
      message:
        count > 0
          ? `✓ Tavily: ${count} תוצאות (${data.hits?.[0]?.title?.slice(0, 40) ?? "—"}…)`
          : "מחובר — 0 תוצאות לבדיקה",
    };
  } catch (e) {
    return { ok: false, count: 0, message: e instanceof Error ? e.message : "שגיאת רשת" };
  }
};

export { isTavilyConfigured };
