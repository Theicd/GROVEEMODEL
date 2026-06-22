import { getScavioApiKey, isScavioConfigured } from "../../apiKeys/apiKeyStore";
import { recordProviderUsage } from "../../apiKeys/apiProviderUsage";
import type { SearchSourceResult, WebSerpHit } from "../types";
import { promoteArchiveWebHitsToMedia } from "./archiveOrgVideo";

export type ScavioProxyResponse = {
  ok: boolean;
  hits?: WebSerpHit[];
  count?: number;
  creditsRemaining?: number;
  error?: string;
  fetchedAt?: string;
};

const devProxyAvailable = (): boolean =>
  import.meta.env.DEV ||
  (typeof window !== "undefined" &&
    (window.location.hostname === "127.0.0.1" || window.location.hostname === "localhost"));

const localeForQuery = (query: string): { countryCode?: string; language?: string; searchType?: string } => {
  const he = /[\u0590-\u05FF]/.test(query);
  if (/חדשות|news|מה קורה/i.test(query)) {
    return { countryCode: he ? "il" : "us", language: he ? "he" : "en", searchType: "news" };
  }
  if (he) return { countryCode: "il", language: "he" };
  return { countryCode: "us", language: "en" };
};

/** Google web search via Scavio (POST /api/scavio/google — dev proxy). */
export const fetchScavioSearch = async (
  query: string,
  options?: { maxResults?: number },
): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "scavio" as const;
  const label = "Scavio Google (web)";
  const q = query.trim();

  if (!q) {
    return { provider, label, ok: false, text: "", error: "שאילתה ריקה", latencyMs: 0 };
  }

  const apiKey = getScavioApiKey();
  if (!apiKey || !isScavioConfigured()) {
    return {
      provider,
      label,
      ok: false,
      text: "",
      error: apiKey ? "Scavio כבוי — הפעל במסך 🔑" : "Scavio לא מוגדר — הוסף מפתח במסך 🔑",
      latencyMs: Math.round(performance.now() - started),
    };
  }

  if (!devProxyAvailable()) {
    return {
      provider,
      label,
      ok: false,
      text: "",
      error: "Scavio proxy זמין רק ב-npm run dev (127.0.0.1:5180)",
      latencyMs: Math.round(performance.now() - started),
    };
  }

  const locale = localeForQuery(q);

  try {
    const res = await fetch("/api/scavio/google", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({
        apiKey,
        query: q,
        lightRequest: true,
        maxResults: options?.maxResults ?? 12,
        ...locale,
      }),
      signal: AbortSignal.timeout(32_000),
    });
    const data = (await res.json()) as ScavioProxyResponse;
    const rawJson = JSON.stringify(data);
    if (!res.ok || !data.ok) {
      recordProviderUsage("scavio", { ok: false, bytesApprox: rawJson.length });
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
    recordProviderUsage("scavio", {
      ok: true,
      hitCount,
      bytesApprox: rawJson.length,
      creditsRemaining: data.creditsRemaining,
    });
    const { webHits: hits, mediaHits } = await promoteArchiveWebHitsToMedia(rawHits);
    if (!hits.length) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: "אין תוצאות Scavio",
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const lines = hits.map((r, i) => {
      const snippet = r.snippet.slice(0, 140);
      return `${i + 1}. ${r.title} · ${r.url}${snippet ? `\n   ${snippet}` : ""}`;
    });
    const creditNote =
      data.creditsRemaining != null ? ` · credits: ${data.creditsRemaining}` : "";
    const header = [`תוצאות Google (Scavio)${creditNote}:`];

    return {
      provider,
      label,
      ok: true,
      text: [...header, ...lines].join("\n"),
      url: "https://scavio.dev",
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

export const probeScavioConnection = async (
  apiKey: string,
  query = "hello world",
): Promise<{ ok: boolean; count: number; message: string }> => {
  if (!devProxyAvailable()) {
    return { ok: false, count: 0, message: "Proxy זמין רק ב-npm run dev (127.0.0.1:5180)" };
  }
  try {
    const res = await fetch("/api/scavio/google", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ apiKey, query, lightRequest: true, maxResults: 3 }),
      signal: AbortSignal.timeout(35_000),
    });
    const data = (await res.json()) as ScavioProxyResponse;
    if (!res.ok || !data.ok) {
      recordProviderUsage("scavio", { ok: false, bytesApprox: JSON.stringify(data).length });
      return { ok: false, count: 0, message: data.error ?? `HTTP ${res.status}` };
    }
    const count = data.count ?? data.hits?.length ?? 0;
    recordProviderUsage("scavio", {
      ok: true,
      hitCount: count,
      bytesApprox: JSON.stringify(data).length,
      creditsRemaining: data.creditsRemaining,
    });
    return {
      ok: count > 0,
      count,
      message:
        count > 0
          ? `✓ Scavio Google: ${count} תוצאות (${data.hits?.[0]?.title?.slice(0, 36) ?? "—"}…)`
          : "מחובר — 0 תוצאות לבדיקה",
    };
  } catch (e) {
    return { ok: false, count: 0, message: e instanceof Error ? e.message : "שגיאת רשת" };
  }
};

export { isScavioConfigured };
