import { fetchJson } from "../../webSearch/fetchJson";
import type { PluginHealthResult } from "../types";
import { resolveSearchCompanionFetchBase, getSearchCompanionServiceUrl, getSearchCompanionUrl, setSearchCompanionUrl } from "./companionSettings";
import { SEARCH_COMPANION_MANIFEST } from "./manifest";

type OpenSerpHealth = {
  status?: string;
  version?: string;
  engines?: Record<string, { status?: string }>;
};

let healthCache: (PluginHealthResult & { checkedAt: number }) | null = null;
const HEALTH_TTL_MS = 8_000;

export const getSearchCompanionHealthCache = (): (PluginHealthResult & { checkedAt: number }) | null => {
  if (!healthCache) return null;
  if (Date.now() - healthCache.checkedAt > HEALTH_TTL_MS * 4) return healthCache;
  return healthCache;
};

export const isSearchCompanionReachable = (): boolean => {
  const c = healthCache;
  if (!c) return false;
  if (Date.now() - c.checkedAt > HEALTH_TTL_MS * 4) return false;
  return c.status === "online" || c.status === "degraded";
};

export const probeSearchCompanionHealth = async (): Promise<PluginHealthResult> => {
  const started = performance.now();
  const base = resolveSearchCompanionFetchBase();
  const url = `${base}/health`;

  try {
    const data = await fetchJson<OpenSerpHealth>(url, undefined, { timeoutMs: 4_000 });
    const raw = (data.status ?? "unknown").toLowerCase();
    const enginesReady: string[] = [];
    const enginesFailed: string[] = [];
    if (Array.isArray(data.engines)) {
      for (const row of data.engines) {
        const name = typeof row === "object" && row && "name" in row ? String((row as { name?: string }).name) : "";
        const st = typeof row === "object" && row && "status" in row ? String((row as { status?: string }).status).toLowerCase() : "";
        if (!name) continue;
        if (st === "ready" || st === "healthy" || st === "ok") enginesReady.push(name);
        else enginesFailed.push(name);
      }
    } else if (data.engines && typeof data.engines === "object") {
      for (const [name, info] of Object.entries(data.engines)) {
        const st = (info as { status?: string })?.status?.toLowerCase() ?? "";
        if (st === "ready" || st === "healthy" || st === "ok") enginesReady.push(name);
        else if (st) enginesFailed.push(name);
      }
    }

    let status: PluginHealthResult["status"] = "offline";
    if (raw === "healthy") status = "online";
    else if (raw === "degraded") status = "degraded";
    else if (raw === "unhealthy") status = "offline";

    const latencyMs = Math.round(performance.now() - started);
    const messageHe =
      status === "online"
        ? `מנוע חיפוש פעיל על ${getSearchCompanionServiceUrl()}`
        : status === "degraded"
          ? "פועל — חלק מהמנועים לא זמינים"
          : "לא מגיב — הפעל «Grove Search» משולחן העבודה";

    const result: PluginHealthResult = {
      status,
      messageHe,
      version: data.version,
      enginesReady: enginesReady.length ? enginesReady : undefined,
      enginesFailed: enginesFailed.length ? enginesFailed : undefined,
      latencyMs,
    };

    healthCache = { ...result, checkedAt: Date.now() };

    if (status === "online" || status === "degraded") {
      if (!getSearchCompanionUrl()) {
        setSearchCompanionUrl(SEARCH_COMPANION_MANIFEST.defaultBaseUrl);
      }
    }

    return result;
  } catch (err) {
    const messageHe =
      err instanceof Error && /fetch|Failed|HTTP|Timeout/i.test(err.message)
        ? "לא מגיב — הורד, התקן והפעל את Grove Search Companion"
        : "שגיאת חיבור למנוע החיפוש המקומי";
    const result: PluginHealthResult = {
      status: "offline",
      messageHe,
      latencyMs: Math.round(performance.now() - started),
    };
    healthCache = { ...result, checkedAt: Date.now() };
    return result;
  }
};

export const resetSearchCompanionHealthCache = (): void => {
  healthCache = null;
};
