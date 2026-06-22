import { SEARCH_COMPANION_MANIFEST } from "./manifest";

const STORAGE_KEY = "grovee_plugin_search_companion_url";

export const isLocalOpenSerpUrl = (url: string): boolean => {
  try {
    const u = new URL(url);
    const port = u.port || (u.protocol === "https:" ? "443" : "80");
    const defaultPort = String(SEARCH_COMPANION_MANIFEST.defaultPort ?? 7000);
    return (
      (u.hostname === "127.0.0.1" || u.hostname === "localhost") &&
      (port === defaultPort || port === "7000")
    );
  } catch {
    return false;
  }
};

export const getSearchCompanionUrl = (): string => {
  const env = import.meta.env.VITE_OPENSERP_URL as string | undefined;
  if (env?.trim()) return env.trim().replace(/\/$/, "");
  if (typeof window === "undefined") return "";
  const saved = window.localStorage.getItem(STORAGE_KEY)?.trim();
  return saved ? saved.replace(/\/$/, "") : "";
};

/** Human-facing service address (plugins panel, errors). */
export const getSearchCompanionServiceUrl = (): string =>
  getSearchCompanionUrl() || SEARCH_COMPANION_MANIFEST.defaultBaseUrl;

export const setSearchCompanionUrl = (url: string): void => {
  if (typeof window === "undefined") return;
  const clean = url.trim().replace(/\/$/, "");
  if (!clean) {
    window.localStorage.removeItem(STORAGE_KEY);
    return;
  }
  window.localStorage.setItem(STORAGE_KEY, clean);
};

/**
 * Base URL for fetch.
 * In dev, local OpenSERP (127.0.0.1:7000) must go through /api/openserp — direct browser
 * calls are blocked by CORS and show a raw URL in timeout errors.
 */
export const resolveSearchCompanionFetchBase = (): string => {
  const configured = getSearchCompanionUrl();

  if (import.meta.env.DEV && typeof window !== "undefined") {
    if (!configured || isLocalOpenSerpUrl(configured)) {
      return `${window.location.origin}/api/openserp`;
    }
  }

  if (configured) return configured;

  if (import.meta.env.DEV && typeof window !== "undefined") {
    return `${window.location.origin}/api/openserp`;
  }

  return SEARCH_COMPANION_MANIFEST.defaultBaseUrl;
};

export const usesDevOpenSerpProxy = (): boolean => {
  if (!import.meta.env.DEV || typeof window === "undefined") return false;
  const base = resolveSearchCompanionFetchBase();
  return base.endsWith("/api/openserp");
};
