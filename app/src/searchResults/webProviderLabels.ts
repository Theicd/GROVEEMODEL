import type { SearchProviderId } from "../webSearch/types";
import { hostFromUrl } from "./sourceBranding";

const WEB_PROVIDER_LABEL: Partial<Record<SearchProviderId, { he: string; en: string }>> = {
  openserp: { he: "OpenSERP", en: "OpenSERP" },
  tavily: { he: "Tavily", en: "Tavily" },
  scavio: { he: "Scavio Google", en: "Scavio" },
  searxng: { he: "SearXNG", en: "SearXNG" },
};

export const webProviderLabel = (
  provider: SearchProviderId,
  uiLang: "he" | "en" = "he",
): string | null => {
  const row = WEB_PROVIDER_LABEL[provider];
  if (!row) return null;
  return uiLang === "he" ? row.he : row.en;
};

/** Site line for SERP row — includes search engine provider + optional SERP engine. */
export const webHitSourceLabel = (
  provider: SearchProviderId,
  url: string,
  serpEngine?: string,
): string => {
  const host = hostFromUrl(url);
  const parts = [webProviderLabel(provider) ?? "Web", serpEngine, host].filter(Boolean);
  return parts.join(" · ");
};

export const isCompanionWebHit = (provider: SearchProviderId): boolean => provider === "openserp";

export const isGenericWebHit = (kind: string, provider: SearchProviderId): boolean =>
  kind === "web" && !isCompanionWebHit(provider);
