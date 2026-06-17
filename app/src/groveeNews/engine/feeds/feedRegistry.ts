// @ts-nocheck
import { RSS_POLL_INTERVAL_MS } from "./englishFeeds";
import { ALL_CATALOG_FEEDS, type CatalogFeed } from "./catalog";
import { getUserNewsProfile, type UserNewsProfile } from "../settings/userNewsProfile";

export { RSS_POLL_INTERVAL_MS };

export type { CatalogFeed };

export const FEED_BY_KEY = Object.fromEntries(ALL_CATALOG_FEEDS.map((f) => [f.key, f])) as Record<
  string,
  CatalogFeed
>;

export function getFeedLang(sourceKey: string): string {
  const lang = FEED_BY_KEY[sourceKey]?.lang ?? "en";
  if (lang === "multi") return "en";
  return lang;
}

/** All catalog feeds — same list for every user; display language controls translation only. */
export function resolveActiveFeeds(profile: UserNewsProfile = getUserNewsProfile()): CatalogFeed[] {
  if (profile.pollTier === "full") return ALL_CATALOG_FEEDS;
  return ALL_CATALOG_FEEDS.filter((f) => f.tier === "core");
}
