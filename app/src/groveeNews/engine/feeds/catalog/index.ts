// @ts-nocheck
import { INTELLIGENCE_FEEDS } from "../englishFeeds";
import type { CatalogFeed } from "./types";
import { IL_HE_FEEDS } from "./regions/il-he";
import { WORLD_LANG_FEEDS } from "./world-lang-feeds";
import { WORLD_PERIPHERY_FEEDS } from "./world-periphery-feeds";

export type { CatalogFeed, FeedLang, FeedRegion, FeedScope, FeedTier } from "./types";
export { IL_HE_FEEDS } from "./regions/il-he";
export { WORLD_LANG_FEEDS } from "./world-lang-feeds";
export { WORLD_PERIPHERY_FEEDS } from "./world-periphery-feeds";

export function globalToCatalog(feed: (typeof INTELLIGENCE_FEEDS)[number]): CatalogFeed {
  const lang = feed.language === "zh" ? "zh" : feed.language === "multi" ? "multi" : "en";
  return {
    ...feed,
    lang,
    region: "GLOBAL",
    scope: "global",
    tier: "core",
  };
}

export const GLOBAL_CATALOG_FEEDS: CatalogFeed[] = INTELLIGENCE_FEEDS.map(globalToCatalog);

/** All feeds polled for every user — English core + multilingual world sources. */
export const ALL_CATALOG_FEEDS: CatalogFeed[] = (() => {
  const keys = new Set<string>();
  const out: CatalogFeed[] = [];
  for (const f of [...GLOBAL_CATALOG_FEEDS, ...WORLD_LANG_FEEDS, ...WORLD_PERIPHERY_FEEDS, ...IL_HE_FEEDS]) {
    if (keys.has(f.key)) continue;
    keys.add(f.key);
    out.push(f);
  }
  return out;
})();

/** @deprecated use ALL_CATALOG_FEEDS */
export const REGION_PACKS: Record<string, CatalogFeed[]> = {
  IL: IL_HE_FEEDS,
};
