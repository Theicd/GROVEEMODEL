// @ts-nocheck
import type { NewsFeedDef } from "../englishFeeds";

export type FeedLang = string;
export type FeedRegion = string;
export type FeedScope = "global" | "national" | "regional";
export type FeedTier = "core" | "extended";
export type FeedIdeology = "conservative" | "center-right";

export type CatalogFeed = NewsFeedDef & {
  lang: FeedLang;
  region: FeedRegion;
  scope: FeedScope;
  tier: FeedTier;
  ideology?: FeedIdeology;
};
