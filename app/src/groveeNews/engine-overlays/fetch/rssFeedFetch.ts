// @ts-nocheck
import type { CatalogFeed } from "../feeds/feedRegistry";
import { parseRssXml } from "../rss/parser";
import type { RssItem } from "../types";
import { getCachedRssXml } from "./rssCache";
import { currentFetchContext, fetchRemoteText, isStaticWebHost } from "./remoteFetch";

export type FeedFetchMode = "live-first" | "cache-first";

export async function fetchFeedItems(
  feed: CatalogFeed,
  mode: FeedFetchMode = "cache-first",
): Promise<RssItem[]> {
  const urls = [feed.url, ...(feed.fallbackUrls ?? [])];
  let lastErr: unknown;

  const tryCache = async (): Promise<RssItem[] | null> => {
    const cached = await getCachedRssXml(urls);
    if (!cached?.trim()) return null;
    try {
      return parseRssXml(cached, {
        source: feed.label,
        sourceKey: feed.key,
        category: feed.category,
      });
    } catch (err) {
      lastErr = err;
      return null;
    }
  };

  const tryNetwork = async (): Promise<RssItem[] | null> => {
    const ctx = currentFetchContext();
    if (isStaticWebHost() && !ctx.proxyUrl) return null;
    for (const url of urls) {
      try {
        const xml = await fetchRemoteText(url);
        return parseRssXml(xml, {
          source: feed.label,
          sourceKey: feed.key,
          category: feed.category,
        });
      } catch (err) {
        lastErr = err;
      }
    }
    return null;
  };

  if (mode === "live-first") {
    const live = await tryNetwork();
    if (live?.length) return live;
    const cached = await tryCache();
    if (cached?.length) return cached;
    throw lastErr instanceof Error ? lastErr : new Error("Feed fetch failed");
  }

  const cached = await tryCache();
  if (cached?.length) return cached;

  const ctx = currentFetchContext();
  if (isStaticWebHost() && !ctx.proxyUrl) {
    throw lastErr instanceof Error ? lastErr : new Error("RSS cache miss on static host");
  }

  const live = await tryNetwork();
  if (live?.length) return live;
  throw lastErr instanceof Error ? lastErr : new Error("Feed fetch failed");
}
