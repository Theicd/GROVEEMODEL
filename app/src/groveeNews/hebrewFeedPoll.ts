import { resolveActiveFeeds, type CatalogFeed } from "./engine/feeds/feedRegistry";
import { getCachedRssXml } from "./engine/fetch/rssCache";
import { fetchRemoteText, isStaticWebHost, currentFetchContext } from "./engine/fetch/remoteFetch";
import { parseRssXml } from "./engine/rss/parser";
import { upsertRssItems } from "./engine/storage/db";
import type { RssItem } from "./engine/types";

async function fetchFeedItems(feed: CatalogFeed): Promise<RssItem[]> {
  const urls = [feed.url, ...(feed.fallbackUrls ?? [])];
  let lastErr: unknown;

  const cached = await getCachedRssXml(urls);
  if (cached) {
    try {
      return parseRssXml(cached, {
        source: feed.label,
        sourceKey: feed.key,
        category: feed.category,
      });
    } catch (err) {
      lastErr = err;
    }
  }

  const ctx = currentFetchContext();
  if (isStaticWebHost() && !ctx.proxyUrl) {
    throw lastErr instanceof Error ? lastErr : new Error("RSS cache miss on static host");
  }

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
  throw lastErr instanceof Error ? lastErr : new Error("Feed fetch failed");
}

/** Poll Hebrew RSS sources first — used before Hebrew UI news search. */
export async function priorityPollHebrewFeeds(opts: { timeoutMs: number }): Promise<number> {
  const deadline = Date.now() + opts.timeoutMs;
  const feeds = resolveActiveFeeds().filter((f) => f.lang === "he");
  let totalAdded = 0;

  for (const feed of feeds) {
    if (Date.now() >= deadline) break;
    try {
      const items = await fetchFeedItems(feed);
      totalAdded += await upsertRssItems(items);
    } catch {
      /* skip single feed failure */
    }
    if (Date.now() >= deadline) break;
    await new Promise((r) => setTimeout(r, 180));
  }

  return totalAdded;
}
