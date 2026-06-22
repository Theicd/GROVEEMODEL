import { resolveActiveFeeds } from "./engine/feeds/feedRegistry";
import { fetchFeedItems, type FeedFetchMode } from "./engine/fetch/rssFeedFetch";
import { upsertRssItems } from "./engine/storage/db";

/** Poll a batch of catalog feeds until timeout — used to bootstrap RSS DB. */
export async function pollCatalogFeeds(opts: {
  timeoutMs: number;
  lang?: string;
  maxFeeds?: number;
  fetchMode?: FeedFetchMode;
  sessionStart?: number;
}): Promise<number> {
  const deadline = Date.now() + opts.timeoutMs;
  const fetchMode = opts.fetchMode ?? "live-first";
  const live = fetchMode === "live-first";
  let feeds = resolveActiveFeeds();
  if (opts.lang) feeds = feeds.filter((f) => f.lang === opts.lang);
  if (opts.maxFeeds) feeds = feeds.slice(0, opts.maxFeeds);

  let totalAdded = 0;
  for (const feed of feeds) {
    if (Date.now() >= deadline) break;
    try {
      const items = await fetchFeedItems(feed, fetchMode);
      totalAdded += await upsertRssItems(items, {
        fetchedAt: opts.sessionStart ?? Date.now(),
        live,
      });
    } catch {
      /* skip single feed failure */
    }
    if (Date.now() >= deadline) break;
    await new Promise((r) => setTimeout(r, 120));
  }
  return totalAdded;
}

/** Poll Hebrew RSS sources first — used before Hebrew UI news search. */
export async function priorityPollHebrewFeeds(opts: {
  timeoutMs: number;
  fetchMode?: FeedFetchMode;
  sessionStart?: number;
}): Promise<number> {
  return pollCatalogFeeds({
    timeoutMs: opts.timeoutMs,
    lang: "he",
    fetchMode: opts.fetchMode ?? "live-first",
    sessionStart: opts.sessionStart,
  });
}
