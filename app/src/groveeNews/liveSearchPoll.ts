import { queryHasHebrew } from "./hebrewSearchTerms";
import { fetchFeedItems, type FeedFetchMode } from "./engine/fetch/rssFeedFetch";
import { resolveActiveFeeds, type CatalogFeed } from "./engine/feeds/feedRegistry";
import { upsertRssItems } from "./engine/storage/db";
import { getUserNewsProfile } from "./engine/settings/userNewsProfile";

export type LiveSearchPollResult = {
  sessionStart: number;
  feedsOk: number;
  feedsFailed: number;
  newHeadlines: number;
  feedsPolled: number;
};

async function pollFeedsBatch(
  feeds: CatalogFeed[],
  opts: { timeoutMs: number; fetchMode: FeedFetchMode; sessionStart: number },
): Promise<{ ok: number; failed: number; added: number }> {
  const deadline = Date.now() + opts.timeoutMs;
  let ok = 0;
  let failed = 0;
  let added = 0;

  for (const feed of feeds) {
    if (Date.now() >= deadline) break;
    try {
      const items = await fetchFeedItems(feed, opts.fetchMode);
      added += await upsertRssItems(items, { fetchedAt: opts.sessionStart, live: opts.fetchMode === "live-first" });
      ok++;
    } catch {
      failed++;
    }
    if (Date.now() >= deadline) break;
    await new Promise((r) => setTimeout(r, 120));
  }

  return { ok, failed, added };
}

/** Poll RSS feeds before search — live-first when network is available. */
export async function pollRssForLiveSearch(
  query: string,
  timeoutMs = 28_000,
): Promise<LiveSearchPollResult> {
  const sessionStart = Date.now();
  const heUi = getUserNewsProfile().uiLanguage === "he";
  const preferHe = heUi || queryHasHebrew(query);

  let feeds = resolveActiveFeeds();
  const heFeeds = preferHe ? feeds.filter((f) => f.lang === "he") : [];
  const worldFeeds = preferHe ? feeds.filter((f) => f.lang !== "he") : feeds;

  const heBudget = preferHe ? Math.round(timeoutMs * 0.55) : 0;
  const worldBudget = timeoutMs - heBudget;

  let feedsOk = 0;
  let feedsFailed = 0;
  let newHeadlines = 0;
  let feedsPolled = 0;

  if (heFeeds.length && heBudget > 0) {
    const batch = heFeeds.slice(0, 48);
    const r = await pollFeedsBatch(batch, {
      timeoutMs: heBudget,
      fetchMode: "live-first",
      sessionStart,
    });
    feedsOk += r.ok;
    feedsFailed += r.failed;
    newHeadlines += r.added;
    feedsPolled += r.ok + r.failed;
  }

  if (worldBudget > 0 && worldFeeds.length) {
    const batch = worldFeeds.slice(0, preferHe ? 40 : 64);
    const r = await pollFeedsBatch(batch, {
      timeoutMs: worldBudget,
      fetchMode: "live-first",
      sessionStart,
    });
    feedsOk += r.ok;
    feedsFailed += r.failed;
    newHeadlines += r.added;
    feedsPolled += r.ok + r.failed;
  }

  return { sessionStart, feedsOk, feedsFailed, newHeadlines, feedsPolled };
}
