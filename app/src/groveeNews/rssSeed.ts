import { getRssHeadlineCount } from "./engine/storage/db";
import { pollCatalogFeeds, priorityPollHebrewFeeds } from "./hebrewFeedPoll";

/** Bootstrap empty RSS DB at engine boot — live poll only. */
export async function ensureRssCatalogReady(timeoutMs = 14_000): Promise<number> {
  const count = await getRssHeadlineCount();
  if (count > 0) return count;

  await priorityPollHebrewFeeds({
    timeoutMs: Math.min(timeoutMs, 10_000),
    fetchMode: "live-first",
    sessionStart: Date.now(),
  });
  await pollCatalogFeeds({
    timeoutMs: Math.min(timeoutMs, 12_000),
    maxFeeds: 24,
    fetchMode: "live-first",
    sessionStart: Date.now(),
  });
  return getRssHeadlineCount();
}
