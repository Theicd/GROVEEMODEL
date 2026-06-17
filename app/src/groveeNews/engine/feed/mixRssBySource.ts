// @ts-nocheck
import type { RssItem } from "../types";

/** Newest headline per RSS source, sorted by publish time (newest first). */
export function latestRssPerSource(items: RssItem[]): RssItem[] {
  const byKey = new Map<string, RssItem>();
  for (const item of items) {
    const prev = byKey.get(item.sourceKey);
    if (!prev || item.publishedTs > prev.publishedTs) byKey.set(item.sourceKey, item);
  }
  return [...byKey.values()].sort((a, b) => b.publishedTs - a.publishedTs);
}

/**
 * Round-robin mix so the live feed alternates sources instead of long runs from one RSS.
 */
export function mixRssBySource(items: RssItem[], maxItems?: number): RssItem[] {
  const buckets = new Map<string, RssItem[]>();
  for (const item of items) {
    const list = buckets.get(item.sourceKey) ?? [];
    list.push(item);
    buckets.set(item.sourceKey, list);
  }

  const keys = [...buckets.keys()].sort();
  const out: RssItem[] = [];
  let round = 0;

  while (!maxItems || out.length < maxItems) {
    let any = false;
    for (const key of keys) {
      const list = buckets.get(key)!;
      if (round < list.length) {
        out.push(list[round]);
        any = true;
        if (maxItems && out.length >= maxItems) break;
      }
    }
    if (!any) break;
    round++;
  }

  return out;
}
