// @ts-nocheck
import type { FeedItem } from "./buildFeed";

export function slicePage(items: FeedItem[], offset: number, pageSize: number) {
  const page = items.slice(offset, offset + pageSize);
  return {
    items: page,
    hasMore: items.length > offset + pageSize,
    nextOffset: offset + pageSize,
  };
}
