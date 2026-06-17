// @ts-nocheck
import { extractArticleImageFromHtml, peekCachedArticleImage, resolveImageUrl } from "../extract/articleImage";
import { isBlockedArticleHost } from "../feeds/blockedArticleHosts";
import { hasRealImageUrl, normalizeImageUrl } from "./imageFields";

/** Pull hero URL from RSS description HTML without fetching the article page. */
export function extractImageFromDescription(description: string, pageUrl = ""): string {
  if (!description?.trim()) return "";
  const html = description;
  const og = html.match(/<img[^>]+src=["']([^"']+)["']/i);
  if (og?.[1]) {
    const resolved = pageUrl ? resolveImageUrl(og[1], pageUrl) : og[1].trim();
    if (resolved && !/pixel|tracker|1x1|spacer|blank\.gif/i.test(resolved)) return resolved;
  }
  return "";
}

export type ResolveImageInput = {
  articleUrl: string;
  rssImage?: string;
  description?: string;
};

/** Fast paths only — in-memory cache, RSS fields, description embed. No network. */
export function resolveWarmArticleImage(input: ResolveImageInput): string {
  const rss = normalizeImageUrl(input.rssImage);
  if (hasRealImageUrl(rss)) return rss;

  const cached = peekCachedArticleImage(input.articleUrl);
  if (cached && hasRealImageUrl(cached)) return cached;

  const fromDesc = extractImageFromDescription(input.description ?? "", input.articleUrl);
  if (fromDesc && hasRealImageUrl(fromDesc)) return fromDesc;

  return "";
}

export function canFetchArticlePageImage(articleUrl: string): boolean {
  return Boolean(articleUrl?.trim()) && !isBlockedArticleHost(articleUrl);
}

export { extractArticleImageFromHtml };
