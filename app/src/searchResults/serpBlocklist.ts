import type { GroveeNewsCard } from "../groveeNews/types";
import type { UnifiedSearchHit } from "./types";

const BLOCKED_URL_PARTS = [
  "theverge.com/gadgets/950958",
  "theverge.com/gadgets/950929",
  "ghanaweb.com",
];

const BLOCKED_URL_PATTERNS = [
  /theverge\.com\/gadgets\/\d+\/[^/?#]*(?:deal|deals|sale|discount|gift-guide|fathers-day|father-day|mothers-day|prime-day|black-friday|cyber-monday|promo)/i,
];

const BLOCKED_TITLE_PATTERNS = [
  /^פרסומות$/i,
  /^advertisements?$/i,
  /^promotions?$/i,
  /^sponsored$/i,
  /^ghanaweb$/i,
  /calvin\s+and\s+hobbes/i,
  /father'?s\s+day\s+gift/i,
  /last-?minute\s+(?:father'?s|mother'?s)\s+day/i,
  /writes\s+for\s+wired.*digital\s+trends/i,
  /tech\s+journalist\s+from\s+portland/i,
];

const normalizeUrl = (url: string): string => {
  try {
    const u = new URL(url);
    return `${u.hostname.replace(/^www\./, "")}${u.pathname}`.toLowerCase();
  } catch {
    return url.toLowerCase().split("?")[0] ?? "";
  }
};

const titleBlob = (title: string, titleOriginal?: string): string =>
  `${titleOriginal ?? ""} ${title}`.trim().toLowerCase();

export function isBlockedSerpUrl(url: string): boolean {
  const path = normalizeUrl(url);
  if (!path) return false;
  if (BLOCKED_URL_PARTS.some((part) => path.includes(part))) return true;
  return BLOCKED_URL_PATTERNS.some((re) => re.test(path));
}

export function isBlockedSerpTitle(title: string, titleOriginal?: string): boolean {
  const blob = titleBlob(title, titleOriginal);
  if (!blob) return false;
  return BLOCKED_TITLE_PATTERNS.some((re) => re.test(blob));
}

export function isBlockedSerpHit(hit: Pick<UnifiedSearchHit, "url" | "title" | "titleOriginal" | "sourceLabel" | "sourceKey">): boolean {
  if (hit.sourceKey?.toLowerCase() === "gh_ghanaweb") return true;
  if (/ghanaweb/i.test(hit.sourceLabel ?? "")) return true;
  return isBlockedSerpUrl(hit.url) || isBlockedSerpTitle(hit.title, hit.titleOriginal);
}

export function isBlockedNewsCard(card: Pick<GroveeNewsCard, "url" | "title" | "titleOriginal" | "source" | "sourceKey">): boolean {
  if (card.sourceKey?.toLowerCase() === "gh_ghanaweb") return true;
  if (/ghanaweb/i.test(card.source ?? "")) return true;
  return isBlockedSerpUrl(card.url) || isBlockedSerpTitle(card.title, card.titleOriginal);
}

export function filterBlockedHits(hits: UnifiedSearchHit[]): UnifiedSearchHit[] {
  return hits.filter((hit) => !isBlockedSerpHit(hit));
}

export function filterBlockedNewsCards<T extends GroveeNewsCard>(cards: T[]): T[] {
  return cards.filter((card) => !isBlockedNewsCard(card));
}
