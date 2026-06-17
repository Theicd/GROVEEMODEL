import type { GroveeNewsCard } from "./types";

function storyKey(card: GroveeNewsCard): string {
  const url = card.url?.split("?")[0]?.toLowerCase().trim();
  if (url) return `u:${url}`;
  const title = (card.titleOriginal || card.title).toLowerCase().replace(/\s+/g, " ").trim();
  return `t:${title.slice(0, 120)}`;
}

/** Drop duplicate URLs / near-duplicate titles in panel cards. */
export function dedupeNewsCards(cards: GroveeNewsCard[]): GroveeNewsCard[] {
  const seen = new Set<string>();
  const out: GroveeNewsCard[] = [];
  for (const card of cards) {
    const key = storyKey(card);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(card);
  }
  return out;
}
