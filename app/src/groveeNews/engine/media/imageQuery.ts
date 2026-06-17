// @ts-nocheck
const QUERY_STOP = new Set([
  "the",
  "a",
  "an",
  "in",
  "on",
  "at",
  "to",
  "for",
  "of",
  "and",
  "or",
  "is",
  "are",
  "was",
  "were",
  "after",
  "before",
  "with",
  "from",
  "as",
  "by",
  "new",
  "says",
  "say",
  "report",
  "reports",
  "breaking",
  "update",
  "live",
  "news",
  "photo",
  "image",
]);

const GENERIC_TAGS = new Set([
  "background",
  "wallpaper",
  "abstract",
  "pattern",
  "texture",
  "design",
  "banner",
  "template",
  "mockup",
  "icon",
  "logo",
]);

/** Build a short English stock-photo query from headline + optional topic hint. */
export function buildImageSearchQuery(title: string, hint = ""): string {
  const words = title
    .replace(/[^\w\s-]/g, " ")
    .split(/\s+/)
    .map((w) => w.trim())
    .filter((w) => w.length > 2 && !QUERY_STOP.has(w.toLowerCase()));

  const fromTitle = words.slice(0, 5).join(" ");
  const hintTrim = hint.trim();
  const combined = hintTrim && hintTrim.length > 1 ? `${hintTrim} ${fromTitle}` : fromTitle;
  return combined.trim().slice(0, 96);
}

function dedupeQueries(items: string[]): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const raw of items) {
    const q = raw.trim();
    if (!q) continue;
    const key = q.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(q);
  }
  return out;
}

/** English query variants — headline+hint, headline only, hint only. */
export function buildStockSearchQueries(title: string, hint = ""): string[] {
  return dedupeQueries([
    buildImageSearchQuery(title, hint),
    buildImageSearchQuery(title, ""),
    hint.trim(),
  ]);
}

export function tokenizeForImageMatch(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, " ")
    .split(/\s+/)
    .map((w) => w.trim())
    .filter((w) => w.length > 2 && !QUERY_STOP.has(w));
}

/** Score how well image tags/title match the search query (higher = better). */
export function scoreStockCandidate(query: string, tags: string, title = ""): number {
  const qTokens = tokenizeForImageMatch(query);
  if (!qTokens.length) return 0;

  const hay = `${tags} ${title}`.toLowerCase();
  const tagTokens = new Set(tokenizeForImageMatch(hay));
  let score = 0;

  for (const token of qTokens) {
    if (tagTokens.has(token)) score += 14;
    if (hay.includes(token)) score += 6;
  }

  for (const generic of GENERIC_TAGS) {
    if (tagTokens.has(generic) && qTokens.length > 1) score -= 8;
  }

  return score;
}
