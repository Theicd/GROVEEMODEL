import { LIVE_MEDIA_CATEGORIES } from "./catalogs";
import { isRadioFrequencyQuery } from "./mediaIntent";

const CHANNEL_HINTS =
  /(?:ערוץ|channel|now\s*\d+|כאן\s*\d+|ער\s*14|now14|ch\s*\d+|\b14\b|\b11\b|\b12\b|\b13\b)/i;

export function resolveCategoryFromQuery(query: string): string | null {
  const q = query.trim().toLowerCase();
  if (!q) return null;
  for (const cat of LIVE_MEDIA_CATEGORIES) {
    if (q === cat.id || q.includes(cat.id)) return cat.id;
    if (q.includes(cat.name.toLowerCase())) return cat.id;
    if (cat.nameHe && q.includes(cat.nameHe)) return cat.id;
  }
  return null;
}

export function isChannelNameQuery(query: string): boolean {
  return CHANNEL_HINTS.test(query.trim());
}

export function extractChannelDigits(query: string): string | null {
  if (isRadioFrequencyQuery(query)) return null;
  const m = query.match(/(?:now|כאן|ערוץ|channel)?\s*(\d{1,3})/i);
  return m?.[1] ?? null;
}

export function expandLiveMediaSearchTerms(query: string): string[] {
  const terms = new Set<string>([query.trim()]);
  const cat = resolveCategoryFromQuery(query);
  if (cat) {
    terms.add(cat);
    const meta = LIVE_MEDIA_CATEGORIES.find((c) => c.id === cat);
    if (meta) {
      terms.add(meta.name);
      terms.add(meta.nameHe);
    }
  }
  const digits = extractChannelDigits(query);
  if (digits) terms.add(digits);
  return [...terms].filter(Boolean);
}
