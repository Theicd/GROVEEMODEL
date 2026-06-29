import type { Channel } from "./types";
import { inferChannelLanguages, languageDisplayLabel } from "./languageMetadata";

/** User-facing channel categories for favorites, tuner, and TV guide. */
export type UserChannelCategory =
  | "movies"
  | "series"
  | "sports"
  | "reality"
  | "kids"
  | "news"
  | "music"
  | "documentary"
  | "general";

export type ViewLanguageCode =
  | "heb"
  | "eng"
  | "rus"
  | "ara"
  | "fra"
  | "deu"
  | "spa"
  | "und";

export type ChannelUserOverride = {
  displayName?: string;
  category?: UserChannelCategory;
  broadcastLanguage?: ViewLanguageCode;
  /** User-provided logo/thumbnail URL (http/https). */
  imageUrl?: string;
  /** User-provided stream URL (http/https) — replaces catalog source when set. */
  streamUrl?: string;
};

export const USER_CHANNEL_CATEGORIES: {
  id: UserChannelCategory;
  nameHe: string;
  nameEn: string;
}[] = [
  { id: "news", nameHe: "חדשות", nameEn: "News" },
  { id: "sports", nameHe: "ספורט", nameEn: "Sports" },
  { id: "movies", nameHe: "סרטים", nameEn: "Movies" },
  { id: "series", nameHe: "סדרות", nameEn: "Series" },
  { id: "reality", nameHe: "ריאלטי", nameEn: "Reality" },
  { id: "kids", nameHe: "ילדים", nameEn: "Kids" },
  { id: "music", nameHe: "מוסיקה", nameEn: "Music" },
  { id: "documentary", nameHe: "תיעודי", nameEn: "Documentary" },
  { id: "general", nameHe: "כללי", nameEn: "General" },
];

export const VIEW_LANGUAGE_OPTIONS: { code: ViewLanguageCode; nameHe: string; nameEn: string }[] = [
  { code: "heb", nameHe: "עברית", nameEn: "Hebrew" },
  { code: "eng", nameHe: "אנגלית", nameEn: "English" },
  { code: "rus", nameHe: "רוסית", nameEn: "Russian" },
  { code: "ara", nameHe: "ערבית", nameEn: "Arabic" },
  { code: "fra", nameHe: "צרפתית", nameEn: "French" },
  { code: "deu", nameHe: "גרמנית", nameEn: "German" },
  { code: "spa", nameHe: "ספרדית", nameEn: "Spanish" },
];

export const ALL_USER_CATEGORIES: UserChannelCategory[] = USER_CHANNEL_CATEGORIES.map((c) => c.id);

export function categoryLabelHe(id: UserChannelCategory): string {
  return USER_CHANNEL_CATEGORIES.find((c) => c.id === id)?.nameHe ?? id;
}

export function categoryLabelEn(id: UserChannelCategory): string {
  return USER_CHANNEL_CATEGORIES.find((c) => c.id === id)?.nameEn ?? id;
}

export function defaultViewLanguagesForCountry(countryCode: string): ViewLanguageCode[] {
  const cc = countryCode.trim().toLowerCase();
  if (cc === "il") return ["heb", "eng", "rus"];
  if (cc === "ru" || cc === "ua") return ["rus", "eng"];
  if (cc === "us" || cc === "gb" || cc === "au" || cc === "ca") return ["eng"];
  if (cc === "de" || cc === "at" || cc === "ch") return ["deu", "eng"];
  if (cc === "fr") return ["fra", "eng"];
  if (cc === "es" || cc === "mx") return ["spa", "eng"];
  return ["eng", "heb"];
}

const REALITY_HINT =
  /\b(reality|real\s*housewives|survivor|big\s*brother|love\s*island|bachelor|ראליטי|האח\s*הגדול)\b/i;
const NEWS_HINT =
  /\b(news|חדשות|i24|cnn|bbc\s*news|sky\s*news|fox\s*news|msnbc|ערוץ\s*14|now\s*14)\b/i;
const KIDS_HINT = /\b(kids|children|cartoon|disney|nickelodeon|ילדים|ניקלודיאון|ג'וניור)\b/i;
const SPORTS_HINT = /\b(sport|sports|espn|nba|nfl|fifa|football|ספורט|one\s*football)\b/i;
const MOVIES_HINT = /\b(movie|movies|cinema|film|סרט|סרטים|moviephere|hbo)\b/i;
const MUSIC_HINT = /\b(music|mtv|vh1|vevo|stingray|מוזיקה|מוסיקה)\b/i;
const DOCUMENTARY_HINT = /\b(documentary|docu|תיעוד|discovery|nat\s*geo|history\s*channel)\b/i;

function haystack(c: Pick<Channel, "name" | "category" | "tags" | "groupTitle">): string {
  return [c.name, c.category, c.groupTitle, ...(c.tags ?? [])].filter(Boolean).join(" ");
}

/** Map IPTV catalog category + name hints → user category. */
export function mapCatalogCategoryToUser(c: Pick<Channel, "name" | "category" | "tags" | "groupTitle" | "country" | "tvgId">): UserChannelCategory {
  const cat = (c.category || "general").toLowerCase();
  const text = haystack(c).toLowerCase();

  if (cat === "movies" || MOVIES_HINT.test(text)) return "movies";
  if (cat === "music" || MUSIC_HINT.test(text)) return "music";
  if (cat === "documentary" || DOCUMENTARY_HINT.test(text)) return "documentary";
  if (cat === "sports" || SPORTS_HINT.test(text)) return "sports";
  if (cat === "kids" || KIDS_HINT.test(text)) return "kids";
  if (cat === "news" || NEWS_HINT.test(text)) return "news";
  if (REALITY_HINT.test(text)) return "reality";
  if (cat === "general") return "general";

  if (cat === "entertainment" || cat === "comedy" || cat === "series") {
    return "series";
  }

  if (c.country === "il" || /\.il@/i.test(c.tvgId ?? "")) {
    if (NEWS_HINT.test(text)) return "news";
    return "general";
  }

  return "general";
}

export function normalizeViewLanguageCode(raw: string | undefined): ViewLanguageCode | null {
  if (!raw?.trim()) return null;
  const t = raw.trim().toLowerCase();
  const map: Record<string, ViewLanguageCode> = {
    he: "heb",
    heb: "heb",
    en: "eng",
    eng: "eng",
    ru: "rus",
    rus: "rus",
    ar: "ara",
    ara: "ara",
    fr: "fra",
    fra: "fra",
    de: "deu",
    deu: "deu",
    es: "spa",
    spa: "spa",
    und: "und",
  };
  return map[t] ?? null;
}

/** Israeli broadcast (Kan, Reshet, Now 14, iptv-org-il, etc.). */
export function isIsraeliChannel(
  c: Pick<Channel, "country" | "tvgId" | "source" | "name">,
): boolean {
  if (c.country === "il") return true;
  if (c.source === "iptv-org-il") return true;
  if (/\.il@/i.test(c.tvgId ?? "")) return true;
  if (/\b(kan\s*11|now\s*14|reshet\s*13|ערוץ\s*1[234])\b/i.test(c.name)) return true;
  return false;
}

/** Auto-detect broadcast language from catalog metadata (no user override). */
export function defaultBroadcastLanguageForChannel(c: Channel): ViewLanguageCode {
  if (isIsraeliChannel(c)) return "heb";

  const langs = c.languages?.length ? c.languages : inferChannelLanguages(c);
  const priority: ViewLanguageCode[] = ["eng", "rus", "ara", "fra", "deu", "spa"];
  for (const p of priority) {
    if (langs.some((l) => normalizeViewLanguageCode(l) === p)) return p;
  }
  for (const code of langs) {
    const norm = normalizeViewLanguageCode(code);
    if (norm && norm !== "und") return norm;
  }

  const tvg = c.tvgId ?? "";
  if (/\.(us|uk|gb|au|ca)@/i.test(tvg)) return "eng";
  if (c.country === "us" || c.country === "gb" || c.country === "au" || c.country === "ca") return "eng";
  if (c.country === "ru") return "rus";
  if (c.country === "fr") return "fra";
  if (c.country === "de" || c.country === "at" || c.country === "ch") return "deu";
  if (c.country === "es" || c.country === "mx") return "spa";

  return "eng";
}

export function resolveBroadcastLanguage(
  c: Channel,
  override?: ChannelUserOverride,
): ViewLanguageCode {
  const auto = defaultBroadcastLanguageForChannel(c);
  const fromOverride = override?.broadcastLanguage;
  if (fromOverride && fromOverride !== "und") {
    // Drop mistaken Hebrew overrides on international channels (from old modal default).
    if (fromOverride === "heb" && !isIsraeliChannel(c) && auto === "eng") return auto;
    return fromOverride;
  }

  const langs = c.languages?.length ? c.languages : c.language ? [c.language] : [];
  for (const code of langs) {
    const norm = normalizeViewLanguageCode(code);
    if (norm && norm !== "und") return norm;
  }
  return auto;
}

export function resolveUserCategory(
  channelId: string,
  c: Channel,
  overrides?: Record<string, ChannelUserOverride>,
): UserChannelCategory {
  const o = overrides?.[channelId];
  if (o?.category) return o.category;
  return mapCatalogCategoryToUser(c);
}

export function resolveDisplayName(
  channelId: string,
  c: Channel,
  overrides?: Record<string, ChannelUserOverride>,
): string {
  const custom = overrides?.[channelId]?.displayName?.trim();
  return custom || c.name;
}

/** Accept http(s) image URLs only. */
export function normalizeChannelImageUrl(raw: string | undefined): string | undefined {
  const t = raw?.trim();
  if (!t) return undefined;
  try {
    const u = new URL(t);
    if (u.protocol !== "http:" && u.protocol !== "https:") return undefined;
    return u.href;
  } catch {
    return undefined;
  }
}

/** Accept http(s) stream URLs only. */
export function normalizeChannelStreamUrl(raw: string | undefined): string | undefined {
  const t = raw?.trim();
  if (!t) return undefined;
  try {
    const u = new URL(t);
    if (u.protocol !== "http:" && u.protocol !== "https:") return undefined;
    return u.href;
  } catch {
    return undefined;
  }
}

export function resolveChannelImageUrl(
  channelId: string,
  c: Channel,
  overrides?: Record<string, ChannelUserOverride>,
): string | undefined {
  const custom = normalizeChannelImageUrl(overrides?.[channelId]?.imageUrl);
  if (custom) return custom;
  return normalizeChannelImageUrl(c.logo) || undefined;
}

export function resolveChannelStreamUrl(
  channelId: string,
  c: Channel,
  overrides?: Record<string, ChannelUserOverride>,
): string {
  const custom = normalizeChannelStreamUrl(overrides?.[channelId]?.streamUrl);
  if (custom) return custom;
  return normalizeChannelStreamUrl(c.stream) || c.stream;
}

export function buildChannelCardSnippet(
  c: Channel,
  opts: {
    category: UserChannelCategory;
    broadcastLanguage: ViewLanguageCode;
    he?: boolean;
    statusLabel: string;
    score: number;
  },
): string {
  const he = opts.he !== false;
  const langLabel =
    opts.broadcastLanguage === "und"
      ? he
        ? "שפה לא ידועה"
        : "Unknown lang"
      : languageDisplayLabel(opts.broadcastLanguage, he);
  const catLabel = he ? categoryLabelHe(opts.category) : categoryLabelEn(opts.category);
  const parts = [langLabel, catLabel, opts.statusLabel, he ? `ניקוד ${opts.score}` : `score ${opts.score}`];
  return parts.join(" · ");
}
