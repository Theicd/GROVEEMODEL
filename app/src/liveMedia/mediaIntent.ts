import { resolveCategoryFromQuery } from "./queryMatch";

/** FM/AM dial frequency: "103 FM", "91.8 MHz", "fm 103". */
export const isRadioFrequencyQuery = (text: string): boolean => {
  const q = text.trim();
  return (
    /\b\d{2,3}(?:\.\d{1,2})?\s*(?:fm|mhz|am|khz)\b/i.test(q) ||
    /\b(?:fm|am)\s*\d{2,3}(?:\.\d{1,2})?\b/i.test(q)
  );
};

/** User wants internet radio (not TV). */
export const isRadioMediaQuery = (text: string): boolean => {
  const q = text.trim();
  if (!q) return false;
  if (isRadioFrequencyQuery(q)) return true;
  if (/(?:רדיו|radio\s+stations?|radio\s+browser|תחנ(?:ה|ת)\s+רדיו|גלגלצ|גלגל|כאן\s*(?:תרבות|ב|ג)|eco\s*99)/i.test(q)) {
    return true;
  }
  if (/(?:train\s+station|bus\s+station|metro\s+station|תחנ(?:ת|ה)\s+רכבת)/i.test(q)) return false;
  if (/\bfm\b|\bam\b/i.test(q) && !/(?:טל(?:ו)?יזיה|tv\s*live|live\s*tv|iptv|ערוץ\s*\d)/i.test(q)) {
    return true;
  }
  return false;
};

/** User wants live TV (not radio). */
export const isTvMediaQuery = (text: string): boolean => {
  const q = text.trim();
  if (!q) return false;
  if (isRadioFrequencyQuery(q)) return false;
  if (isRadioMediaQuery(q) && !/(?:טל(?:ו)?יזיה|טלויזיה|tv|ערוץ|channel|iptv)/i.test(q)) {
    return false;
  }
  if (/(?:טל(?:ו)?יזיה|טלויזיה|tv\s*live|live\s*tv|iptv|ערוץ|channel|כאן\s*\d+|now\s*\d+)/i.test(q)) {
    return true;
  }
  if (/(?:ערוץ\s*\d+|now\s*\d+|כאן\s*\d+)/i.test(q)) return true;
  return false;
};

/** Live TV channel browse by category — not TMDB/OMDb movie plots. */
export function isLiveTvCategoryChannelQuery(text: string): boolean {
  const q = text.trim();
  return (
    /(?:ערוץ|ערוצים|channels?)\s*(?:של|for)?\s*(?:סרט|סרטים|movies?|cinema|ילדים|kids|children|ספורט|sport|חדשות|news|מוזיקה|music|קומדיה|comedy|documentary|אנימ)/i.test(
      q,
    ) ||
    /(?:סרט|סרטים|movies?|cinema|kids|children|ילדים|ספורט|sport|חדשות|news|music|מוזיקה|comedy|קומדיה).*(?:ערוץ|ערוצים|channels?|live\s*tv|טל(?:ו)?יזיה|שידור\s*חי)/i.test(
      q,
    ) ||
    /(?:חפש|מצא|הראה|תן|show|search|find)\s+.*(?:ערוץ|ערוצים).*(?:סרט|סרטים|movies?|kids|ילדים|ספורט|sport|חדשות|news)/i.test(
      q,
    )
  );
}

/**
 * Queries that should use the local TV/radio catalog only — no web news/GitHub/OMDb.
 * Includes "חפש ערוץ סרטים", radio stations, sports package, etc.
 */
export function isLiveMediaCatalogQuery(text: string): boolean {
  const q = text.trim();
  if (!q) return false;
  if (isRadioMediaQuery(q) || isRadioBrowseQuery(q)) return true;
  if (isLiveTvCategoryChannelQuery(q)) return true;
  if (isSportsLiveMediaRequest(q)) return true;
  if (isTvMediaQuery(q) && resolveCategoryFromQuery(q)) return true;
  if (
    isTvMediaQuery(q) &&
    /(?:חפש|מצא|הראה|תן|show|search|find|רשימ|list|תציג|הצג)/i.test(q)
  ) {
    return true;
  }
  return false;
}

/** Normalized catalog search term (category id or station name). */
export function liveMediaCatalogSearchQuery(text: string): string {
  const fromCat = resolveCategoryFromQuery(text);
  if (fromCat) return fromCat;
  if (/סרט|movies?|cinema|film/i.test(text)) return "movies";
  if (/ילד|kids|children/i.test(text)) return "kids";
  if (/חדשות|news/i.test(text)) return "news";
  if (/מוזיק|music/i.test(text)) return "music";
  if (/קומד|comedy/i.test(text)) return "comedy";
  if (/ספורט|sport/i.test(text)) return "sports";
  if (/אנימ|anime/i.test(text)) return "anime";
  return text.trim();
}

export function isSportsLiveMediaRequest(text: string): boolean {
  const t = text.trim();
  return (
    /(?:חביל(?:ה|ת)|ערוץ|ערוצים|טל(?:ו)?יזיה|tv|live).*ספורט/i.test(t) ||
    /ספורט.*(?:ערוץ|שידור|live|טל(?:ו)?יזיה)/i.test(t) ||
    /supersport|super\s*sport|sport\s*(?:channel|tv|package|channels)/i.test(t)
  );
}

/** Browse regional radio lineup (no specific station name). */
export const isRadioBrowseQuery = (text: string): boolean => {
  const q = text.trim();
  if (!q) return false;
  if (isRadioFrequencyQuery(q)) return false;
  return (
    /(?:תחנ(?:ות|ת)\s+רדיו|רדיו\s+(?:אזורי|מקומי|local|regional)|radio\s+stations?|(?:חפש|מצא|הראה|תן).*(?:רדיו|radio)(?:\s|$|[?!.]))/i.test(
      q,
    ) ||
    /^(?:רדיו|radio)(?:\s+(?:אזורי|מקומי|local|stations?))?(?:[\s!?.]*)$/i.test(q)
  );
};

export type LiveMediaKind = "radio" | "livetv" | "both";

export function resolveLiveMediaKind(query: string, sportsPackage = false): LiveMediaKind {
  if (sportsPackage) return "livetv";
  const radio = isRadioMediaQuery(query);
  const tv = isTvMediaQuery(query);
  if (radio && !tv) return "radio";
  if (tv && !radio) return "livetv";
  if (radio && tv) {
    if (isRadioFrequencyQuery(query)) return "radio";
    return "both";
  }
  if (isRadioBrowseQuery(query)) return "radio";
  return "both";
}
