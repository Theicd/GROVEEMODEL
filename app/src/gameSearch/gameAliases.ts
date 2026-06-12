import type { GameCategoryId, ResolvedGameSearch } from "./types";

type AliasRule = {
  patterns: RegExp[];
  search: string;
  category?: GameCategoryId;
  year?: number;
};

type DecadeRule = { re: RegExp; from: number; to: number; label: string };

const DECADE_RULES: DecadeRule[] = [
  { re: /(?:שנות|מ(?:שנות|ה)?)\s*(?:ה)?שמונ|(?:מ)?שנות\s*ה?80|(?:ה)?80s|1980s|\b80\b(?=\s*(?:s|'|ים|משנות))/i, from: 1980, to: 1989, label: "שנות ה-80" },
  { re: /(?:שנות|מ(?:שנות|ה)?)\s*(?:ה)?תשע|(?:מ)?שנות\s*ה?90|(?:ה)?90s|1990s|\b90\b(?=\s*(?:s|'|ים|משנות))/i, from: 1990, to: 1999, label: "שנות ה-90" },
  { re: /(?:שנות|מ(?:שנות|ה)?)\s*(?:ה)?שבע|(?:מ)?שנות\s*ה?70|(?:ה)?70s|1970s/i, from: 1970, to: 1979, label: "שנות ה-70" },
  { re: /(?:שנות|מ(?:שנות|ה)?)\s*(?:ה)?שיש|(?:מ)?שנות\s*ה?60|(?:ה)?60s|1960s/i, from: 1960, to: 1969, label: "שנות ה-60" },
];

const ALIAS_RULES: AliasRule[] = [
  { patterns: [/הטירה\s*הנאצ/i, /הטירה\s*הנא/i, /וולפנשט/i, /wolfenstein/i], search: "wolfenstein", category: "dos" },
  { patterns: [/אחוזת\s*המטורפ/i, /maniac\s*mansion/i], search: "maniac mansion", year: 1987, category: "dos" },
  { patterns: [/מורטל|mortal\s*kombat|mk\s*\d/i], search: "mortal kombat", category: "fighting" },
  { patterns: [/ה?נסיך\s*(?:ה)?פרס|prince\s*of\s*persia/i], search: "prince of persia", category: "dos" },
  { patterns: [/עולם\s*אחר|another\s*world|out\s*of\s*this\s*world/i], search: "another world", category: "dos", year: 1991 },
  { patterns: [/חולית|\bdune\b/i], search: "dune", category: "dos" },
  { patterns: [/מחלק\s*(?:ה)?בירות|police\s*quest|משטרה/i], search: "police quest", category: "dos" },
  { patterns: [/swat\b|ס.w.a.t/i], search: "swat", category: "dos" },
  { patterns: [/ק(?:ינג|ינג'?s)\s*quest|kings?\s*quest/i], search: "kings quest", category: "dos" },
  { patterns: [/space\s*quest|ספייס\s*קווסט/i], search: "space quest", category: "dos" },
  { patterns: [/monkey\s*island|קופ\s*קאק/i], search: "monkey island", category: "dos" },
  { patterns: [/zork\b|זורק/i], search: "zork", category: "dos" },
  { patterns: [/quest\s*for\s*glory|קווסט\s*לכ\b/i], search: "quest for glory", category: "rpg" },
  { patterns: [/lemmings|למינג/i], search: "lemmings", category: "dos" },
  { patterns: [/commandos|קומando/i], search: "commandos", category: "dos" },
  { patterns: [/pac-?man|פac-?man|פק-?מן/i], search: "pac-man", category: "arcade" },
  { patterns: [/גלaga|galaga/i], search: "galaga", category: "arcade" },
  { patterns: [/דון\s*קיי|donkey\s*kong/i], search: "donkey kong", category: "arcade" },
  { patterns: [/טקן|tekken/i], search: "tekken", category: "ps1" },
  { patterns: [/פי\s*nal\s*fantasy|final\s*fantasy/i], search: "final fantasy", category: "ps1" },
  { patterns: [/ריזident|resident\s*evil/i], search: "resident evil", category: "ps1" },
];

const PLATFORM_HINTS: Array<{ re: RegExp; category: GameCategoryId }> = [
  { re: /פלייסטיישן\s*2|playstation\s*2|\bps2\b|פס\s*2|סוני\s*2/i, category: "ps2" },
  { re: /פלייסטיישן|\bps1\b|\bpsx\b|פס\s*1|סוני|sony|playstation/i, category: "ps1" },
];

const YEAR_RE = /\b(19[6-9]\d|20[0-2]\d)\b/;

const META_PHRASE_RES = [
  /(?:ו)?תציג(?:\s+(?:אות(?:ם|ן|ו)|לי))?/gi,
  /(?:ו)?תראה(?:\s+לי)?/gi,
  /(?:ו)?הראה(?:\s+לי)?/gi,
  /(?:ו)?הצג(?:\s+(?:אות(?:ם|ן|ו)|לי))?/gi,
  /בקטגור(?:יה|יית)\s*(?:של\s*)?/gi,
  /(?:שנות|מ(?:שנות|ה)?)\s*(?:ה)?(?:שמונ|תשע|שבע|שיש)/gi,
  /(?:מ)?שנות\s*ה?\s*(?:80|90|70|60|'80|'90)/gi,
  /\b(?:80|90|70)s\b/gi,
  /(?:אני\s+)?רוצה\s+לשחק/gi,
  /(?:און\s*ליין|online|בדפדפן)/gi,
];

const CATEGORY_LABELS: Record<GameCategoryId, string> = {
  featured: "מומלצים",
  arcade: "ארקייד",
  shooter: "ירי",
  action: "אקשן",
  rpg: "RPG / קווסט",
  strategy: "אסטרטגיה",
  racing: "מירוצים",
  fighting: "קרב",
  puzzle: "חידות",
  sports: "ספורט",
  retro: "רטרו",
  dos: "PC / DOS",
  console: "קונסולות",
  ps1: "PS1",
  ps2: "PS2",
  sony: "Sony",
};

export const extractYearFromText = (text: string): number | null => {
  const m = text.match(YEAR_RE);
  if (!m) return null;
  const y = parseInt(m[1], 10);
  return y >= 1960 && y <= 2030 ? y : null;
};

export const extractDecadeRange = (
  text: string,
): { yearFrom: number; yearTo: number; label: string } | null => {
  for (const rule of DECADE_RULES) {
    if (rule.re.test(text)) {
      return { yearFrom: rule.from, yearTo: rule.to, label: rule.label };
    }
  }
  return null;
};

export const stripYearFromQuery = (text: string): string =>
  text.replace(YEAR_RE, "").replace(/\s{2,}/g, " ").trim();

export const stripMetaPhrases = (text: string): string => {
  let t = text;
  for (const re of META_PHRASE_RES) {
    t = t.replace(re, " ");
  }
  return t.replace(/\s{2,}/g, " ").trim();
};

const applyAlias = (
  text: string,
): { query: string; category: GameCategoryId | null; year: number | null } => {
  const lower = text.toLowerCase();
  for (const rule of ALIAS_RULES) {
    if (rule.patterns.some((p) => p.test(text) || p.test(lower))) {
      return {
        query: rule.search,
        category: rule.category ?? null,
        year: rule.year ?? null,
      };
    }
  }
  return { query: text, category: null, year: null };
};

const isNoiseQuery = (q: string): boolean =>
  !q ||
  /^(?:משחק(?:ים)?|games?|של|את|ה)$/i.test(q) ||
  q.length < 2;

export const buildGamePanelTitle = (resolved: ResolvedGameSearch): string => {
  if (resolved.browseMode && resolved.category === "featured" && !resolved.query) {
    return "משחקים מומלצים";
  }
  const parts: string[] = [];
  if (resolved.query) parts.push(resolved.query);
  if (resolved.category && resolved.category !== "featured") {
    parts.push(CATEGORY_LABELS[resolved.category] ?? resolved.category);
  }
  if (resolved.yearFrom !== null && resolved.yearTo !== null) {
    parts.push(`${resolved.yearFrom}–${resolved.yearTo}`);
  } else if (resolved.year) {
    parts.push(String(resolved.year));
  }
  if (!parts.length) return "משחקים און־ליין";
  return `משחקים: ${parts.join(" · ")}`;
};

export const resolveGameSearch = (
  rawText: string,
  hintCategory: GameCategoryId | null,
): ResolvedGameSearch => {
  const decade = extractDecadeRange(rawText);
  let text = stripMetaPhrases(rawText.trim());
  const yearFromText = extractYearFromText(text);
  text = stripYearFromQuery(text);

  const alias = applyAlias(text);
  let query = alias.query.trim();
  let category = alias.category ?? hintCategory;
  let year = alias.year ?? yearFromText;
  let yearFrom = decade?.yearFrom ?? null;
  let yearTo = decade?.yearTo ?? null;

  for (const { re, category: cat } of PLATFORM_HINTS) {
    if (re.test(rawText)) {
      category = cat;
      query = query
        .replace(/פלייסטיישן\s*2?|playstation\s*2?|\bps[12x]\b|פס\s*[12]|סוני\s*2?|sony/gi, "")
        .trim();
      break;
    }
  }

  if (/משחקים?\s*מ+?ו?לצ|מ+?ו?לצים|המלצ(?:ה|ות).*משחק|recommended\s*games?|top\s*games?/i.test(rawText)) {
    category = category ?? "featured";
  }

  query = stripMetaPhrases(query);
  const browseMode = isNoiseQuery(query) || (!query && (yearFrom !== null || category !== null));

  const panelTitle = buildGamePanelTitle({
    query,
    year,
    yearFrom,
    yearTo,
    category,
    browseMode,
    panelTitle: "",
  });

  return { query: isNoiseQuery(query) ? "" : query, year, yearFrom, yearTo, category, browseMode, panelTitle };
};

export const formatPopularityLabel = (downloads?: number): string | null => {
  if (!downloads || downloads < 100) return null;
  if (downloads >= 1_000_000) return `${(downloads / 1_000_000).toFixed(1)}M הורדות`;
  if (downloads >= 1_000) return `${Math.round(downloads / 1_000)}K הורדות`;
  return `${downloads} הורדות`;
};
