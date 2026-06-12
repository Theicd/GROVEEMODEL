import type { GameCategoryId } from "./types";

/** Internet Archive advancedsearch clauses (online = has emulator). */
export const ARCHIVE_CATEGORY_QUERIES: Record<GameCategoryId, string> = {
  featured:
    "(collection:internetarcade OR collection:classicpcgames OR collection:consolelivingroom) AND mediatype:software",
  arcade: "collection:internetarcade AND mediatype:software",
  shooter:
    "(collection:internetarcade AND mediatype:software) AND (subject:shooter OR title:shooter OR title:shooting)",
  action:
    "(collection:internetarcade AND mediatype:software) AND (subject:action OR title:action)",
  rpg:
    '(mediatype:software) AND (emulator:*) AND (subject:"role playing" OR subject:rpg OR subject:adventure OR subject:quest OR title:quest)',
  strategy: '(mediatype:software) AND (subject:"strategy game" OR subject:strategy) AND (emulator:*)',
  racing:
    "(collection:internetarcade OR mediatype:software) AND (subject:racing OR title:racing) AND (emulator:*)",
  fighting:
    "(mediatype:software) AND (emulator:*) AND (subject:fighting OR title:fighting OR title:mortal OR title:tekken OR title:street fighter)",
  puzzle: '(mediatype:software) AND (subject:"puzzle game" OR subject:puzzle) AND (emulator:*)',
  sports:
    "(collection:internetarcade OR mediatype:software) AND subject:sports AND (emulator:*)",
  retro:
    '(mediatype:software) AND (emulator:*) AND (subject:"DOS games" OR subject:"MS-DOS" OR subject:"classic games" OR collection:classicpcgames)',
  dos: "(mediatype:software) AND (emulator:dosbox)",
  console: "collection:consolelivingroom AND mediatype:software",
  ps1: "(mediatype:software) AND (emulator:psx)",
  ps2: "(mediatype:software) AND (emulator:pcsx2)",
  sony: "(mediatype:software) AND (emulator:psx OR emulator:pcsx2)",
};

export const FEATURED_ROTATION_POOL: GameCategoryId[] = [
  "arcade",
  "dos",
  "retro",
  "fighting",
  "ps1",
  "console",
  "shooter",
  "racing",
  "puzzle",
  "sports",
];

export const GAME_CATEGORIES: Array<{ id: GameCategoryId; labelHe: string; icon: string }> = [
  { id: "featured", labelHe: "מומלצים", icon: "🌟" },
  { id: "arcade", labelHe: "ארקייד", icon: "🕹️" },
  { id: "ps1", labelHe: "PS1 / Sony", icon: "🎮" },
  { id: "ps2", labelHe: "PS2", icon: "💿" },
  { id: "fighting", labelHe: "קרב", icon: "🥊" },
  { id: "dos", labelHe: "PC / DOS", icon: "💻" },
  { id: "console", labelHe: "קונסולות", icon: "📺" },
  { id: "shooter", labelHe: "ירי", icon: "🔫" },
  { id: "action", labelHe: "אקשן", icon: "⚔️" },
  { id: "rpg", labelHe: "RPG / קווסט", icon: "🗡️" },
  { id: "strategy", labelHe: "אסטרטגיה", icon: "🗼" },
  { id: "racing", labelHe: "מירוצים", icon: "🏎️" },
  { id: "puzzle", labelHe: "חידות", icon: "🧩" },
  { id: "sports", labelHe: "ספורט", icon: "⚽" },
  { id: "retro", labelHe: "רטרו", icon: "👾" },
];

const appendYearClauses = (
  clauses: string[],
  params: { year?: number | null; yearFrom?: number | null; yearTo?: number | null },
): void => {
  const { year, yearFrom, yearTo } = params;
  if (yearFrom != null && yearTo != null) {
    clauses.push(
      `(year:[${yearFrom} TO ${yearTo}] OR date:[${yearFrom}-01-01 TO ${yearTo}-12-31])`,
    );
  } else if (year != null) {
    clauses.push(`(year:${year} OR date:[${year}-01-01 TO ${year}-12-31])`);
  }
};

export const buildOnlineArchiveQuery = (params: {
  query?: string;
  category?: GameCategoryId | null;
  year?: number | null;
  yearFrom?: number | null;
  yearTo?: number | null;
}): string => {
  const clauses: string[] = [];
  const cat =
    params.category && ARCHIVE_CATEGORY_QUERIES[params.category]
      ? `(${ARCHIVE_CATEGORY_QUERIES[params.category]})`
      : "(collection:internetarcade AND mediatype:software)";

  clauses.push(cat);

  if (!params.category || !["ps1", "ps2", "sony", "dos"].includes(params.category)) {
    clauses.push("(emulator:*)");
  }

  appendYearClauses(clauses, params);

  const q = String(params.query ?? "").trim();
  if (q) {
    const escaped = q.replace(/"/g, "");
    clauses.push(`(title:"${escaped}" OR subject:"${escaped}" OR description:"${escaped}" OR ${escaped})`);
  }

  return clauses.join(" AND ");
};

export const buildTitleSearchQuery = (
  query: string,
  year: number | null,
  yearFrom: number | null = null,
  yearTo: number | null = null,
): string => {
  const clauses = ["mediatype:software", "(emulator:*)"];
  const escaped = query.replace(/"/g, "").trim();
  if (escaped) {
    clauses.push(
      `(title:"${escaped}" OR subject:"${escaped}" OR description:"${escaped}" OR ${escaped})`,
    );
  }
  appendYearClauses(clauses, { year, yearFrom, yearTo });
  return clauses.join(" AND ");
};
