import { GAME_CATEGORIES } from "./archiveQueries";
import type { GameCategoryId } from "./types";

/** Hebrew / English keywords per sidebar category (source of truth: GAME_CATEGORIES labels). */
export const CATEGORY_KEYWORD_RES: Record<GameCategoryId, RegExp> = {
  featured: /מ+?ו?לצ|featured|recommended|top\s*games?|המלצ(?:ה|ות).*משחק|אילו\s+משחקים/i,
  arcade: /ארקייד|arcade|ארקד|internet\s*arcade/i,
  ps1: /(?:ps\s*1|\bps1\b|\bpsx\b|פלייסטיישן(?!\s*2)|playstation(?!\s*2)|פס\s*1|סוני(?!\s*2))/i,
  ps2: /(?:ps\s*2|\bps2\b|פלייסטיישן\s*2|playstation\s*2|פס\s*2)/i,
  fighting: /קרב|fighting|fighter|מורטל|mortal\s*kombat|tekken|טקן/i,
  dos: /\bdos\b|\bpc\b|pc\s*\/\s*dos|מחשב|classic\s*pc|ms-?dos/i,
  console: /קונסול|console|nes|snes|genesis|megadrive|nintendo/i,
  shooter: /ירי|shooter|shoot(?:er|ing)?|\bfps\b/i,
  action: /אקשן|\baction\b/i,
  rpg: /rpg|קווסט|quest|הרפתק|תפקידים|role\s*playing/i,
  strategy: /אסטרטג|strategy|\brts\b/i,
  racing: /מירוצ|racing|\brace(?:s|ing)?\b|נהיג/i,
  puzzle: /חיד(?:ה|ות)|puzzle/i,
  sports: /ספורט|sports|football|soccer|כדור/i,
  retro: /רטרו|retro|ישנ(?:ים|ים)?|old\s*games?|classic\s*games?|vintage/i,
  sony: /סוני|sony/i,
};

const CATEGORY_LABEL_RES: Array<{ id: GameCategoryId; re: RegExp }> = GAME_CATEGORIES.map(
  (cat) => {
    const parts = cat.labelHe
      .split("/")
      .map((p) => p.trim())
      .filter(Boolean);
    const escaped = parts.map((p) => p.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    return {
      id: cat.id,
      re: new RegExp(escaped.join("|"), "i"),
    };
  },
);

export const detectCategoryFromText = (text: string): GameCategoryId | null => {
  const t = text.trim();
  if (!t) return null;
  for (const { id, re } of CATEGORY_LABEL_RES) {
    if (re.test(t)) return id;
  }
  for (const [id, re] of Object.entries(CATEGORY_KEYWORD_RES) as Array<
    [GameCategoryId, RegExp]
  >) {
    if (re.test(t)) return id;
  }
  return null;
};

export const isCategoryOnlyText = (text: string, category: GameCategoryId | null): boolean => {
  const q = text.trim();
  if (!q || !category) return false;
  const cat = GAME_CATEGORIES.find((c) => c.id === category);
  if (!cat) return false;
  const stripped = q
    .replace(/^(?:משחק(?:ים)?|games?)\s*(?:של|ב)?\s*/i, "")
    .replace(/[?!.]+$/g, "")
    .trim();
  if (!stripped) return true;
  return CATEGORY_KEYWORD_RES[category].test(stripped) || CATEGORY_LABEL_RES.some(
    (c) => c.id === category && c.re.test(stripped),
  );
};

export const categoryLabelHe = (id: GameCategoryId): string =>
  GAME_CATEGORIES.find((c) => c.id === id)?.labelHe ?? id;

export const formatCategoryListForPrompt = (): string =>
  GAME_CATEGORIES.map((c) => `${c.icon} ${c.labelHe}`).join(" · ");
