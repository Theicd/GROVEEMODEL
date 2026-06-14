import type { SearchIntent } from "./types";

/** Multi-source questions (הצלבה / טבעי) — need several live providers. */
export const isCrossSourceQuery = (text: string): boolean => {
  if (/^כמה\s+(?:מטוס|אונ(?:iyot|י)?|ספינ)/i.test(text)) return false;
  return (
  /^האם\s+/i.test(text) ||
  /קשר\s+בין/i.test(text) ||
  /(?:מטוס|אוני|רעיד|סופה|תחנת\s+חלל|starlink).*(?:וב(?:ו|ה)|באזור|מעל|ליד)/i.test(text) ||
  /(?:יש\s+.*(?:גם|וב(?:ו|ה))).*(?:מזג|תעופ|תנועה|אירוע)/i.test(text) ||
  /(?:הכי\s+עמוס|הכי\s+פעיל|כמה\s+אירועים\s+משמעותיים|משהו\s+חריג).*(?:עולם|ישראל|מעל)/i.test(
    text,
  ) ||
  /איזה\s+אזור\s+בעולם\s+נראה\s+הכי\s+פעיל/i.test(text)
  );
};

export const expandCrossSourceIntents = (
  query: string,
  base: SearchIntent[],
): SearchIntent[] => {
  const out = new Set<SearchIntent>(base);
  const q = query;

  if (/מטוס|תעופ|טיס|awacs/i.test(q)) out.add("aviation");
  if (/אוני|ספינ|נמל/i.test(q)) out.add("ships");
  if (/רעיד|earthquake/i.test(q)) out.add("earthquake");
  if (/סופה|hurricane|typhoon|טропי|צונאמ/i.test(q)) out.add("disaster");
  if (/סופה|hurricane|מזג|אזהר/i.test(q)) out.add("weather");
  if (/גלים|wave|גובה\s*גל|marine\s+weather|ocean\s+wave|ים\s+תיכון|בים\b/i.test(q)) out.add("marine");
  if (/שריפ|wildfire/i.test(q)) out.add("disaster");
  if (/צונאמ/i.test(q)) out.add("earthquake");
  if (/תחנת\s+חלל|\biss\b/i.test(q)) out.add("satellite");
  if (/starlink/i.test(q)) {
    out.add("satellite");
    out.add("news");
  }
  if (/חדשות|headline|מוזכר.*חדשות/i.test(q)) out.add("news");
  if (/מעל\s+ישראל|israel/i.test(q)) {
    out.add("alerts");
    out.add("aviation");
  }
  if (/קשר\s+בין\s+סופה.*תעבור|שיבושים\s+בתעבורה\s+אווירית/i.test(q)) {
    out.add("disaster");
    out.add("aviation");
    out.add("weather");
  }
  if (/נמל/i.test(q)) out.add("ships");
  if (/מזג.*תעופ|תעופ.*מזג/i.test(q)) {
    out.add("weather");
    out.add("aviation");
  }

  if (
    /(?:מה\s+קורה|אירועים\s+משמעותיים|משהו\s+חריג|הכי\s+פעיל)/i.test(q) &&
    out.size < 3
  ) {
    out.add("disaster");
    out.add("earthquake");
    out.add("aviation");
    out.add("news");
    out.add("alerts");
  }

  return [...out];
};
