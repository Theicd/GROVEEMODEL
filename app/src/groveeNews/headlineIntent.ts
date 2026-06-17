import { isGeneralNewsDigestQuery, isWorldHeadlineQuery } from "../webSearch/queryExtract";
import { isBareWorldNewsQuery } from "../webSearch/intents";
import { isExplicitNewsTopicSearch } from "./newsQueryNormalize";

/** Broad world-headlines / digest queries → Topics digest (not live RSS). */
export function isTopicsOverviewQuery(query: string): boolean {
  const q = query.trim();
  if (!q) return false;
  if (isExplicitNewsTopicSearch(q)) return false;
  if (isBareWorldNewsQuery(q)) return true;
  if (isGeneralNewsDigestQuery(q) || isWorldHeadlineQuery(q)) return true;
  return (
    /(?:מה\s+חדש\s+בעולם|מה\s+קורה\s+בעולם|מה\s+הולך\s+בעולם)/i.test(q) ||
    /(?:what'?s\s+(?:new|happening)\s+(?:in\s+)?(?:the\s+)?world)/i.test(q) ||
    (/(?:כותרות|headlines?)\s+(?:מובילות|עיקריות|ראשיות)/i.test(q) &&
      /(?:עולם|world|global|בינלאומ)/i.test(q))
  );
}
