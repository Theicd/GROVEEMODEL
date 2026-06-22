import {
  isBareWorldNewsQuery,
  isCurrencyQuery,
  isMarketPriceQuery,
  isNewsQuery,
  userRequestsSearch,
} from "./intents";
import { extractCountryPhrases, isWorldHeadlineQuery } from "./queryExtract";
import type { AnswerShape } from "./types";
import {
  buildFocusedWebSearchQuery,
  isEventsCalendarQuery,
  isFormulaOneQuery,
  isIsraelCinemaNowQuery,
  isOpenWebTopicQuery,
  isSportsChampionshipQuery,
  isSportsStandingsQuery,
  requestedBulletCount,
} from "./openWebTopicDetect";

export {
  buildFocusedWebSearchQuery,
  isEventsCalendarQuery,
  isFormulaOneQuery,
  isIsraelCinemaNowQuery,
  isOpenWebTopicQuery,
  isSportsChampionshipQuery,
  isSportsStandingsQuery,
  isEuroFootballNotCurrency,
  requestedBulletCount,
} from "./openWebTopicDetect";

/** Topics that need Tavily/SearXNG — no explicit «חפש» required. */
export const needsOpenWebEnrichment = (text: string): boolean => isOpenWebTopicQuery(text);

/** FX + stock in one question (e.g. USD/EUR rates + AAPL weekly). */
export const isCompositeFinanceQuery = (text: string): boolean => {
  const fx =
    isCurrencyQuery(text) ||
    (/שער/i.test(text) && /(?:דולר|usd|יורו|eur)/i.test(text) && /(?:שקל|ils)/i.test(text));
  const market =
    isMarketPriceQuery(text) ||
    (/(?:מנ(?:י(?:ה|ית|ות)?|stock)|AAPL|אפל|NVDA|TSLA)/i.test(text) &&
      /(?:אחוז|percent|%|שבוע|week|על(?:ת(?:ה)?|ה)|ירד(?:ה)?)/i.test(text));
  return fx && market;
};

/** Multi-country leadership (UK + France). */
export const isMultiCountryGovernmentQuery = (text: string): boolean =>
  /(?:ראש\s+(?:ה)?ממשל(?:ה|ת)?|נשיא|prime\s+minister|president)/i.test(text) &&
  extractCountryPhrases(text).length >= 2;

/** User wants headline bullets in chat, not only the news panel. */
export const wantsNewsHeadlineBulletsInChat = (text: string): boolean => {
  if (!isNewsQuery(text)) return false;
  if (isWorldHeadlineQuery(text) || isBareWorldNewsQuery(text)) return true;
  return (
    /(?:סיכום\s+קצר|שורה\s+אחת|one\s+line|bullet|נקוד(?:ה|ות))/i.test(text) ||
    /(?:מהן|מהם|what\s+are)\s+(?:\d+|שלוש|three|3)\s+(?:ה)?(?:כותרות|headlines|אירועים|events)/i.test(
      text,
    ) ||
    /(?:\d+|שלוש|three|3)\s+(?:כותרות|headlines|אירועים|events)/i.test(text)
  );
};

export const inferAnswerShape = (query: string): AnswerShape | null => {
  if (isCompositeFinanceQuery(query)) return "bullet_list";
  if (wantsNewsHeadlineBulletsInChat(query)) return "bullet_list";
  if (
    /(?:\d+|שלוש|three|3)\s+(?:סרטים|אירועים|כותרות|headlines|events|movies|נקודות)/i.test(
      query,
    )
  ) {
    return "bullet_list";
  }
  if (isMultiCountryGovernmentQuery(query)) return "bullet_list";
  if (needsOpenWebEnrichment(query)) return "bullet_list";
  return null;
};

/** True when user explicitly asked to search OR topic needs live web either way. */
export const shouldRunWebSearchForQuery = (text: string): boolean =>
  userRequestsSearch(text) || isOpenWebTopicQuery(text);
