/** Pure topic detection — no imports from intents (avoids circular deps). */

export const isSportsStandingsQuery = (text: string): boolean =>
  /(?:פרמייר\s*ליג|premier\s*league|la\s*liga|serie\s*a|bundesliga|champions\s*league|ליג(?:ה|ת)\s*(?:ה)?(?:אנגל|ספרד|כדורגל)|טבל(?:ה|ת)|standings|league\s+table|משחק\s+ה(?:בא|גדול)|next\s+match)/i.test(
    text,
  ) &&
  /(?:מוביל|leader|first|נקודות|points|טבל|standings|משחק|match|קבוצה|team)/i.test(text);

export const isFormulaOneQuery = (text: string): boolean =>
  /(?:formula\s*1|formula\s*one|פורמול(?:ה|א)\s*1|\bf1\b|אליפות\s+(?:ה)?נהגים|drivers?\s+championship|grand\s+prix)/i.test(
    text,
  );

export const isEventsCalendarQuery = (text: string): boolean =>
  /(?:פסטיבל|festival|תערוכ(?:ה|ות)|exhibition|אירוע(?:ים)?|events?|concerts?|הופע(?:ה|ות)|show\s+calendar)/i.test(
    text,
  ) &&
  /(?:ב[\u0590-\u05FF]{2,}|(?:in|at)\s+[A-Za-z])/i.test(text) &&
  /(?:חודש|month|הקרוב|upcoming|next\s+month|במהלך)/i.test(text);

export const isIsraelCinemaNowQuery = (text: string): boolean =>
  /(?:סרט|cinema|קולנוע|movies?|films?)/i.test(text) &&
  /(?:בתי\s+קולנוע|בקולנוע|קולנוע|cinema|box\s*office|now\s+playing|מציג(?:ים)?|רץ\s+עכשיו)/i.test(
    text,
  ) &&
  /(?:ישראל|israel|עכשיו|currently|now|הכי\s+(?:מצליח|פופולרי|נצפ(?:ים|ה)?))/i.test(text);

export const isSportsChampionshipQuery = (text: string): boolean => {
  const tournament =
    /(?:יורו|\beuro\b|euro\s*20|מונדיאל|world\s+cup|אליפות\s+(?:אירופ|העולם|היורו)|european\s+championship|uefa)/i.test(
      text,
    );
  const ask =
    /(?:זכ(?:ת(?:ה)?|ה)|ניצח(?:ון|ה)|מנצח(?:ת|ים)?|אלופ(?:ה|ים)?|winner|won|champion|מצטיין|player\s+of|golden\s+ball|best\s+player|כדור\s+הזהב|שחקן\s+(?:ה)?(?:משחק|מצטיין|הטורניר))/i.test(
      text,
    );
  return tournament && ask;
};

/** Euro/World Cup context — not FX (יורו as tournament, not currency). */
export const isEuroFootballNotCurrency = (text: string): boolean =>
  /(?:יורו|\beuro\b|euro\s*20|uefa|מונדיאל|world\s+cup|אליפות\s+(?:אירופ|העולם|היורו))/i.test(text) &&
  /(?:זכ|winner|אליפ|מונדיאל|יורו|cup|champion|מצטיין|player|כדורגל|football|soccer|קבוצה|team|טורניר|tournament)/i.test(
    text,
  ) &&
  !/(?:שער|exchange\s+rate|currency\s+rate|convert|המר|יחס|קונה|buy|מול\s+(?:ה)?שקל|שקל|שווים|worth|\bils\b)/i.test(text);

export const isOpenWebTopicQuery = (text: string): boolean =>
  isSportsStandingsQuery(text) ||
  isFormulaOneQuery(text) ||
  isEventsCalendarQuery(text) ||
  isIsraelCinemaNowQuery(text) ||
  isSportsChampionshipQuery(text);

export const requestedBulletCount = (text: string): number => {
  const wordMap: Record<string, number> = { שלוש: 3, שלושה: 3, three: 3 };
  const m = text.match(/(?:^|\s)(3|three|שלוש|שלושה|\d{1,2})(?:\s|$|[?!.,])/i);
  if (!m?.[1]) return 3;
  const raw = m[1].toLowerCase();
  if (wordMap[raw]) return wordMap[raw];
  const n = Number(raw);
  return Number.isFinite(n) && n >= 1 && n <= 10 ? n : 3;
};

export { buildFocusedWebSearchQuery } from "./webTopicQueryPlan";
