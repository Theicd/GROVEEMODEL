/**
 * Central open-web search planning — intent → short Hebrew engine queries + relevance filters.
 */
import { sanitizeSearchQuery } from "./queryExtract";
import {
  isEventsCalendarQuery,
  isFormulaOneQuery,
  isIsraelCinemaNowQuery,
  isOpenWebTopicQuery,
  isSportsChampionshipQuery,
  isSportsStandingsQuery,
  requestedBulletCount,
} from "./openWebTopicDetect";
import type { AnswerShape, WebSerpHit } from "./types";
import { isCinemaHomepageHit, parseCinemaMoviesFromText } from "./cinemaIlExtract";

export type OpenWebTopicKind =
  | "cinema_il"
  | "sports_championship"
  | "sports_standings"
  | "f1"
  | "events"
  | "generic";

export type WebTopicSearchPlan = {
  kind: OpenWebTopicKind;
  /** Short Hebrew phrases → Tavily / SearXNG / Scavio (2–3). */
  engineQueries: string[];
  userQuery: string;
  relevanceTerms: string[];
  blockPatterns: RegExp[];
  boostPatterns: RegExp[];
  answerShape: AnswerShape;
  bulletCount: number;
  blendNewsWithWeb: boolean;
  useWebFallback: boolean;
};

const TECH_NEWS_BLOCK =
  /(?:securityweek|fortibleed|malware|cybersecurity|crowdstrike|businessworld|workforce|cnet|tech\s+today|crypto|apple\s+patch|aws\s+continuum|in\s+other\s+news)/i;

const detectKind = (q: string): OpenWebTopicKind | null => {
  if (isIsraelCinemaNowQuery(q)) return "cinema_il";
  if (isSportsChampionshipQuery(q)) return "sports_championship";
  if (isSportsStandingsQuery(q)) return "sports_standings";
  if (isFormulaOneQuery(q)) return "f1";
  if (isEventsCalendarQuery(q)) return "events";
  if (isOpenWebTopicQuery(q)) return "generic";
  return null;
};

export const buildWebTopicSearchPlan = (rawQuery: string): WebTopicSearchPlan | null => {
  const userQuery = sanitizeSearchQuery(rawQuery);
  const kind = detectKind(userQuery);
  if (!kind) return null;

  const bullets = requestedBulletCount(userQuery);

  const base = {
    userQuery,
    answerShape: "bullet_list" as const,
    bulletCount: bullets,
    blendNewsWithWeb: false,
    useWebFallback: true,
  };

  switch (kind) {
    case "cinema_il":
      return {
        ...base,
        kind,
        engineQueries: [
          "סרטים בקולנוע ישראל עכשיו",
          "מה מציגים בקולנוע השבוע",
          "הסרטים המצליחים בקולנוע",
        ].slice(0, 3),
        relevanceTerms: ["סרט", "קולנוע", "מציג", "בקולנוע", "hotcinema", "seret", "offscreen", "הביפרים", "קופה"],
        blockPatterns: [TECH_NEWS_BLOCK, /(?:politics|netanyahu|בibi|בחירות)/i],
        boostPatterns: [
          /hotcinema|seret|offscreen|cinema-city|mako.*סרט|walla.*קולנוע|haaretz.*cinema|ynet.*סרט/i,
          /(?:ShowingNow|\/movies)/i,
          /(?:קופה ראשית|·\s*\d+\s*;)/i,
        ],
      };

    case "sports_championship": {
      const euro = /יורו|\beuro\b|uefa|אליפות\s+אירופ/i.test(userQuery);
      const wc = /מונדיאל|world\s+cup/i.test(userQuery);
      const player = /מצטיין|player\s+of|golden\s+ball|כדור\s+הזהב/i.test(userQuery);
      const engineQueries = euro
        ? player
          ? ["מנצחת יורו 2024", "שחקן מצטיין יורו", "גמר יורו אלופה"]
          : ["מנצחת יורו 2024", "גמר יורו תוצאה", "אלופת יורו"]
        : wc
          ? player
            ? ["מנצחת מונדיאל", "שחקן מצטיין מונדיאל", "גמר מונדיאל"]
            : ["מנצחת מונדיאל", "גמר מונדיאל", "אלופת העולם כדורגל"]
          : ["אלופת טורניר כדורגל", "מנצחת גמר", "שחקן מצטיין טורניר"];
      return {
        ...base,
        kind,
        engineQueries,
        relevanceTerms: ["יורו", "מונדיאל", "גמר", "מנצח", "אלופ", "שחקן", "מצטיין", "spain", "rodri"],
        blockPatterns: [TECH_NEWS_BLOCK],
        boostPatterns: [/uefa|fifa|euro|world\s+cup|יורו|מונדיאל|espn|bbc\s+sport/i],
      };
    }

    case "sports_standings":
      return {
        ...base,
        kind,
        engineQueries: /פרמייר|premier/i.test(userQuery)
          ? ["טבלת פרמייר ליג", "מובילים פרמייר ליג", "נקודות פרמייר ליג"]
          : ["טבלת ליגה כדורגל", "מובילים בליגה", "דירוג ליגה"],
        relevanceTerms: ["טבלה", "ליג", "נקודות", "מוביל", "standings", "premier"],
        blockPatterns: [TECH_NEWS_BLOCK],
        boostPatterns: [/premier|standings|טבל|ליג/i],
      };

    case "f1":
      return {
        ...base,
        kind,
        engineQueries: ["דירוג פורמולה 1", "מובילים פורמולה 1", "אליפות נהגים פורמולה"],
        relevanceTerms: ["פורמול", "f1", "נהגים", "דירוג"],
        blockPatterns: [TECH_NEWS_BLOCK],
        boostPatterns: [/formula|f1|פורמול/i],
      };

    case "events": {
      const city =
        userQuery.match(/(?:ב)([\u0590-\u05FF]{2,20})/)?.[1] ??
        userQuery.match(/\b(?:in|at)\s+([A-Za-z][A-Za-z\s-]{2,24})/i)?.[1]?.trim();
      const place = city ?? "לונדון";
      return {
        ...base,
        kind,
        engineQueries: [`אירועים ב${place}`, `פסטיבלים ב${place}`, `הופעות ב${place}`],
        relevanceTerms: ["אירוע", "פסטיבל", "הופע", place.replace(/\s/g, "")],
        blockPatterns: [TECH_NEWS_BLOCK],
        boostPatterns: [/event|festival|אירוע|פסטיבל/i],
      };
    }

    default:
      return {
        ...base,
        kind: "generic",
        engineQueries: [userQuery.slice(0, 60)],
        relevanceTerms: [],
        blockPatterns: [TECH_NEWS_BLOCK],
        boostPatterns: [],
      };
  }
};

export const buildFocusedWebSearchQuery = (query: string): string =>
  buildWebTopicSearchPlan(query)?.engineQueries[0] ?? sanitizeSearchQuery(query).slice(0, 60);

export const scoreWebHitForPlan = (hit: WebSerpHit, plan: WebTopicSearchPlan): number => {
  const blob = `${hit.title} ${hit.snippet} ${hit.url}`.toLowerCase();
  let score = 0;

  for (const re of plan.blockPatterns) {
    if (re.test(blob)) score -= 200;
  }
  for (const re of plan.boostPatterns) {
    if (re.test(blob)) score += 80;
  }
  for (const term of plan.relevanceTerms) {
    const t = term.toLowerCase();
    if (blob.includes(t)) score += t.length >= 5 ? 25 : 15;
  }
  if (plan.kind === "cinema_il" && TECH_NEWS_BLOCK.test(blob)) score -= 300;
  if (plan.kind === "cinema_il") {
    if (isCinemaHomepageHit(hit)) score -= 300;
    if (parseCinemaMoviesFromText(hit.snippet ?? "").length >= 2) score += 200;
    if (/cinema-city\.co\.il\/movies|hotcinema\.co\.il\/ShowingNow|seret\.co\.il\/movies/i.test(blob)) {
      score += 100;
    }
  }
  return score;
};

export const filterWebHitsForPlan = (
  hits: WebSerpHit[],
  plan: WebTopicSearchPlan,
  limit = 8,
): WebSerpHit[] => {
  const scored = hits
    .map((hit) => ({ hit, score: scoreWebHitForPlan(hit, plan) }))
    .filter(({ score }) => score > 0)
    .sort((a, b) => b.score - a.score);

  return scored.slice(0, limit).map(({ hit }) => hit);
};

export const planToWebSearchHint = (plan: WebTopicSearchPlan) => ({
  queries: plan.engineQueries,
  answerShape: plan.answerShape,
  useWebFallback: plan.useWebFallback,
  blendNewsWithWeb: plan.blendNewsWithWeb,
});

export const planToSearchPlan = (plan: WebTopicSearchPlan) => ({
  intents: [] as const,
  queries: plan.engineQueries,
  answerShape: plan.answerShape,
  useWebFallback: true,
  blendNewsWithWeb: false,
  reason: `open-web:${plan.kind}`,
});
