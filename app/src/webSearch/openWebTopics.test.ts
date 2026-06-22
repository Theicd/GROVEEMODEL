import { describe, expect, it } from "vitest";
import { buildCapabilityLiveReply, buildOpenWebTopicReply, shouldDeliverStructuredLiveReply } from "./capabilityReplyMessages";
import { classifySearchIntents, needsWebSearch } from "./intents";
import {
  buildFocusedWebSearchQuery,
  inferAnswerShape,
  isCompositeFinanceQuery,
  isEventsCalendarQuery,
  isFormulaOneQuery,
  isMultiCountryGovernmentQuery,
  isSportsChampionshipQuery,
  isSportsStandingsQuery,
  needsOpenWebEnrichment,
  requestedBulletCount,
  wantsNewsHeadlineBulletsInChat,
} from "./openWebTopics";
import { isIsraelCinemaNowQuery } from "./openWebTopicDetect";
import type { SearchSourceResult } from "./types";

const EURO_Q =
  "מי הקבוצה שזכתה באליפות היורו האחרונה ומי נבחר לשחקן המצטיין של הטורניר?";

const CINEMA_Q =
  "חפש באינטרנט: מהם 3 הסרטים הכי מצליחים שמציגים עכשיו בבתי הקולנוע בישראל? תן תקציר שורה אחת";

describe("openWebTopics", () => {
  it("detects Premier League standings without explicit חפש", () => {
    const q = "מי מוביל בפרמייר ליג כרגע";
    expect(isSportsStandingsQuery(q)).toBe(true);
    expect(needsOpenWebEnrichment(q)).toBe(true);
    expect(needsWebSearch(q)).toBe(true);
  });

  it("detects Euro championship winner + player of tournament", () => {
    expect(isSportsChampionshipQuery(EURO_Q)).toBe(true);
    expect(needsOpenWebEnrichment(EURO_Q)).toBe(true);
    expect(needsWebSearch(EURO_Q)).toBe(true);
    expect(classifySearchIntents(EURO_Q)).not.toContain("currency");
    expect(classifySearchIntents(EURO_Q)).not.toContain("products");
  });

  it("builds focused web query for Euro (short Hebrew, not full sentence)", () => {
    const focused = buildFocusedWebSearchQuery(EURO_Q);
    expect(focused).toMatch(/[\u0590-\u05FF]/);
    expect(focused.length).toBeLessThan(40);
    expect(focused).not.toMatch(/מי הקבוצה/);
  });

  it("builds focused query for Israeli cinema (Hebrew only)", () => {
    expect(isIsraelCinemaNowQuery(CINEMA_Q)).toBe(true);
    const focused = buildFocusedWebSearchQuery(CINEMA_Q);
    expect(focused).toMatch(/קולנוע|סרט/);
    expect(focused).not.toMatch(/site:|box office|top 3|Israel cinema/i);
    expect(focused).not.toMatch(/תקציר/);
  });

  it("detects Formula 1", () => {
    const q = "חפש מי מוביל בפורמולה 1";
    expect(isFormulaOneQuery(q)).toBe(true);
    expect(needsOpenWebEnrichment(q)).toBe(true);
  });

  it("detects London events calendar", () => {
    const q = "חפש אירועים בלונדון בחודש הקרוב";
    expect(isEventsCalendarQuery(q)).toBe(true);
    expect(needsOpenWebEnrichment(q)).toBe(true);
  });

  it("detects composite finance (FX + weekly stock)", () => {
    const q = "מה שער הדולר והיורו מול השקל וכמה עלתה אפל השבוע";
    expect(isCompositeFinanceQuery(q)).toBe(true);
    expect(inferAnswerShape(q)).toBe("bullet_list");
  });

  it("detects multi-country government", () => {
    const q = "מי ראש הממשלה בבריטניה ובצרפת";
    expect(isMultiCountryGovernmentQuery(q)).toBe(true);
    expect(inferAnswerShape(q)).toBe("bullet_list");
  });

  it("detects news headline bullets in chat", () => {
    const q = "חפש שלוש כותרות חדשות מעולם ב-24 שעות האחרונות";
    expect(wantsNewsHeadlineBulletsInChat(q)).toBe(true);
    expect(requestedBulletCount(q)).toBe(3);
    expect(inferAnswerShape(q)).toBe("bullet_list");
  });

  it("buildOpenWebTopicReply uses filtered Tavily web hits", () => {
    const src: SearchSourceResult = {
      provider: "tavily",
      label: "Tavily (web)",
      ok: true,
      text: "תוצאות חיפוש כללי (Tavily):",
      webHits: [
        {
          title: "Apple Patches Beats - SecurityWeek",
          url: "https://securityweek.com/x",
          snippet: "CryptoBandits Malware",
        },
        {
          title: "Spain win Euro 2024 — Rodri player of the tournament",
          url: "https://example.com/euro",
          snippet: "Spain beat England 2-1 in Berlin.",
        },
      ],
      latencyMs: 100,
    };
    const reply = buildOpenWebTopicReply(EURO_Q, [src]);
    expect(reply).toMatch(/Spain|Euro 2024|Rodri/i);
    expect(reply).not.toMatch(/SecurityWeek|CryptoBandits/i);
    expect(shouldDeliverStructuredLiveReply(EURO_Q, [], [src], reply)).toBe(true);
  });
});
