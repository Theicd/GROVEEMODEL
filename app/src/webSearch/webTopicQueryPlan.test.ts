import { describe, expect, it } from "vitest";

import {

  buildWebTopicSearchPlan,

  filterWebHitsForPlan,

  scoreWebHitForPlan,

} from "./webTopicQueryPlan";



const CINEMA_Q =

  "חפש באינטרנט: מהם 3 הסרטים הכי מצליחים שמציגים עכשיו בבתי הקולנוע בישראל? תן תקציר";



describe("webTopicQueryPlan", () => {

  it("builds cinema plan with short Hebrew engine queries (not full chat)", () => {

    const plan = buildWebTopicSearchPlan(CINEMA_Q);

    expect(plan).not.toBeNull();

    expect(plan!.kind).toBe("cinema_il");

    expect(plan!.engineQueries.length).toBeGreaterThanOrEqual(2);

    for (const q of plan!.engineQueries) {

      expect(q).toMatch(/[\u0590-\u05FF]/);

      expect(q).not.toMatch(/site:|https?:|\.com\b|\btop\b|\bmovies\b|\bbox office\b/i);

      expect(q).not.toMatch(/תקציר|מהם 3/);

    }

    expect(plan!.blendNewsWithWeb).toBe(false);

  });



  it("filters out tech/security noise from cinema hits", () => {

    const plan = buildWebTopicSearchPlan(CINEMA_Q)!;

    const bad = {

      title: "Apple Patches Beats Eavesdropping Flaw - SecurityWeek",

      url: "https://securityweek.com/apple",

      snippet: "CryptoBandits Malware Doubles as a Backdoor",

    };

    const good = {

      title: "עכשיו בקולנוע - Hot Cinema",

      url: "https://hotcinema.co.il/ShowingNow",

      snippet: "השטן לובשת פראדה 2 · ג'ורג' · קופה ראשית",

    };

    expect(scoreWebHitForPlan(bad, plan)).toBeLessThan(0);

    expect(scoreWebHitForPlan(good, plan)).toBeGreaterThan(0);

    const filtered = filterWebHitsForPlan([bad, good], plan, 3);

    expect(filtered).toHaveLength(1);

    expect(filtered[0]!.url).toContain("hotcinema");

  });

});

