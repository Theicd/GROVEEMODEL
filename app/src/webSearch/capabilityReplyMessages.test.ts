import { describe, expect, it } from "vitest";
import { clearLiveWorldSnapshotCache, setLiveWorldSnapshot } from "../liveWorld/snapshotStore";
import { buildCapabilityLiveReply, buildWebFallbackNoDataReply, shouldDeliverStructuredLiveReply } from "./capabilityReplyMessages";
import type { SearchSourceResult } from "./types";

const src = (over: Partial<SearchSourceResult>): SearchSourceResult => ({
  provider: "adsb-aviation",
  label: "תעופה (ADS-B)",
  ok: true,
  text: "",
  latencyMs: 1,
  ...over,
});

describe("buildCapabilityLiveReply", () => {
  it("returns aviation count for B09-style query", () => {
    const reply = buildCapabilityLiveReply(
      "כמה מטוסים נמצאים כרגע מעל ישראל?",
      ["aviation"],
      [
        src({
          text: "אזור: ישראל\nמטוסים בטווח: 61\n1. BAW60",
        }),
      ],
    );
    expect(reply).toContain("61");
    expect(reply).toContain("Sources:");
  });

  it("returns Starlink count from live catalog source", () => {
    const reply = buildCapabilityLiveReply(
      "כמה לווייני Starlink פעילים כרגע?",
      ["satellite"],
      [
        src({
          provider: "starlink-catalog",
          label: "Starlink (CelesTrak)",
          text: "ANSWER (Starlink active): 6421\nלווייני Starlink בקטalog CelesTrak (GROUP=starlink): 6421",
        }),
      ],
    );
    expect(reply).toMatch(/6421/);
    expect(reply).toMatch(/Starlink|CelesTrak/i);
    expect(reply).not.toMatch(/לא נתמך/);
  });

  it("returns regional Starlink unsupported without catalog", () => {
    const reply = buildCapabilityLiveReply(
      "אילו לווייני Starlink נמצאים מעל אירופה?",
      ["satellite"],
      [],
    );
    expect(reply).toMatch(/לא נתמך|אזור|CelesTrak/i);
  });

  it("returns concise GitHub top repo for B16", () => {
    const reply = buildCapabilityLiveReply(
      "מהו הפרויקט הפופולרי ביותר היום ב-GitHub?",
      ["github"],
      [
        src({
          provider: "github",
          label: "GitHub Repositories",
          text: [
            "ANSWER (GitHub top): freeCodeCamp/freeCodeCamp ★430000",
            "סינון: stars:>500 pushed:>2026-06-07 archived:false fork:false",
            "הפרויקט הפופולרי ביותר בין מאגרים עם push אחרון לאחרונה: freeCodeCamp/freeCodeCamp [TypeScript]: Learn to code (https://github.com/freeCodeCamp/freeCodeCamp) ★430,000",
          ].join("\n"),
        }),
      ],
    );
    expect(reply).toMatch(/freeCodeCamp/);
    expect(reply).toMatch(/430,?000|430000/);
    expect(reply).not.toMatch(/funNLP|vuejs\/vue/i);
    expect(reply).not.toBeNull();
    expect(reply!.length).toBeLessThan(900);
  });

  it("news queries return panel UI guide — no headline dump", () => {
    const reply = buildCapabilityLiveReply(
      "חפש חדשות בנושא סייבר",
      ["news"],
      [
        src({
          provider: "grovee-news",
          label: "חדשות (GROVEE NEWS)",
          text: "ANSWER (headline): [BBC] Cyber headline\n[BBC] 1. Cyber headline",
        }),
      ],
    );
    expect(reply).toContain("כרטיסיות");
    expect(reply).toContain("סכם כתבה");
    expect(reply).not.toMatch(/Sources:/i);
    expect(reply).not.toMatch(/Breaking world/i);
  });

  it("shouldDeliverStructuredLiveReply is false for news", () => {
    expect(
      shouldDeliverStructuredLiveReply(
        "מה הכותרות בעולם",
        ["news"],
        [
          {
            provider: "grovee-news",
            label: "חדשות (BBC)",
            ok: true,
            text: "ANSWER (headline): [BBC] x",
            latencyMs: 1,
          },
        ],
        "canned",
      ),
    ).toBe(false);
  });

  it("shouldDeliverStructuredLiveReply is true when canned reply exists", () => {
    const reply = buildCapabilityLiveReply(
      "מה שער הדולר מול השקל כרגע?",
      ["currency"],
      [
        src({
          provider: "frankfurter-fx",
          label: "Frankfurter",
          text: "תאריך: 2026-06-12\n1 USD = 2.9207 ILS",
        }),
      ],
    );
    expect(reply).toMatch(/DATA AGE|2\.9207/);
  });

  it("returns AWACS count from live snapshot when available", () => {
    clearLiveWorldSnapshotCache();
    setLiveWorldSnapshot({
      fetchedAt: Date.now(),
      source: "globe",
      aviation: {
        count: 2,
        regionLabel: "עולם חי",
        sample: [],
        items: [
          { callsign: "NATO02", isMilitary: true, milLabel: "NATO", awacsSuspect: true },
          { callsign: "ELY1", isMilitary: false },
        ],
      },
    });

    const reply = buildCapabilityLiveReply(
      "כמה מטוסי AWACS פעילים כרגע?",
      ["aviation"],
      [src({ text: "ANSWER (AWACS): 0" })],
    );
    expect(reply).toMatch(/AWACS|NATO|עולם חי/i);
    expect(reply).not.toMatch(/לא ניתן לספור AWACS/);
  });

  it("returns cross-source canned reply for storm + aircraft question", () => {
    const reply = buildCapabilityLiveReply(
      "האם יש סופה באזור ישראל וגם מטוסים",
      ["weather", "aviation"],
      [
        src({
          provider: "open-meteo",
          label: "מזג אוויר",
          text: "מיקום: Israel\nמצב: סופת רעמים\nרוח: 50 km/h",
        }),
        src({
          provider: "adsb-aviation",
          label: "ADS-B",
          text: "אזור: ישראל\nמטוסים בטווח: 22",
        }),
      ],
      { answerShape: "short_fact", regionLabel: "ישראל" },
    );
    expect(reply).toMatch(/22|סופה|מז"א/i);
    expect(reply).toContain("Sources:");
  });

  it("returns structured weather reply with exact temperature", () => {
    const reply = buildCapabilityLiveReply(
      "מה מזג האוויר בגרמניה?",
      ["weather"],
      [
        src({
          provider: "open-meteo",
          label: "מזג אוויר (Open-Meteo)",
          text: [
            "מיקום: Germany, DE",
            "מצב: מעונן חלקית",
            "טמפרטורה: 18°C (מרגיש 17°C)",
            "לחות: 62%",
            "רוח: 12 km/h, כיוון 180°",
          ].join("\n"),
        }),
      ],
    );
    expect(reply).toMatch(/18°C/);
    expect(reply).toMatch(/גרמניה|Germany/i);
    expect(reply).not.toMatch(/להזין|placeholder|\[כאן/i);
    expect(reply).toContain("Sources:");
  });

  it("returns earthquake list for strong recent quakes query", () => {
    const reply = buildCapabilityLiveReply(
      "ספר לי על רעידות אדמה חזקות שהיו לאחרונה",
      ["earthquake"],
      [
        src({
          provider: "usgs-earthquake",
          label: "רעידות אדמה (USGS)",
          text: [
            'סה"כ 12 רעידות מעל M5 ב-24 שעות (USGS). 3 הגדולות:',
            "הרעידה האחרונה: M6.2 · 67 km ESE of Pondaguitan, Philippines · 2026-06-15 09:18:38 UTC",
            "- M6.2 · 67 km ESE of Pondaguitan, Philippines · 2026-06-15 09:18:38 UTC",
          ].join("\n"),
        }),
      ],
    );
    expect(reply).toMatch(/M6\.2/);
    expect(reply).toMatch(/Philippines/);
    expect(reply).toMatch(/Sources:.*USGS/);
    expect(reply).not.toMatch(/\[N\/A\]/);
  });

  it("shouldDeliverStructuredLiveReply is true when canned reply exists", () => {
    expect(
      shouldDeliverStructuredLiveReply("מה הטמפרטורה בחיפה", ["weather"], [
        {
          provider: "open-meteo",
          label: "מזג אוויר",
          ok: true,
          text: "מיקום: חיפה\nטמפרטורה: 22°C",
          latencyMs: 1,
        },
      ], "כרגע בחיפה: 22°C"),
    ).toBe(true);
  });

  it("web fallback failure returns honest canned reply", () => {
    const reply = buildWebFallbackNoDataReply("מה קורה בעולם הרובוטיקה?", [
      src({
        provider: "searxng",
        label: "SearXNG",
        ok: false,
        text: "",
        error: "SearXNG לא מוגדר — הגדר VITE_SEARXNG_URL",
      }),
    ]);
    expect(reply).toMatch(/לא הצלחתי|fetch failed/i);
    expect(reply).toMatch(/VITE_SEARXNG_URL/);
    expect(reply).not.toMatch(/פיתוחים משמעותיים/);
  });

  it("topical gaming uses multi-source bullets not fluff", () => {
    const reply = buildCapabilityLiveReply(
      "מה חדש בעולם הגיימינג?",
      ["hackernews", "github"],
      [
        src({
          provider: "hacker-news",
          label: "Hacker News",
          text: "1. Steam Deck update (★420) https://example.com",
        }),
        src({
          provider: "github",
          label: "GitHub Repositories",
          text: "1. user/game-engine ★12000",
        }),
      ],
      { answerShape: "overview" },
    );
    expect(reply).toMatch(/Steam Deck|Hacker News/i);
    expect(reply).toMatch(/GitHub|game-engine/i);
    expect(reply).toMatch(/Sources:.*Hacker News.*GitHub/i);
    expect(reply).not.toMatch(/פלטפורמות מרכזיות/);
  });
});
