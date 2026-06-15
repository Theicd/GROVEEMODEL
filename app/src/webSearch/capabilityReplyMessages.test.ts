import { describe, expect, it } from "vitest";
import { clearLiveWorldSnapshotCache, setLiveWorldSnapshot } from "../liveWorld/snapshotStore";
import { buildCapabilityLiveReply, buildWebFallbackNoDataReply } from "./capabilityReplyMessages";
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

  it("returns news headline for B01", () => {
    const reply = buildCapabilityLiveReply(
      "מה הכותרת הראשית בעולם כרגע?",
      ["news"],
      [
        src({
          provider: "news-rss",
          label: "חדשות (RSS — BBC · CNN · Reuters · Guardian)",
          text: [
            "ANSWER (headline): [BBC] Breaking news headline",
            "מקורות RSS בינלאומיים (BBC · CNN · Reuters · Guardian):",
            "[BBC] 1. Breaking news headline",
            "[CNN] 1. CNN world headline",
            "[Reuters] 1. Reuters headline",
          ].join("\n"),
        }),
      ],
    );
    expect(reply).toMatch(/Breaking|BBC/i);
    expect(reply).toMatch(/CNN|Reuters|כותרות נוספות/i);
  });

  it("includes DATA AGE for stale FX", () => {
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
