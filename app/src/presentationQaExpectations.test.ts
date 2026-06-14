import { describe, expect, it } from "vitest";
import { autoGradePresentationQuery } from "./presentationQaExpectations";
import { USER_PRESENTATION_QUERIES } from "./userPresentationQueries";
import type { QaTurnResult } from "./qaChatBridge";

const q = (id: string) => USER_PRESENTATION_QUERIES.find((x) => x.id === id)!;

const turn = (over: Partial<QaTurnResult>): QaTurnResult => ({
  query: "",
  reply: "",
  replySource: "model",
  usedModel: true,
  webContextSent: "",
  modelPromptOut: "",
  modelResponseIn: "",
  searchProviders: [],
  searchSummary: "",
  ms: 100,
  ...over,
});

describe("autoGradePresentationQuery", () => {
  it("fails B01 without news context", () => {
    expect(
      autoGradePresentationQuery(
        q("B01"),
        turn({
          query: q("B01").prompt,
          reply: "העולם מלא באירועים גיאופוליטיים מורכבים שמשפיעים על כולנו.",
          webContextSent: "[WEB SEARCH — NO LIVE DATA]",
        }),
      ),
    ).toBe("fail");
  });

  it("fails B02 on Trevelyan hallucination", () => {
    expect(
      autoGradePresentationQuery(
        q("B02"),
        turn({
          query: q("B02").prompt,
          reply: "ראש ממשלת בריטניה הוא ג'סטין טרווורד.",
          webContextSent: "[SEARCH BRIEF]\nFACTS:\n- [Wikidata] Keir Starmer",
        }),
      ),
    ).toBe("fail");
  });

  it("fails B10 on invented AWACS count", () => {
    expect(
      autoGradePresentationQuery(
        q("B10"),
        turn({
          query: q("B10").prompt,
          reply: "ישנם 12 מטוסי AWACS פעילים מעל ניו יורק.",
          webContextSent: "ANSWER (AWACS): לא ניתן לספור",
        }),
      ),
    ).toBe("fail");
  });

  it("passes B06 with weather brief", () => {
    expect(
      autoGradePresentationQuery(
        q("B06"),
        turn({
          query: q("B06").prompt,
          reply: "בטוקיו כרגע 22 מעלות, מעונן חלקית לפי Open-Meteo.",
          webContextSent: "[SEARCH BRIEF — live data]\nFACTS:\n- [מזג אוויר] טמפרatura: 22°C",
          searchProviders: ["open-meteo"],
        }),
      ),
    ).toBe("pass");
  });

  it("partial on cross-source with single provider", () => {
    expect(
      autoGradePresentationQuery(
        q("C01"),
        turn({
          query: q("C01").prompt,
          reply: "יש מטוסים באזור עם סופה פעילה לפי נתוני תעופה.",
          webContextSent: "[SEARCH BRIEF]\nFACTS:\n- [ADS-B] 40 מטוסים",
          searchProviders: ["adsb-aviation"],
        }),
      ),
    ).toBe("partial");
  });

  it("partial B03 when DATA AGE present", () => {
    expect(
      autoGradePresentationQuery(
        q("B03"),
        turn({
          query: q("B03").prompt,
          reply: "1 USD = 2.9207 ILS לפי Frankfurter.",
          replySource: "canned-live",
          webContextSent: "[SEARCH BRIEF]\nDATA AGE: שער ECB מ-2026-06-12",
          searchProviders: ["frankfurter-fx"],
        }),
      ),
    ).toBe("partial");
  });

  it("passes B09 canned with aircraft count", () => {
    expect(
      autoGradePresentationQuery(
        q("B09"),
        turn({
          query: q("B09").prompt,
          reply: "מטוסים בטווח: 61 לפי ADS-B.",
          replySource: "canned-live",
          webContextSent: "[SEARCH BRIEF]\nANSWER (aircraft count): מטוסים בטווח: 61",
          searchProviders: ["adsb-aviation"],
        }),
      ),
    ).toBe("pass");
  });
});
