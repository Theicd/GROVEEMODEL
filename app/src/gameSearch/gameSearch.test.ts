import { describe, expect, it } from "vitest";
import { buildOnlineArchiveQuery, buildTitleSearchQuery } from "./archiveQueries";
import { extractDecadeRange, resolveGameSearch } from "./gameAliases";
import {
  detectGameCategory,
  extractGameQuery,
  extractUserIntentPrefix,
  isGameSearchRequest,
  isTextCompositionRequest,
  parseGameUserRequest,
  shouldOpenGamePanel,
} from "./gameIntents";
import { buildGameSearchFoundReply, buildGameSearchNotFoundReply } from "./gameReplyMessages";

describe("gameIntents", () => {
  it("detects bored / game requests", () => {
    expect(isGameSearchRequest("משעמם לי תמצא משחקים און ליין")).toBe(true);
    expect(isGameSearchRequest("hello")).toBe(false);
  });

  it("does not treat thinking-game chat as archive search", () => {
    expect(isGameSearchRequest("בוא נשחק משחק חשיבה")).toBe(false);
    expect(isGameSearchRequest("let's play a thinking game")).toBe(false);
    expect(shouldOpenGamePanel("בוא נשחק משחק חשיבה", "")).toBe(false);
  });

  it("detects decade and recommendation browse requests", () => {
    expect(isGameSearchRequest("חפש משחקים משנות ה80 ותציג אותם")).toBe(true);
    expect(isGameSearchRequest("האם יש משחקים מומלצים?")).toBe(true);
    expect(isGameSearchRequest("חפש משחקים משנות ה80 בקטגוריית ארקייד")).toBe(true);
  });

  it("parses arcade category from Hebrew bored message", () => {
    const r = parseGameUserRequest("משעמם לי, משחקי ארקייד");
    expect(r.category).toBe("arcade");
    expect(r.browseMode).toBe(true);
  });

  it("extracts game title from 'חפש את המשחק …'", () => {
    expect(extractGameQuery("חפש את המשחק עולם אחר")).toBe("another world");
    expect(extractGameQuery("חפש את המשחק הנסיך הפרסי משעמם לי אני רוצה לשחק")).toBe(
      "prince of persia",
    );
    expect(extractGameQuery("חפש את המשחק DUNE")).toBe("dune");
  });

  it("parses 80s arcade browse", () => {
    const r = parseGameUserRequest("חפש משחקים משנות ה80 בקטגוריית ארקייד");
    expect(r.browseMode).toBe(true);
    expect(r.category).toBe("arcade");
    expect(r.yearFrom).toBe(1980);
    expect(r.yearTo).toBe(1989);
  });

  it("parses recommended games", () => {
    const r = parseGameUserRequest("האם יש משחקים מומלצים?");
    expect(r.category).toBe("featured");
    expect(r.browseMode).toBe(true);
    expect(r.panelTitle).toContain("מומלצים");
  });

  it("parses racing category browse from Hebrew", () => {
    const r = parseGameUserRequest("חפש משחקי מירוצים והצג אותם");
    expect(r.category).toBe("racing");
    expect(r.browseMode).toBe(true);
    expect(r.query).toBe("");
  });

  it("parses recommended games question", () => {
    const r = parseGameUserRequest("אילו משחקים ממולצים יש");
    expect(r.category).toBe("featured");
    expect(r.browseMode).toBe(true);
  });

  it("detects action category", () => {
    expect(detectGameCategory("הראה משחקי אקשן")).toBe("action");
  });

  it("parses police quest alias with 80s", () => {
    const r = parseGameUserRequest("חפש משחק מחלק הבירות משנות ה80");
    expect(r.query).toContain("police quest");
    expect(r.yearFrom).toBe(1980);
  });

  it("detects PS1 category from Hebrew", () => {
    expect(detectGameCategory("תביא משחקי פלייסטיישן")).toBe("ps1");
  });

  it("shouldOpenGamePanel for bored_play topic", () => {
    expect(shouldOpenGamePanel("שלום", "bored_play")).toBe(false);
    expect(shouldOpenGamePanel("משעמם לי", "bored_play")).toBe(true);
  });

  it("does not treat text composition as game search when payload mentions games", () => {
    const msg = `נסח את ההוראה הזו מחדש

Design a high-end brand official משחק ארקייד בקובץ HTML יחיד בסגנון שנות ה80

Visual Strategy:
Imagery: athletic poses.
Color Palette: neon orange.
Overall Vibe: professional, hardcore.`;
    expect(extractUserIntentPrefix(msg)).toBe("נסח את ההוראה הזו מחדש");
    expect(isTextCompositionRequest(msg)).toBe(true);
    expect(isGameSearchRequest(msg)).toBe(false);
    expect(shouldOpenGamePanel(msg, "general")).toBe(false);
    expect(shouldOpenGamePanel(msg, "bored_play")).toBe(false);
  });

  it("does not match bare game words inside long pasted content", () => {
    const doc =
      "This spec describes a משחק ארקייד HTML page inspired by 80s arcade aesthetics and quest design.";
    expect(isGameSearchRequest(doc)).toBe(false);
  });
});

describe("gameReplyMessages", () => {
  it("found reply mentions right panel", () => {
    const r = buildGameSearchFoundReply(12, {
      panelTitle: "משחקים און־ליין",
      category: "featured",
      browseMode: true,
      query: "",
    });
    expect(r).toContain("12");
    expect(r).toContain("ימין");
    expect(r).toContain("שחק");
  });

  it("not found reply mentions categories", () => {
    const r = buildGameSearchNotFoundReply({ panelTitle: "x", query: "xyz" });
    expect(r).toContain("xyz");
    expect(r).toContain("קטגוריה");
  });
});

describe("gameAliases", () => {
  it("extracts 80s decade range", () => {
    const d = extractDecadeRange("משחקים משנות השמונים");
    expect(d?.yearFrom).toBe(1980);
    expect(d?.yearTo).toBe(1989);
  });

  it("maps Hebrew Wolfenstein slang to search term", () => {
    const r = resolveGameSearch("הטירה הנאית", null);
    expect(r.query).toContain("wolfenstein");
  });
});

describe("buildOnlineArchiveQuery", () => {
  it("adds year range filter", () => {
    const q = buildOnlineArchiveQuery({ category: "arcade", yearFrom: 1980, yearTo: 1989 });
    expect(q).toContain("1980");
    expect(q).toContain("1989");
  });

  it("buildTitleSearchQuery with year range", () => {
    const q = buildTitleSearchQuery("wolfenstein", null, 1980, 1989);
    expect(q).toContain("1980");
  });
});
