import { describe, expect, it } from "vitest";
import {
  buildGrooveCapabilitiesReply,
  disambiguateIntent,
  handleMetaIntent,
  handleResetCommand,
  isMetaCapabilitiesQuery,
  resolveEarlyTurnRouting,
} from "./chatRoutePrelude";

describe("chatRoutePrelude", () => {
  it("disambiguate: bored alone routes without clarify", () => {
    const r = disambiguateIntent("משעמם לי", "bored_play");
    expect(r.kind).toBe("route");
    if (r.kind === "route") {
      expect(r.preferLiveSearch).toBe(false);
    }
  });

  it("disambiguate: earthquake alone prefers live search", () => {
    const r = disambiguateIntent("תראה לי רעידות אדמה", "general");
    expect(r.kind).toBe("route");
    if (r.kind === "route") {
      expect(r.preferLiveSearch).toBe(true);
      expect(r.preferGames).toBe(false);
    }
  });

  it("disambiguate: bored + earthquakes asks clarify", () => {
    const r = disambiguateIntent("משעמם ורעידות", "bored_play");
    expect(r.kind).toBe("clarify");
  });

  it("meta intent returns capabilities without web search", () => {
    expect(isMetaCapabilitiesQuery("מה אתה יודע לעשות?")).toBe(true);
    const reply = handleMetaIntent("מה אתה יודע לעשות?", "he");
    expect(reply).toContain("GROVEE");
    expect(reply).toContain("רעידות");
  });

  it("meta intent: tell me about yourself", () => {
    expect(isMetaCapabilitiesQuery("ספר לי על עצמך")).toBe(true);
    expect(isMetaCapabilitiesQuery("tell me about yourself")).toBe(true);
    const reply = handleMetaIntent("ספר לי על עצמך", "he");
    expect(reply).toContain("GROVEE");
  });

  it("reset command detected", () => {
    expect(handleResetCommand("/reset", "he")).toContain("איפסתי");
    expect(handleResetCommand("איפוס שיחה", "he")).toContain("איפסתי");
  });

  it("resolveEarlyTurnRouting: meta before search", () => {
    const r = resolveEarlyTurnRouting({
      text: "מה אתה יכול לעשות?",
      effectivePrompt: "מה אתה יכול לעשות?",
      chatTopic: "general",
      uiLang: "he",
      startupContext: null,
      blockGames: false,
    });
    expect(r.action).toBe("canned");
    if (r.action === "canned") {
      expect(r.replySource).toBe("meta-capabilities");
    }
  });

  it("resolveEarlyTurnRouting: trivia stays on LLM not games", () => {
    const r = resolveEarlyTurnRouting({
      text: "בוא נשחק טריוויה במשחק מילים של מסע בין כוכבים",
      effectivePrompt: "בוא נשחק טריוויה במשחק מילים של מסע בין כוכבים",
      chatTopic: "bored_play",
      uiLang: "he",
      startupContext: null,
      blockGames: false,
    });
    expect(r.action).toBe("continue");
    if (r.action === "continue") {
      expect(r.wantsGameSearch).toBe(false);
    }
  });

  it("resolveEarlyTurnRouting: earthquake goes to search not games", () => {
    const r = resolveEarlyTurnRouting({
      text: "רעידת אדמה אחרונה",
      effectivePrompt: "רעידת אדמה אחרונה",
      chatTopic: "general",
      uiLang: "he",
      startupContext: null,
      blockGames: false,
    });
    expect(r.action).toBe("continue");
    if (r.action === "continue") {
      expect(r.wantsGameSearch).toBe(false);
      expect(r.shouldRunWebSearch).toBe(true);
    }
  });

  it("buildGrooveCapabilitiesReply english", () => {
    expect(buildGrooveCapabilitiesReply("en")).toContain("GROVEE");
  });

  it("resolveEarlyTurnRouting: שלום גרובי is canned greeting without search", () => {
    const r = resolveEarlyTurnRouting({
      text: "שלום גרובי",
      effectivePrompt: "שלום גרובי",
      chatTopic: "greeting",
      uiLang: "he",
      startupContext: null,
      blockGames: false,
    });
    expect(r.action).toBe("canned");
    if (r.action === "canned") {
      expect(r.replySource).toBe("greeting");
      expect(r.reply).toContain("גרובי");
    }
  });
});
