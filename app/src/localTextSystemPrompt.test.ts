import { describe, expect, it } from "vitest";
import { buildLocalTextSystemPrompt, LOCAL_TEXT_MAX_SYSTEM_CHARS, localTextMaxNewTokens } from "./localTextSystemPrompt";
import type { ChatTurnPreludeContinue } from "./chatTurnPrelude";
import { DEFAULT_LOCAL_TEXT_SETTINGS } from "./modelRack/localTextModelSettings";

const basePrelude: ChatTurnPreludeContinue = {
  webContext: "",
  searchHint: "",
  gameSearchHint: "",
  gameGroundingBlock: "",
  gameNoResults: false,
  globePlaceLabel: "",
  shouldRunWebSearch: false,
  localTimeOnly: false,
  greeting: false,
  imageDescribeMode: false,
  conversationalTurn: false,
};

describe("localTextSystemPrompt", () => {
  it("keeps compact prompt for SmolLM with live web context", () => {
    const prompt = buildLocalTextSystemPrompt({
      uiLang: "en",
      prelude: { ...basePrelude, shouldRunWebSearch: true },
      pendingWebSearch: {
        sources: [{ provider: "searxng", label: "Web", ok: true, text: "data", latencyMs: 1 }],
        summary: "ok",
        query: "test",
      },
      startupContext: null,
      webContext: "Live weather: sunny",
    });
    expect(prompt).toContain("sunny");
    expect(prompt).toContain("Use only these live facts");
    expect(prompt.length).toBeLessThanOrEqual(LOCAL_TEXT_MAX_SYSTEM_CHARS);
  });

  it("includes Groovie identity in he UI prompt", () => {
    const prompt = buildLocalTextSystemPrompt({
      uiLang: "he",
      prelude: { ...basePrelude, greeting: true },
      pendingWebSearch: null,
      startupContext: null,
      webContext: "",
    });
    expect(prompt).toContain("Groovie");
    expect(prompt).toContain("greeting");
    expect(prompt.length).toBeLessThanOrEqual(LOCAL_TEXT_MAX_SYSTEM_CHARS);
  });

  it("adds compact game grounding block when games found", () => {
    const prompt = buildLocalTextSystemPrompt({
      uiLang: "en",
      prelude: { ...basePrelude, gameGroundingBlock: "1. Tetris" },
      pendingWebSearch: null,
      startupContext: null,
      webContext: "",
    });
    expect(prompt).toContain("Tetris");
    expect(prompt).toContain("inline in chat");
  });

  it("skips failed-search warning on conversational turns", () => {
    const prompt = buildLocalTextSystemPrompt({
      uiLang: "he",
      prelude: { ...basePrelude, shouldRunWebSearch: false, conversationalTurn: true },
      pendingWebSearch: null,
      startupContext: null,
      webContext: "",
    });
    expect(prompt).toContain("שיחה חופשית");
    expect(prompt).not.toContain("Do not invent facts");
  });

  it("uses higher token budget on search turns from settings", () => {
    const custom = { ...DEFAULT_LOCAL_TEXT_SETTINGS, maxNewTokensSearch: 512 };
    expect(localTextMaxNewTokens({ ...basePrelude, shouldRunWebSearch: true }, custom)).toBe(320);
    expect(localTextMaxNewTokens({ ...basePrelude, greeting: true }, custom)).toBe(
      custom.maxNewTokensGreeting,
    );
    expect(localTextMaxNewTokens(basePrelude, custom)).toBe(custom.maxNewTokens);
  });
});
