import { describe, expect, it } from "vitest";
import { buildLocalTextSystemPrompt, localTextMaxNewTokens } from "./localTextSystemPrompt";
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
};

describe("localTextSystemPrompt", () => {
  it("includes web context when provided", () => {
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
    expect(prompt).toContain("WEB CONTEXT");
    expect(prompt).toContain("sunny");
    expect(prompt).toContain("ground truth");
  });

  it("adds game grounding block when games found", () => {
    const prompt = buildLocalTextSystemPrompt({
      uiLang: "en",
      prelude: { ...basePrelude, gameGroundingBlock: "1. Tetris" },
      pendingWebSearch: null,
      startupContext: null,
      webContext: "",
    });
    expect(prompt).toContain("Tetris");
    expect(prompt).toContain("ONLINE GAMES");
  });

  it("uses higher token budget on search turns from settings", () => {
    const custom = { ...DEFAULT_LOCAL_TEXT_SETTINGS, maxNewTokensSearch: 512 };
    expect(localTextMaxNewTokens({ ...basePrelude, shouldRunWebSearch: true }, custom)).toBe(512);
    expect(localTextMaxNewTokens({ ...basePrelude, greeting: true }, custom)).toBe(
      custom.maxNewTokensGreeting,
    );
    expect(localTextMaxNewTokens(basePrelude, custom)).toBe(custom.maxNewTokens);
  });
});
