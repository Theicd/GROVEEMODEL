import { describe, expect, it } from "vitest";
import { prepareChatContext, capWebContext } from "./chatResourceBudget";

describe("chatResourceBudget", () => {
  it("caps web context", () => {
    const long = "x".repeat(2000);
    expect(capWebContext(long, 800).length).toBeLessThanOrEqual(800);
  });

  it("reduces stamina when history is long", () => {
    const history = Array.from({ length: 30 }, (_, i) => ({
      role: i % 2 === 0 ? ("user" as const) : ("assistant" as const),
      content: "שורה ".repeat(40),
    }));
    const prepared = prepareChatContext({
      history,
      webContext: "[SEARCH BRIEF]\n" + "fact ".repeat(200),
      systemPrompt: "system ".repeat(100),
      userPrompt: "שאלה",
      imageCount: 0,
      maxNewTokens: 768,
      profileId: "safe",
      isSearchTurn: true,
    });
    expect(prepared.staminaPercent).toBeLessThan(80);
    expect(prepared.maxNewTokens).toBeLessThanOrEqual(320);
    expect(prepared.history.length).toBeLessThanOrEqual(8);
  });
});
