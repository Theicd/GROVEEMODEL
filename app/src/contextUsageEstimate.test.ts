import { describe, expect, it } from "vitest";
import {
  approxTokensFromChars,
  estimateLiveContextUsage,
  formatTokenCount,
} from "./contextUsageEstimate";

describe("contextUsageEstimate", () => {
  it("formats token counts like common AI UIs", () => {
    expect(formatTokenCount(887)).toBe("887");
    expect(formatTokenCount(5500)).toBe("5.5K");
  });

  it("approximates tokens from chars", () => {
    expect(approxTokensFromChars(22000)).toBe(5500);
  });

  it("includes draft prompt and history in live estimate", () => {
    const empty = estimateLiveContextUsage({
      history: [],
      draftPrompt: "",
      systemPromptChars: 2000,
      profileId: "ultra",
    });
    const withDraft = estimateLiveContextUsage({
      history: [],
      draftPrompt: "נסח את ההוראה הזו מחדש",
      systemPromptChars: 2000,
      profileId: "ultra",
    });
    expect(withDraft.usedChars).toBeGreaterThan(empty.usedChars);
    expect(withDraft.breakdown.user).toBeGreaterThan(0);
  });

  it("grows when conversation history grows", () => {
    const short = estimateLiveContextUsage({
      history: [{ role: "user", content: "היי" }],
      draftPrompt: "",
      systemPromptChars: 2000,
      profileId: "ultra",
    });
    const long = estimateLiveContextUsage({
      history: [
        { role: "user", content: "שאלה ".repeat(200) },
        { role: "assistant", content: "תשובה ".repeat(400) },
      ],
      draftPrompt: "",
      systemPromptChars: 2000,
      profileId: "ultra",
    });
    expect(long.usedChars).toBeGreaterThan(short.usedChars);
    expect(long.percent).toBeLessThan(short.percent);
    expect(long.breakdown.history).toBeGreaterThan(short.breakdown.history);
  });
});
