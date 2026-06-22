import { describe, expect, it } from "vitest";
import {
  DEFAULT_LOCAL_TEXT_SETTINGS,
  localTextBaseSystemForUi,
  mergeLocalTextSettings,
} from "./localTextModelSettings";

describe("localTextModelSettings", () => {
  it("merges partial settings with defaults", () => {
    const merged = mergeLocalTextSettings({ maxNewTokens: 512, temperature: 0.5 });
    expect(merged.maxNewTokens).toBe(512);
    expect(merged.temperature).toBe(0.5);
    expect(merged.historyTurns).toBe(DEFAULT_LOCAL_TEXT_SETTINGS.historyTurns);
  });

  it("clamps history turns", () => {
    expect(mergeLocalTextSettings({ historyTurns: 99 }).historyTurns).toBe(24);
    expect(mergeLocalTextSettings({ historyTurns: 0 }).historyTurns).toBe(2);
  });

  it("adds English-only hint for Hebrew UI when missing", () => {
    const base = localTextBaseSystemForUi("he", {
      ...DEFAULT_LOCAL_TEXT_SETTINGS,
      systemPrompt: "You are helpful.",
    });
    expect(base).toContain("English only");
  });
});
