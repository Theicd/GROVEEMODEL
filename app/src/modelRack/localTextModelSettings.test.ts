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

  it("migrates legacy historyTurns 6 to 12", () => {
    expect(mergeLocalTextSettings({ historyTurns: 6 }).historyTurns).toBe(12);
  });

  it("uses GROVEE core identity for Hebrew UI", () => {
    const base = localTextBaseSystemForUi("he", {
      ...DEFAULT_LOCAL_TEXT_SETTINGS,
      systemPrompt: "You are helpful.",
    });
    expect(base).toContain("Groovie");
    expect(base).toContain("GROVEEMODEL");
  });
});
