import { describe, expect, it } from "vitest";
import {
  formatSessionSummaryForPrompt,
  shouldRefreshSessionSummary,
  updateRollingSessionSummary,
} from "./chatSessionSummary";

describe("chatSessionSummary", () => {
  it("formats summary block for system prompt", () => {
    const block = formatSessionSummaryForPrompt("User was late; car broke down.");
    expect(block).toContain("Earlier in this chat");
    expect(block).toContain("car broke");
  });

  it("returns empty for blank summary", () => {
    expect(formatSessionSummaryForPrompt("")).toBe("");
  });

  it("builds rolling summary from long chat", () => {
    const msgs = [
      { role: "user", content: "איחרתי לעבודה הבוס כועס" },
      { role: "assistant", content: "מצטער לשמוע." },
      { role: "user", content: "נתקע לי הרכב" },
      { role: "assistant", content: "זה מתסכל." },
    ];
    for (let i = 0; i < 20; i++) {
      msgs.push(
        { role: "user", content: `שאלה ${i}` },
        { role: "assistant", content: `תשובה ${i}` },
      );
    }
    const summary = updateRollingSessionSummary("", msgs, { keepRecent: 8 });
    expect(summary.length).toBeGreaterThan(20);
    expect(summary).toMatch(/איחר|נתקע|Topics/i);
  });

  it("shouldRefresh when over keep window and enough turns", () => {
    expect(shouldRefreshSessionSummary(20, 4)).toBe(true);
    expect(shouldRefreshSessionSummary(10, 4)).toBe(false);
  });
});
