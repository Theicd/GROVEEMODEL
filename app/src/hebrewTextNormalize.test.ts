import { describe, expect, it } from "vitest";
import { normalizeHebrewChatText, normalizeHebrewIfNeeded } from "./hebrewTextNormalize";

describe("hebrewTextNormalize", () => {
  it("fixes עבדוה typo to עבודה", () => {
    expect(normalizeHebrewChatText("למה אני מאחר לעבדוה?")).toBe("למה אני מאחר לעבודה?");
    expect(normalizeHebrewChatText("בעבדוה")).toBe("בעבודה");
  });

  it("expands vague follow-up phrases", () => {
    expect(normalizeHebrewChatText("וזה הסיבה שאני מאחר")).toContain("בגלל");
  });

  it("skips normalization for English UI", () => {
    expect(normalizeHebrewIfNeeded("hello", "en")).toBe("hello");
  });

  it("normalizes Hebrew when UI is Hebrew", () => {
    expect(normalizeHebrewIfNeeded("לעבדוה", "he")).toBe("לעבודה");
  });
});
