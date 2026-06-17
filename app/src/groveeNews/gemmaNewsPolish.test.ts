import { describe, expect, it } from "vitest";
import {
  buildGemmaNewsPolishUserPrompt,
  cleanGemmaNewsPolishOutput,
} from "./gemmaNewsPolish";

describe("gemmaNewsPolish", () => {
  it("builds user prompt with title and notes", () => {
    const prompt = buildGemmaNewsPolishUserPrompt("Title: Foo\nSummary: Bar", "כתבה מ-Nature");
    expect(prompt).toContain("כתבה מ-Nature");
    expect(prompt).toContain("Title: Foo");
    expect(prompt).toContain("לניסוח מחדש");
  });

  it("strips prompt leakage and keeps כותרת/תקציר", () => {
    const raw = `(One Clear Headline Line)
תקציר (Two or Three Short Fluent Sentences)
Rules:
- No Bullet Points, No Numbers
כותרת: לייזר משפר מיקרוסקופיה
תקציר: צוות בביוהאב פיתח טכנולוגיה חדשה. היא משפרת תמונות של מבני חלבון.`;
    const out = cleanGemmaNewsPolishOutput(raw);
    expect(out).toContain("כותרת:");
    expect(out).toContain("תקציר:");
    expect(out).not.toMatch(/One Clear Headline/i);
    expect(out).not.toMatch(/No Bullet Points/i);
  });
});
