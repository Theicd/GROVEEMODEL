import { describe, expect, it } from "vitest";
import {
  GEMMA_NEWS_INSTRUCTION,
  GEMMA_NEWS_PROMPT_DELIM,
  GEMMA_NEWS_PROMPT_END,
  buildArticleExcerptForGemma,
  buildGemmaNewsPolishUserPrompt,
  cleanGemmaNewsPolishOutput,
  finalizeGemmaNewsSummary,
  sanitizeHebrewNewsOutput,
} from "./gemmaNewsPolish";

describe("gemmaNewsPolish", () => {
  it("builds excerpt from multiple paragraphs up to limit", () => {
    const body = [
      "Short.",
      "An artist critical of Russia's President Vladimir Putin was shot near his home in Poland.",
      "The 44-year-old victim was identified as Robert Kuzovkov by local media.",
      "Polish prosecutors said two Belarusian citizens were arrested after the killing.",
    ].join("\n\n");
    const excerpt = buildArticleExcerptForGemma(body, 500);
    expect(excerpt).toContain("Robert Kuzovkov");
    expect(excerpt).toContain("Belarusian");
    expect(excerpt).not.toContain("Short.");
  });

  it("builds delimited user prompt: נסח בעברית|!@!@!@ english !@!@!@", () => {
    const prompt = buildGemmaNewsPolishUserPrompt(
      "SpaceX launched 24 Starlink satellites on a Falcon 9 rocket.",
      "SpaceX launch",
    );
    expect(prompt).toBe(
      `${GEMMA_NEWS_INSTRUCTION}${GEMMA_NEWS_PROMPT_DELIM}
SpaceX launched 24 Starlink satellites on a Falcon 9 rocket.
${GEMMA_NEWS_PROMPT_END}`,
    );
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

  it("sanitizes mixed Cyrillic/Arabic script in Hebrew output", () => {
    const raw = `כותרת: שיגור לווייני סטארלינק
תקציר: חברת ספייס اكس שיגרה ракетת פל קון 9 עם 24 לוויינים.
חוקים: ракетת פל קון 9 של ספייס اكس העבירה 24 לוויינים.`;
    const out = cleanGemmaNewsPolishOutput(raw);
    expect(out).toContain("SpaceX");
    expect(out).toContain("Falcon 9");
    expect(out).not.toMatch(/[\u0400-\u04FF]/);
    expect(out).not.toMatch(/חוקים:/);
  });

  it("rejects English echo as failed summary", () => {
    const out = finalizeGemmaNewsSummary(
      "One of the biggest misconceptions surrounding artificial intelligence is that it eliminates the need for human thinking.",
    );
    expect(out).toContain("לא הצלחתי");
    expect(out).not.toContain("misconceptions");
  });

  it("sanitizeHebrewNewsOutput removes delimiter echoes", () => {
    expect(sanitizeHebrewNewsOutput("נסח בעברית|!@!@!@ טקסט !@!@!@")).toBe("נסח בעברית טקסט");
  });
});
