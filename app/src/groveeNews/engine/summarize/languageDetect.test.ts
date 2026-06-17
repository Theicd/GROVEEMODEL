import { describe, expect, it } from "vitest";
import { isLikelyEnglish, isLikelyInLanguage, needsDisplayTranslation, needsEnglishDisplay } from "./languageDetect";

describe("languageDetect", () => {
  it("detects German headline as non-English", () => {
    const title =
      "Hannover: Polizisten wegen Vorwürfen der Körperverletzung und der Strafvereitelung im Amt suspendiert";
    const summary =
      "Gegen vier Beamte und eine Beamtin der Polizeidirektion Hannover laufen strafrechtliche Ermittlungen.";
    expect(isLikelyEnglish(title)).toBe(false);
    expect(isLikelyEnglish(summary)).toBe(false);
    expect(needsEnglishDisplay(title, summary)).toBe(true);
  });

  it("accepts normal English headlines", () => {
    expect(isLikelyEnglish("Police suspend officers amid misconduct probe in Hannover")).toBe(true);
  });

  it("detects Hebrew, Cyrillic, and CJK as non-English", () => {
    expect(isLikelyEnglish("משטרה עצרה חשודים בירושלים")).toBe(false);
    expect(isLikelyEnglish("Путин выступил с заявлением")).toBe(false);
    expect(isLikelyEnglish("中国发布新的经济政策")).toBe(false);
  });

  it("detects target language for display skip", () => {
    expect(isLikelyInLanguage("Bonjour le monde", "fr")).toBe(true);
    expect(isLikelyInLanguage("Police launch probe", "fr")).toBe(false);
    expect(needsDisplayTranslation("Police launch probe", "summary", "fr")).toBe(true);
    expect(needsDisplayTranslation("משטרה עצרה", "פרטים", "he")).toBe(false);
  });
});
