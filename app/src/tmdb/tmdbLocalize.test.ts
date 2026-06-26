import { describe, expect, it } from "vitest";
import { needsHebrewTranslation } from "./tmdbLocalize";

describe("needsHebrewTranslation", () => {
  it("skips empty and already-Hebrew text", () => {
    expect(needsHebrewTranslation("")).toBe(false);
    expect(needsHebrewTranslation("התחלה")).toBe(false);
    expect(needsHebrewTranslation("Inception — התחלה")).toBe(false);
  });

  it("flags Latin-only titles and overviews", () => {
    expect(needsHebrewTranslation("Criminal Minds")).toBe(true);
    expect(needsHebrewTranslation("A team of profilers analyzes crimes.")).toBe(true);
  });
});
