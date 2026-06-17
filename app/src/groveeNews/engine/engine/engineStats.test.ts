import { describe, expect, it } from "vitest";
import { countLanguages } from "./engineStats";

describe("engineStats", () => {
  it("countLanguages aggregates article language codes", () => {
    const langs = countLanguages([
      { language: "en" },
      { language: "en" },
      { language: "he" },
      { language: undefined },
    ]);
    expect(langs.find((l) => l.code === "en")?.count).toBe(2);
    expect(langs.find((l) => l.code === "he")?.count).toBe(1);
    expect(langs.find((l) => l.code === "multi")?.count).toBe(1);
  });
});
