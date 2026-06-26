import { describe, expect, it } from "vitest";
import { tmdbFallbackLocale, tmdbLocaleForUi } from "./tmdbLocale";

describe("tmdbLocale", () => {
  it("maps UI language to TMDB locale", () => {
    expect(tmdbLocaleForUi("he")).toBe("he-IL");
    expect(tmdbLocaleForUi("en")).toBe("en-US");
  });

  it("falls back to English only from Hebrew", () => {
    expect(tmdbFallbackLocale("he-IL")).toBe("en-US");
    expect(tmdbFallbackLocale("en-US")).toBeNull();
  });
});
