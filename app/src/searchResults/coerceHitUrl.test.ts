import { describe, expect, it } from "vitest";
import { coerceHttpUrl, coerceText } from "./coerceHitUrl";

describe("coerceHitUrl", () => {
  it("returns trimmed http string", () => {
    expect(coerceHttpUrl("  https://example.com/x  ", "fallback")).toBe("https://example.com/x");
  });

  it("extracts url from nested object", () => {
    expect(
      coerceHttpUrl({ report: "https://www.gdacs.org/report.aspx?eventid=123" }, "fallback"),
    ).toBe("https://www.gdacs.org/report.aspx?eventid=123");
  });

  it("falls back when value is not a url", () => {
    expect(coerceHttpUrl({ foo: 1 }, "https://www.gdacs.org")).toBe("https://www.gdacs.org");
    expect(coerceHttpUrl(null, "https://earthquake.usgs.gov")).toBe("https://earthquake.usgs.gov");
  });

  it("coerceText handles non-strings", () => {
    expect(coerceText(42)).toBe("42");
    expect(coerceText(undefined, "—")).toBe("—");
  });
});
