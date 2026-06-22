import { describe, expect, it } from "vitest";
import { extractLocationPhrase } from "./queryExtract";
import { isWeatherQuery } from "./intents";

describe("weekly weather queries", () => {
  it("detects weekly weather intent", () => {
    expect(isWeatherQuery("מה תחזית מזג האוויר לשבוע?")).toBe(true);
  });

  it("extracts location from weekly forecast with city", () => {
    expect(extractLocationPhrase("תחזית מזג האוויר לשבוע בתל אביב")).toBeTruthy();
  });
});
