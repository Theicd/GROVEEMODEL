// @vitest-environment jsdom
import { describe, expect, it } from "vitest";
import { getAisStreamApiKey, listApiKeyEntries, setAisStreamApiKey } from "./apiKeyStore";

describe("apiKeyStore", () => {
  it("stores and masks AISStream key", () => {
    setAisStreamApiKey("abcd1234efgh5678");
    expect(getAisStreamApiKey()).toBe("abcd1234efgh5678");
    const entry = listApiKeyEntries().find((e) => e.id === "aisstream");
    expect(entry?.configured).toBe(true);
    expect(entry?.masked).toMatch(/abcd…5678/);
    setAisStreamApiKey("");
    expect(getAisStreamApiKey()).toBeUndefined();
  });
});
