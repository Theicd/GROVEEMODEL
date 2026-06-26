// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from "vitest";
import {
  DEFAULT_AISSTREAM_API_KEY,
  getAisStreamApiKey,
  listApiKeyEntries,
  setAisStreamApiKey,
} from "./apiKeyStore";

describe("apiKeyStore", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("stores custom AISStream key", () => {
    setAisStreamApiKey("abcd1234efgh5678");
    expect(getAisStreamApiKey()).toBe("abcd1234efgh5678");
    const entry = listApiKeyEntries().find((e) => e.id === "aisstream");
    expect(entry?.configured).toBe(true);
    expect(entry?.masked).toMatch(/abcd…5678/);
  });

  it("falls back to built-in AISStream key when storage is empty", () => {
    expect(getAisStreamApiKey()).toBe(DEFAULT_AISSTREAM_API_KEY);
  });
});
