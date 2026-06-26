/** @vitest-environment jsdom */
import { describe, expect, it } from "vitest";
import { fetchMjhXmltv, resetMjhEpgCacheForTests } from "./mjhSources";

describe("mjh fetch", () => {
  it(
    "decompresses all epg",
    async () => {
      resetMjhEpgCacheForTests();
      const xml = await fetchMjhXmltv("https://i.mjh.nz/all/epg.xml.gz");
      expect(xml).not.toBeNull();
      expect(xml!.length).toBeGreaterThan(1000);
      expect(xml).toContain("COPS");
    },
    60_000,
  );
});
