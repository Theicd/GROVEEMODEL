import { describe, expect, it } from "vitest";
import { fetchAirQualitySearch } from "./providers/openMeteoAirQuality";
import { fetchArxivSearch } from "./providers/arxiv";
import { resolveCountry } from "./providers/restCountries";
import { runWebSearch } from "./orchestrator";
import { clearQueryCache } from "./queryCache";

const LIVE_TIMEOUT_MS = 30_000;

describe.sequential("Phase 3B live provider probes", () => {
  it(
    "Open-Meteo Air Quality returns AQI for Tel Aviv",
    async () => {
      clearQueryCache();
      const r = await fetchAirQualitySearch("מה איכות האוויר בתל אביב?");
      expect(r.ok, r.error).toBe(true);
      expect(r.text).toMatch(/US AQI|PM2\.5/i);
    },
    LIVE_TIMEOUT_MS,
  );

  it(
    "arXiv returns papers for transformer query",
    async () => {
      const r = await fetchArxivSearch("חפש מאמרים על transformer ב-arxiv");
      expect(r.ok, r.error).toBe(true);
      expect(r.text).toMatch(/ANSWER \(arxiv top\)|transformer/i);
    },
    LIVE_TIMEOUT_MS,
  );

  it(
    "Wikidata country fallback resolves Germany capital",
    async () => {
      const c = await resolveCountry("Germany");
      expect(c).not.toBeNull();
      expect(c?.capital).toMatch(/Berlin|ברלין/i);
    },
    LIVE_TIMEOUT_MS,
  );

  it(
    "orchestrator routes aviation synonym live",
    async () => {
      const r = await runWebSearch("מה העומס בשמי ישראל?");
      expect(r.intents).toContain("aviation");
      expect(r.sources.some((s) => s.provider === "adsb-aviation" && s.ok)).toBe(true);
    },
    LIVE_TIMEOUT_MS,
  );
});
