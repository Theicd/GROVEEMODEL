/**
 * Live acceptance tests — hits real free APIs (requires network).
 * Run: npm run test:search
 */
import { describe, it, expect, afterEach } from "vitest";
import { runWebSearch } from "./orchestrator";
import { userRequestsSearch } from "./intents";
import { ACCEPTANCE_QUERIES } from "./acceptanceQueries";

const LIVE_TIMEOUT_MS = 25_000;
const RATE_LIMIT_PAUSE_MS = 350;

describe.sequential("web search live acceptance", () => {
  afterEach(async () => {
    await new Promise((r) => setTimeout(r, RATE_LIMIT_PAUSE_MS));
  });
  for (const spec of ACCEPTANCE_QUERIES) {
    it(
      `[${spec.id}] ${spec.category}: ${spec.query}`,
      async () => {
        const result = await runWebSearch(spec.query);

        for (const intent of spec.expectIntents) {
          expect(result.intents, `${spec.id} intents`).toContain(intent);
        }

        const okProviders = result.sources.filter((s) => s.ok).map((s) => s.provider);
        const matched = spec.expectProvidersOk.some((p) => okProviders.includes(p));
        expect(
          matched,
          `${spec.id} expected one of [${spec.expectProvidersOk.join(", ")}] ok; got [${okProviders.join(", ")}]; failures: ${result.sources
            .filter((s) => !s.ok)
            .map((s) => `${s.provider}:${s.error}`)
            .join("; ")}`,
        ).toBe(true);

        expect(result.contextText.length, `${spec.id} contextText`).toBeGreaterThan(20);

        if (spec.expectTextIncludes?.length) {
          const blob = result.sources
            .filter((s) => s.ok)
            .map((s) => s.text)
            .join("\n")
            .toLowerCase();
          for (const needle of spec.expectTextIncludes) {
            if (!blob.includes(needle.toLowerCase())) {
              // Soft warning — provider data shape may vary
              console.warn(`${spec.id}: optional text "${needle}" not found`);
            }
          }
        }
      },
      LIVE_TIMEOUT_MS,
    );
  }

  it("userRequestsSearch detects Hebrew search verb", () => {
    expect(userRequestsSearch("חפש מידע על ברמודה")).toBe(true);
    expect(userRequestsSearch("שלום")).toBe(false);
  });

  it(
    "full run completes within reasonable time",
    async () => {
      const t0 = performance.now();
      const result = await runWebSearch("מה מזג האוויר בניו יורק");
      const elapsed = performance.now() - t0;
      expect(result.sources.some((s) => s.ok)).toBe(true);
      expect(elapsed).toBeLessThan(15_000);
    },
    LIVE_TIMEOUT_MS,
  );
});
