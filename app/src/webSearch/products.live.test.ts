/**
 * Live integration tests — Israeli supermarket products + Cheapersal prices.
 * Run: npm run test:products  (requires network; CHEAPERSAL_API_KEY in app/.env for prices)
 */
import { afterEach, describe, expect, it } from "vitest";
import { runWebSearch } from "./orchestrator";
import { buildUnifiedSearchPayload } from "../searchResults/mergeSearchHits";
import { buildCapabilityLiveReply } from "./capabilityReplyMessages";
import { PRODUCT_ACCEPTANCE_QUERIES } from "./productsAcceptanceQueries";

const LIVE_TIMEOUT_MS = 60_000;
const RATE_LIMIT_PAUSE_MS = 400;

describe.sequential("products live integration", () => {
  afterEach(async () => {
    await new Promise((r) => setTimeout(r, RATE_LIMIT_PAUSE_MS));
  });

  for (const spec of PRODUCT_ACCEPTANCE_QUERIES) {
    it(
      `[${spec.id}] ${spec.query}`,
      async () => {
        const result = await runWebSearch(spec.query);

        for (const intent of spec.expectIntents) {
          expect(result.intents, `${spec.id} intents`).toContain(intent);
        }

        const prod = result.sources.find((s) => s.provider === spec.expectProvider);
        expect(prod, `${spec.id} provider`).toBeDefined();
        expect(prod?.ok, prod?.error ?? "provider failed").toBe(true);
        expect(prod?.productHits?.length, `${spec.id} hits`).toBeGreaterThan(0);

        const top = prod!.productHits![0];
        if (spec.expectImage) {
          expect(top.imageUrl, `${spec.id} image`).toBeTruthy();
        }
        if (spec.expectTitleIncludes) {
          expect(top.title).toMatch(new RegExp(spec.expectTitleIncludes, "i"));
        }
        if (spec.expectPrice && top.priceNis == null) {
          console.warn(
            `[${spec.id}] no price — add CHEAPERSAL_API_KEY to app/.env for live price checks`,
          );
        }
        if (spec.expectPrice && top.priceNis != null) {
          expect(top.priceNis).toBeGreaterThan(0);
        }

        const payload = buildUnifiedSearchPayload(spec.query, result.sources);
        expect(payload.facets.products).toBeGreaterThan(0);
        const row = payload.hits.find((h) => h.kind === "product");
        if (spec.expectImage) {
          expect(row?.imageUrl, `${spec.id} unified image`).toBeTruthy();
        }

        const reply = buildCapabilityLiveReply(spec.query, result.intents, result.sources);
        expect(reply?.trim(), `${spec.id} canned reply`).toBeTruthy();
        expect(reply).toMatch(/Sources:/);
      },
      LIVE_TIMEOUT_MS,
    );
  }
});
