import { describe, expect, it } from "vitest";
import { enrichHfModelsSearch } from "./hfModelEnrich";

describe("hfModelEnrich live", () => {
  it("returns hub models for huggingface qwen query", async () => {
    const hits = await enrichHfModelsSearch("huggingface qwen instruct");
    expect(hits.length).toBeGreaterThan(0);
    expect(hits.some((h) => /qwen/i.test(h.modelId))).toBe(true);
    expect(hits[0]?.curlSnippet).toContain("curl ");
  }, 30_000);
});
