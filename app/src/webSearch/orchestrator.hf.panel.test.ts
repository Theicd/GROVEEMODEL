import { describe, expect, it } from "vitest";
import { mergeSourcesToHits } from "../searchResults/mergeSearchHits";
import { runWebSearch } from "./orchestrator";

describe("panel HF integration", () => {
  it("panel search always includes huggingface-models source", async () => {
    const result = await runWebSearch("weather tel aviv", {
      panelSearch: true,
      plan: { useWebFallback: true, blendNewsWithWeb: true, queries: ["weather tel aviv"] },
    });
    const hf = result.sources.find((s) => s.provider === "huggingface-models");
    expect(hf).toBeDefined();
  }, 45_000);

  it("maps huggingface-models to hfmodel cards", async () => {
    const result = await runWebSearch("huggingface qwen", {
      panelSearch: true,
      plan: { useWebFallback: true, blendNewsWithWeb: true, queries: ["huggingface qwen"] },
    });
    const hf = result.sources.find((s) => s.provider === "huggingface-models" && s.ok);
    expect(hf?.hfModelHits?.length).toBeGreaterThan(0);
    const hits = mergeSourcesToHits(result.sources, "huggingface qwen");
    expect(hits.some((h) => h.kind === "hfmodel")).toBe(true);
  }, 45_000);
});
