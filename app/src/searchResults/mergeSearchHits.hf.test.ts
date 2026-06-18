import { describe, expect, it } from "vitest";
import { mergeSourcesToHits } from "./mergeSearchHits";
import type { SearchSourceResult } from "../webSearch/types";

describe("mergeSourcesToHits HF models", () => {
  it("maps hfModelHits to hfmodel kind with connection meta", () => {
    const sources: SearchSourceResult[] = [
      {
        provider: "huggingface-models",
        label: "HF",
        ok: true,
        text: "",
        latencyMs: 10,
        hfModelHits: [
          {
            id: "hf-qwen",
            modelId: "Qwen/Qwen2.5-7B-Instruct",
            url: "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
            title: "Qwen/Qwen2.5-7B-Instruct",
            snippet: "Pipeline: text-generation",
            status: "WORKING",
            provider: "HF inference",
            accessMode: "FREE",
            endpoint: "https://router.huggingface.co/v1/chat/completions",
            curlSnippet: "curl example",
            pythonSnippet: "import requests",
            probed: true,
            probeSource: "browser",
            downloads: 1000,
            likes: 50,
          },
        ],
      },
    ];
    const hits = mergeSourcesToHits(sources, "huggingface qwen");
    expect(hits).toHaveLength(1);
    expect(hits[0]?.kind).toBe("hfmodel");
    expect(hits[0]?.meta?.hfStatus).toBe("WORKING");
    expect(hits[0]?.meta?.hfCurl).toContain("curl");
  });
});
