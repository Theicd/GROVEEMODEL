import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("./hfApiScannerClient", () => ({
  isHfScannerAvailable: vi.fn().mockResolvedValue(false),
  testModelViaScanner: vi.fn(),
  fetchWorkingModelsFromScanner: vi.fn().mockResolvedValue([]),
  filterScannerModelsByQuery: vi.fn().mockReturnValue([]),
}));

vi.mock("./hfModelProbe", () => ({
  probeHfModelInBrowser: vi.fn().mockResolvedValue({
    modelId: "Qwen/Qwen2.5-7B-Instruct",
    status: "WORKING",
    provider: "HF inference",
    accessMode: "FREE",
    latency: 0.8,
    endpoint: "https://router.huggingface.co/v1/chat/completions",
  }),
  STATUS_WORKING: "WORKING",
}));

vi.mock("../fetchJson", () => ({
  fetchJson: vi.fn(),
}));

import { fetchJson } from "../fetchJson";
import { enrichHfModelsSearch } from "./hfModelEnrich";

const fetchMock = vi.mocked(fetchJson);

describe("hfModelEnrich", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    fetchMock.mockResolvedValue([
      {
        id: "Qwen/Qwen2.5-7B-Instruct",
        downloads: 50000,
        likes: 120,
        pipeline_tag: "text-generation",
      },
    ]);
  });

  it("returns enriched hits with probe metadata", async () => {
    const hits = await enrichHfModelsSearch("huggingface qwen instruct");
    expect(hits.length).toBeGreaterThan(0);
    expect(hits[0]?.curlSnippet).toContain("curl ");
    expect(hits[0]?.pythonSnippet).toContain("import requests");
    expect(hits[0]?.probed).toBe(true);
  });
});
