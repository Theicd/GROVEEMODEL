import { describe, expect, it, vi, beforeEach } from "vitest";
import { STATUS_WORKING } from "../webSearch/hf/hfModelProbe";

vi.mock("../webSearch/fetchJson", () => ({
  fetchJson: vi.fn(),
}));

vi.mock("../webSearch/hf/hfApiScannerClient", () => ({
  fetchWorkingModelsFromScanner: vi.fn(async () => []),
}));

vi.mock("./hfRackProbe", () => ({
  probeHfModelForRack: vi.fn(),
  probeHfModelFree: vi.fn(),
  mapWithConcurrency: async <T, R>(items: T[], _c: number, fn: (item: T) => Promise<R>) =>
    Promise.all(items.map(fn)),
}));

vi.mock("./pollinationsScan", () => ({
  scanCoreCloudImageModels: vi.fn(async () => [
    {
      id: "pollinations-flux",
      label: "FLUX",
      modality: "image",
      adapter: "pollinations",
      status: "ready",
      source: "cloud-scan",
      pollinationsModel: "flux",
      addedAt: 1,
    },
  ]),
}));

vi.mock("../webSearch/proxyFetch", () => ({
  proxyAwareFetch: vi.fn(async () => ({
    ok: true,
    headers: { get: () => null },
    json: async () => [],
  })),
}));

import { probeHfModelForRack } from "./hfRackProbe";
import { scanCoreCloudImageModels } from "./pollinationsScan";
import { scanFreeWorkingHfModels, refreshCloudModelRack } from "./modelRackScan";

describe("scanFreeWorkingHfModels", () => {
  beforeEach(() => {
    vi.mocked(probeHfModelForRack).mockReset();
  });

  it("probes multiple pipeline tags and keeps only FREE working models", async () => {
    const { proxyAwareFetch } = await import("../webSearch/proxyFetch");
    vi.mocked(proxyAwareFetch).mockImplementation(async (url: string) => {
      const models =
        url.includes("pipeline_tag=text-generation")
          ? [{ id: "free/chat-model", pipeline_tag: "text-generation", downloads: 1000 }]
          : url.includes("pipeline_tag=text-to-image")
            ? [{ id: "free/img-model", pipeline_tag: "text-to-image", downloads: 500 }]
            : [];
      return {
        ok: true,
        headers: { get: () => null },
        json: async () => models,
      } as Response;
    });

    vi.mocked(probeHfModelForRack).mockImplementation(async (modelId: string) => {
      const ok = modelId === "free/chat-model" || modelId === "free/img-model";
      return { ok, accessMode: ok ? "FREE" : "UNKNOWN" };
    });

    const results = await scanFreeWorkingHfModels();
    const ids = results.map((r) => r.hfModelId).sort();
    expect(ids).toEqual(["free/chat-model", "free/img-model"]);
    expect(results.every((r) => r.status === "ready")).toBe(true);
  });

  it("skips models that fail free probe", async () => {
    const { proxyAwareFetch } = await import("../webSearch/proxyFetch");
    vi.mocked(proxyAwareFetch).mockResolvedValue({
      ok: true,
      headers: { get: () => null },
      json: async () => [],
    } as Response);
    vi.mocked(probeHfModelForRack).mockResolvedValue({ ok: false, accessMode: "UNKNOWN" });

    const results = await scanFreeWorkingHfModels();
    expect(results).toHaveLength(0);
  }, 15_000);
});

describe("refreshCloudModelRack", () => {
  beforeEach(() => {
    vi.mocked(scanCoreCloudImageModels).mockClear();
  });

  it("merges builtin gemma with cloud scan results", async () => {
    const rack = await refreshCloudModelRack();
    expect(rack.some((r) => r.source === "builtin")).toBe(true);
    expect(rack.some((r) => r.source === "cloud-scan" && r.pollinationsModel === "flux")).toBe(true);
    expect(scanCoreCloudImageModels).toHaveBeenCalled();
  });
});

describe("rackEntry integration", () => {
  it("STATUS_WORKING constant matches probe contract", () => {
    expect(STATUS_WORKING).toBe("WORKING");
  });
});
