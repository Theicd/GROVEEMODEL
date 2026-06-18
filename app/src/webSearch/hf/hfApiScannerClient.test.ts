import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  filterScannerModelsByQuery,
  resetScannerHealthCache,
  scannerRowToProbe,
} from "./hfApiScannerClient";

vi.mock("../fetchJson", () => ({
  fetchJson: vi.fn(),
}));

vi.mock("./hfModelSettings", () => ({
  getHfScannerBaseUrl: vi.fn(() => "http://scanner.test"),
  getHfToken: vi.fn(),
}));

import { fetchJson } from "../fetchJson";
import { getHfScannerBaseUrl } from "./hfModelSettings";

const fetchMock = vi.mocked(fetchJson);

describe("hfApiScannerClient", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetScannerHealthCache();
  });

  it("maps scanner row to probe result", () => {
    const probe = scannerRowToProbe({
      model_id: "meta-llama/Llama-3.2-1B-Instruct",
      status: "WORKING",
      provider: "HF inference",
      access_mode: "FREE",
      latency: 1.2,
      endpoint: "https://router.huggingface.co/v1/chat/completions",
    });
    expect(probe.modelId).toBe("meta-llama/Llama-3.2-1B-Instruct");
    expect(probe.status).toBe("WORKING");
    expect(probe.accessMode).toBe("FREE");
  });

  it("filters working models by query terms", () => {
    const rows = [
      { model_id: "Qwen/Qwen2.5-7B-Instruct", downloads: 1000 },
      { model_id: "openai/gpt2", downloads: 5000 },
    ];
    const out = filterScannerModelsByQuery(rows, "qwen instruct");
    expect(out[0]?.model_id).toContain("Qwen");
  });

  it("returns false when scanner URL is not configured", async () => {
    vi.mocked(getHfScannerBaseUrl).mockReturnValueOnce(undefined);
    const { isHfScannerAvailable } = await import("./hfApiScannerClient");
    await expect(isHfScannerAvailable()).resolves.toBe(false);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("checks scanner health via fetchJson when URL is configured", async () => {
    const { isHfScannerAvailable } = await import("./hfApiScannerClient");
    fetchMock.mockResolvedValueOnce({ ok: true });
    await expect(isHfScannerAvailable()).resolves.toBe(true);
  });
});
