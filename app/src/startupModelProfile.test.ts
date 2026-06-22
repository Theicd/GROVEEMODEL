import { describe, expect, it, vi, beforeEach } from "vitest";
import {
  detectMobileDevice,
  recommendStartupModel,
  resolveStartupModelChoice,
  type StartupDeviceSignals,
} from "./startupModelProfile";

const desktopGpu: StartupDeviceSignals["webgpu"] = {
  available: true,
  isFallbackAdapter: false,
  vendor: "nvidia",
  architecture: "ampere",
  description: "gpu",
};

const weakSignals = (partial: Partial<StartupDeviceSignals> = {}): StartupDeviceSignals => ({
  deviceMemoryGb: 4,
  hardwareConcurrency: 4,
  isMobile: false,
  webgpu: desktopGpu,
  ...partial,
});

describe("startupModelProfile", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("recommends SmolLM on mobile", () => {
    const rec = recommendStartupModel(weakSignals({ isMobile: true, deviceMemoryGb: 8 }));
    expect(rec.choice).toBe("local-text");
    expect(rec.reasonHe).toContain("נייד");
  });

  it("recommends SmolLM when WebGPU missing", () => {
    const rec = recommendStartupModel(
      weakSignals({
        deviceMemoryGb: 16,
        webgpu: { ...desktopGpu, available: false },
      }),
    );
    expect(rec.choice).toBe("local-text");
  });

  it("recommends Gemma on strong desktop", () => {
    const rec = recommendStartupModel(
      weakSignals({ deviceMemoryGb: 8, hardwareConcurrency: 12, isMobile: false }),
    );
    expect(rec.choice).toBe("gemma");
  });

  it("honors user preference override", async () => {
    const rec = await resolveStartupModelChoice("local-text");
    expect(rec.choice).toBe("local-text");
    expect(rec.fromPreference).toBe(true);
  });

  it("detectMobileDevice matches phone UA", () => {
    vi.stubGlobal("navigator", { userAgent: "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)" });
    vi.stubGlobal("window", { innerWidth: 390, matchMedia: () => ({ matches: true }) });
    expect(detectMobileDevice()).toBe(true);
  });
});
