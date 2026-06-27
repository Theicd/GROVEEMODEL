import { describe, expect, it, vi, beforeEach } from "vitest";
import {
  detectMobileDevice,
  quickStartupModelChoice,
  recommendStartupModel,
  resolveLocalTextBootBackend,
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

  it("forces WASM backend for SmolLM on mobile", () => {
    vi.stubGlobal("navigator", { userAgent: "Mozilla/5.0 (Linux; Android 14) Mobile" });
    vi.stubGlobal("window", { innerWidth: 412, matchMedia: () => ({ matches: true }) });
    expect(resolveLocalTextBootBackend("auto")).toBe("wasm");
    expect(resolveLocalTextBootBackend("webgpu")).toBe("wasm");
    expect(resolveLocalTextBootBackend("auto", { forceWasm: true })).toBe("wasm");
  });

  it("keeps desktop backend preference", () => {
    vi.stubGlobal("navigator", { userAgent: "Mozilla/5.0 Windows NT 10.0" });
    vi.stubGlobal("window", { innerWidth: 1280, matchMedia: () => ({ matches: false }) });
    expect(resolveLocalTextBootBackend("webgpu")).toBe("webgpu");
    expect(resolveLocalTextBootBackend("auto")).toBe("auto");
  });

  it("quickStartupModelChoice picks SmolLM on mobile without awaiting WebGPU", () => {
    vi.stubGlobal("navigator", { userAgent: "Mozilla/5.0 (Linux; Android 14) Mobile" });
    vi.stubGlobal("window", { innerWidth: 412, matchMedia: () => ({ matches: true }) });
    expect(quickStartupModelChoice("auto")).toBe("local-text");
    expect(quickStartupModelChoice("gemma")).toBe("gemma");
  });
});
