import { describe, expect, it } from "vitest";
import {
  buildPipelineReport,
  DIAG_LIMITS,
  probeDigitrafficLocations,
  runShipPipelineDiagnostics,
} from "./shipPipelineDiagnostics";

describe("shipPipelineDiagnostics", () => {
  it("Digitraffic API returns 1000+ Baltic vessels (source layer)", async () => {
    const r = await probeDigitrafficLocations(40_000);
    expect(r.ok, r.error).toBe(true);
    expect(r.count).toBeGreaterThan(1000);
    expect(r.latRange?.[0]).toBeGreaterThan(50);
  }, 45_000);

  it("documents UI caps — SERP is not total fleet count", () => {
    expect(DIAG_LIMITS.serpCardCap).toBeLessThan(DIAG_LIMITS.digitrafficStoreCap);
    expect(DIAG_LIMITS.mapRenderNear).toBeGreaterThan(DIAG_LIMITS.serpCardCap);
  });

  it("flags staticHost mis-detection on local port 5180", () => {
    const report = buildPipelineReport({
      digitraffic: { ok: true, count: 3200 },
      hostFlags: { port: "5180", staticHost: true, localDev: true, proxyUsed: false },
      liveStoreCount: 38,
    });
    expect(report.bottleneck).toBe("host-routing");
    expect(report.layers.find((l) => l.id === "host-routing")?.status).toBe("fail");
  });

  it("runShipPipelineDiagnostics completes without AIS key", async () => {
    const report = await runShipPipelineDiagnostics();
    expect(report.layers.length).toBeGreaterThanOrEqual(2);
    expect(report.summaryHe.length).toBeGreaterThan(5);
  }, 45_000);
});
