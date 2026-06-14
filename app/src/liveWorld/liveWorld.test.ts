import { describe, expect, it } from "vitest";
import {
  applySnapshotFallbacks,
  fallbackFromLiveWorldSnapshot,
  ingestGlobeLivePayload,
  setLiveWorldSnapshot,
  clearLiveWorldSnapshotCache,
} from "./index";

describe("liveWorld snapshot", () => {
  it("ingests globe payload and answers earthquake from cache", () => {
    clearLiveWorldSnapshotCache();
    ingestGlobeLivePayload({
      earthquake: {
        items: [
          { magnitude: 6.2, place: "off coast of Japan", time: Date.now() - 3600_000 },
          { magnitude: 4.1, place: "Dead Sea region", time: Date.now() - 7200_000 },
        ],
      },
    });

    const fb = fallbackFromLiveWorldSnapshot("איפה הרעידה החזקה?", ["earthquake"]);
    expect(fb?.ok).toBe(true);
    expect(fb?.text).toContain("M6.2");
    expect(fb?.provider).toBe("usgs-earthquake");
  });

  it("filters earthquakes for Israel region", () => {
    clearLiveWorldSnapshotCache();
    setLiveWorldSnapshot({
      fetchedAt: Date.now(),
      source: "globe",
      earthquake: {
        feedLabel: "USGS",
        items: [
          { magnitude: 5.5, place: "Near Israel", time: Date.now() },
          { magnitude: 7.0, place: "Chile", time: Date.now() },
        ],
      },
    });

    const fb = fallbackFromLiveWorldSnapshot("רעידות בישראל?", ["earthquake"]);
    expect(fb?.ok).toBe(true);
    expect(fb?.text).toContain("Israel");
    expect(fb?.text).not.toContain("Chile");
  });

  it("answers ISS from cache", () => {
    clearLiveWorldSnapshotCache();
    setLiveWorldSnapshot({
      fetchedAt: Date.now(),
      source: "globe",
      iss: { lat: 12.3, lon: 45.6, altitudeKm: 420 },
    });

    const fb = fallbackFromLiveWorldSnapshot("איפה ISS?", ["satellite"]);
    expect(fb?.ok).toBe(true);
    expect(fb?.text).toContain("12.30");
    expect(fb?.provider).toBe("iss-tracker");
  });

  it("applySnapshotFallbacks fills failed provider slot", () => {
    clearLiveWorldSnapshotCache();
    setLiveWorldSnapshot({
      fetchedAt: Date.now(),
      source: "fetch",
      iss: { lat: 1, lon: 2, altitudeKm: 400 },
    });

    const out = applySnapshotFallbacks(
      "איפה תחנת החלל?",
      ["satellite"],
      [{ provider: "iss-tracker", label: "x", ok: false, text: "", error: "fail", latencyMs: 1 }],
    );
    expect(out.some((s) => s.provider === "iss-tracker" && s.ok)).toBe(true);
  });
});
