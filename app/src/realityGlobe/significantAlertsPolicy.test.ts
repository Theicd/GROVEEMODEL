import { describe, expect, it } from "vitest";

/** Mirrors reality-core/ui/significant-alerts.js policy. */
function parseMag(a: { magnitude?: number; mag?: number; summary?: string }) {
  const m = Number(a.magnitude ?? a.mag);
  if (Number.isFinite(m) && m > 0) return m;
  const hit = String(a.summary || "").match(/\bM\s*([\d.]+)/i);
  return hit ? parseFloat(hit[1]) : 0;
}

function isSignificantFly(a: { category?: string; severity?: number; magnitude?: number; summary?: string }) {
  const cat = String(a.category || "").toUpperCase();
  const sev = Number(a.severity || 0);
  if (cat === "ISRAEL" || cat === "CORRELATION") return sev >= 3;
  if (cat === "SEISMIC" || cat === "TSUNAMI") return parseMag(a) >= 4.5;
  if (cat === "DISASTER" || cat === "FIRE") return sev >= 4;
  if (cat === "WEATHER") return sev >= 4;
  return false;
}

describe("significantAlertsPolicy", () => {
  it("flies for M4.6 earthquake", () => {
    expect(isSignificantFly({ category: "SEISMIC", severity: 4, magnitude: 4.6 })).toBe(true);
  });

  it("does not fly for M4.2 earthquake", () => {
    expect(isSignificantFly({ category: "SEISMIC", severity: 3, magnitude: 4.2 })).toBe(false);
  });

  it("does not fly for aviation panel alerts", () => {
    expect(isSignificantFly({ category: "AVIATION", severity: 4 })).toBe(false);
  });

  it("flies for GDACS disaster severity 4", () => {
    expect(isSignificantFly({ category: "DISASTER", severity: 4 })).toBe(true);
  });

  it("flies for Israel alert severity 3+", () => {
    expect(isSignificantFly({ category: "ISRAEL", severity: 3 })).toBe(true);
  });
});
