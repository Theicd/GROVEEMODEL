import { describe, expect, it } from "vitest";
import { passesNeoAlertFilter } from "./alertFilters";
import type { GlobeAlertEvent } from "./types";

const base: GlobeAlertEvent = {
  id: "n1",
  type: "neo",
  lat: 0,
  lon: 0,
  location: "test",
  time: Date.now() + 86400000,
  source: "nasa-jpl",
  approachTime: Date.now() + 86400000,
};

describe("passesNeoAlertFilter", () => {
  it("rejects distant and moderate-risk NEOs", () => {
    expect(passesNeoAlertFilter({ ...base, distLd: 32 })).toBe(false);
    expect(passesNeoAlertFilter({ ...base, distLd: 12, vRel: 5 })).toBe(false);
    expect(
      passesNeoAlertFilter({
        ...base,
        distLd: 4.2,
        vRel: 12,
        approachTime: Date.now() + 50 * 3_600_000,
      }),
    ).toBe(false);
  });

  it("accepts high/critical close approaches", () => {
    expect(passesNeoAlertFilter({ ...base, distLd: 4.2, vRel: 12 })).toBe(true);
    expect(passesNeoAlertFilter({ ...base, distLd: 0.2, vRel: 8 })).toBe(true);
  });
});
