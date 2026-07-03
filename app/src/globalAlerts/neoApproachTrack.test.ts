import { describe, expect, it } from "vitest";
import { approachClosurePercent, buildApproachDisplayTrack } from "./neoApproachTrack";
import type { NeoOrbitTrack } from "./neoTrack";

const track: NeoOrbitTrack = {
  designation: "test",
  closest: { t: 1000, lat: 10, lon: 20, distAu: 0.02, distLd: 0.5, deldotKmS: -5 },
  points: [
    { t: 0, lat: -30, lon: -40, distAu: 0.2, distLd: 20, deldotKmS: -5 },
    { t: 500, lat: 0, lon: 0, distAu: 0.08, distLd: 8, deldotKmS: -5 },
    { t: 1000, lat: 10, lon: 20, distAu: 0.02, distLd: 0.5, deldotKmS: -5 },
  ],
};

describe("buildApproachDisplayTrack", () => {
  it("builds monotonic inbound segment", () => {
    const display = buildApproachDisplayTrack(track, 250);
    expect(display.points.length).toBeGreaterThan(2);
    const first = display.points[0].distLd;
    const last = display.points[display.points.length - 1].distLd;
    expect(first).toBeGreaterThan(last);
  });
});

describe("approachClosurePercent", () => {
  it("increases as object closes in", () => {
    const far = approachClosurePercent(20, 0.5, 20);
    const near = approachClosurePercent(5, 0.5, 20);
    expect(near).toBeGreaterThan(far);
  });
});
