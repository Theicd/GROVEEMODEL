import { describe, expect, it } from "vitest";
import { MOON_SCENE_ANGULAR, visualRadiusFromLd } from "./neoTrack";
import { asteroidAngularFraction, spaceDisplayMeshSize } from "./spaceObjectVisuals";

describe("spaceDisplayMeshSize", () => {
  it("keeps asteroids visually smaller than the Moon at every LD shell", () => {
    for (const distLd of [0.5, 1, 5, 12, 28, 54]) {
      const large = asteroidAngularFraction(0.16, distLd);
      const small = asteroidAngularFraction(0.03, distLd);
      expect(large).toBeLessThan(0.12);
      expect(small).toBeGreaterThan(0.005);
      expect(large).toBeGreaterThan(small);
    }
  });

  it("scales world radius with distance for constant angular size", () => {
    const nearLd = 4;
    const farLd = 40;
    const near = spaceDisplayMeshSize(0.1, nearLd);
    const far = spaceDisplayMeshSize(0.1, farLd);
    expect(far).toBeGreaterThan(near);
    const nearAng = near / visualRadiusFromLd(nearLd);
    const farAng = far / visualRadiusFromLd(farLd);
    expect(Math.abs(nearAng - farAng)).toBeLessThan(nearAng * 0.05);
  });
});
