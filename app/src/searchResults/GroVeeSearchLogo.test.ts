import { describe, expect, it } from "vitest";
import { pickGroVeeDoodleId } from "./GroVeeSearchLogo";

describe("GroVeeSearchLogo", () => {
  it("rotates doodle id by day", () => {
    const a = pickGroVeeDoodleId(new Date("2026-06-15T12:00:00Z"));
    const b = pickGroVeeDoodleId(new Date("2026-06-16T12:00:00Z"));
    expect(a).not.toBe(b);
  });
});
