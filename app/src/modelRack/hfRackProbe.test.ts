import { describe, expect, it } from "vitest";
import { isHfRackFreeEligible } from "./hfRackProbe";

describe("hfRackProbe", () => {
  it("rack accepts only WORKING + FREE (no HF token)", () => {
    expect(isHfRackFreeEligible("WORKING", "FREE")).toBe(true);
    expect(isHfRackFreeEligible("WORKING", "TOKEN")).toBe(false);
    expect(isHfRackFreeEligible("PROVIDER REQUIRED", "TOKEN")).toBe(false);
    expect(isHfRackFreeEligible("WORKING", "UNKNOWN")).toBe(false);
  });
});
