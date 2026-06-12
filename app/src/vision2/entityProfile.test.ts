import { describe, expect, it } from "vitest";
import { correctAgeForDisplay, updateEntityProfile, createEntityProfileState } from "./entityProfile";
import { EMPTY_BODY_LANGUAGE } from "./types";

describe("correctAgeForDisplay", () => {
  it("keeps youth ages unchanged", () => {
    expect(correctAgeForDisplay(7)).toBe(7);
    expect(correctAgeForDisplay(16)).toBe(16);
    expect(correctAgeForDisplay(40)).toBe(40);
  });

  it("reduces ages over 40 by 20%", () => {
    expect(correctAgeForDisplay(50)).toBe(40);
    expect(correctAgeForDisplay(60)).toBe(48);
    expect(correctAgeForDisplay(55)).toBe(44);
    expect(correctAgeForDisplay(45)).toBe(36);
  });
});

describe("updateEntityProfile", () => {
  it("stores corrected age in profile for adults", () => {
    const state = createEntityProfileState();
    const profile = updateEntityProfile(
      state,
      {
        face: { estimatedAge: 55, estimatedGender: "Male", gazeDirection: "Center" },
        body: EMPTY_BODY_LANGUAGE,
        human: {
          posture: "sitting",
          attention: "camera",
          activity: "unknown",
          energy: "medium",
          engagement: 0.5,
          updatedAt: Date.now(),
        },
        personStable: true,
      },
      Date.now(),
    );
    expect(profile.ageEstimate).toBe(44);
  });
});
