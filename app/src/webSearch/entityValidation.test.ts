import { describe, expect, it } from "vitest";
import {
  detectImpossiblePlace,
  isAbsurdAviationLocation,
  validateLiveDataQuery,
} from "./entityValidation";

describe("entityValidation", () => {
  it("detects moon and mars", () => {
    expect(detectImpossiblePlace("כמה מטוסים מעל הירח?")).toBe("הירח");
    expect(detectImpossiblePlace("weather on Mars now")).toBe("מאדים");
  });

  it("flags absurd aviation locations", () => {
    expect(isAbsurdAviationLocation("כמה מטוסים מעל הירח?")).toBe(true);
    expect(isAbsurdAviationLocation("מה מזג האוויר בתל אביב")).toBe(false);
  });

  it("returns canned block for aviation on moon", () => {
    const v = validateLiveDataQuery("כמה מטוסים מעל הירח?", ["aviation"]);
    expect(v.ok).toBe(false);
    if (!v.ok) {
      expect(v.cannedReply).toMatch(/אין נתונים חיים/);
      expect(v.contextText).toMatch(/NO LIVE DATA/);
    }
  });

  it("allows static questions about moon without live intents", () => {
    expect(validateLiveDataQuery("ספר לי על הירח", []).ok).toBe(true);
  });
});
