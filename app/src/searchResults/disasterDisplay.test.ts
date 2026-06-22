import { describe, expect, it } from "vitest";
import { resolveAlertLevel, resolveDisasterType } from "./disasterDisplay";

describe("disasterDisplay", () => {
  it("maps GDACS type codes to Hebrew labels", () => {
    expect(resolveDisasterType("FL").labelHe).toBe("הצפה");
    expect(resolveDisasterType("TC").labelHe).toBe("הוריקן / סופה");
    expect(resolveDisasterType("EQ").icon).toBe("🫨");
  });

  it("infers type from event name when code missing", () => {
    expect(resolveDisasterType(undefined, "Flood in Turkey").code).toBe("FL");
    expect(resolveDisasterType(undefined, "Tropical Cyclone ALBERTO").code).toBe("TC");
  });

  it("maps alert levels to severity labels", () => {
    expect(resolveAlertLevel("Green").labelHe).toBe("קל");
    expect(resolveAlertLevel("Orange").severity).toBe("orange");
    expect(resolveAlertLevel("Red").labelHe).toBe("חמור");
  });
});
