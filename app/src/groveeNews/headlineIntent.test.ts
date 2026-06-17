import { describe, expect, it } from "vitest";
import { isTopicsOverviewQuery } from "./headlineIntent";

describe("headlineIntent", () => {
  it("detects world topics overview", () => {
    expect(isTopicsOverviewQuery("מה חדש בעולם?")).toBe(true);
    expect(isTopicsOverviewQuery("מה קורה בעולם?")).toBe(true);
    expect(isTopicsOverviewQuery("מה הכותרות המובילות בעולם")).toBe(true);
  });

  it("does not treat specific search as topics-only", () => {
    expect(isTopicsOverviewQuery("חפש חדשות על איראן")).toBe(false);
  });
});
