import { describe, expect, it } from "vitest";
import { networkStatusLabel } from "../hooks/useNetworkStatus";

describe("useNetworkStatus labels", () => {
  it("returns Hebrew labels", () => {
    expect(networkStatusLabel("online", "he")).toContain("מחובר");
    expect(networkStatusLabel("offline", "he")).toContain("ללא");
    expect(networkStatusLabel("limited", "he")).toContain("מוגבל");
  });
});
