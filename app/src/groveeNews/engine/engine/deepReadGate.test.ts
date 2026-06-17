import { describe, expect, it, vi } from "vitest";

vi.mock("../settings/aiMode", () => ({
  isAiDeepReadEnabled: () => false,
}));

describe("deepReadGate", () => {
  it("reflects aiMode setting", async () => {
    const { isDeepReadEnabled } = await import("./deepReadGate");
    expect(isDeepReadEnabled()).toBe(false);
  });
});
