import { describe, expect, it } from "vitest";
import { formatNeoCountdown } from "./neoEta";

describe("formatNeoCountdown", () => {
  const now = 1_700_000_000_000;

  it("formats sub-day countdown as HH:MM:SS", () => {
    expect(formatNeoCountdown(now + 3_661_000, now)).toBe("01:01:01");
  });

  it("includes days when over 24h", () => {
    expect(formatNeoCountdown(now + 90_061_000, now)).toBe("1י 01:01:01");
  });

  it("reports past approach", () => {
    expect(formatNeoCountdown(now - 1000, now)).toBe("עבר את נקודת הקרבה");
  });
});
