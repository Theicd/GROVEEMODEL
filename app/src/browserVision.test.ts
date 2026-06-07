import { describe, expect, it } from "vitest";
import { checkBrowserVisionSupport } from "./browserVision";

describe("browserVision", () => {
  it("reports support shape", () => {
    const s = checkBrowserVisionSupport();
    expect(typeof s.ok).toBe("boolean");
    expect(typeof s.secureContext).toBe("boolean");
    expect(typeof s.worker).toBe("boolean");
    expect(typeof s.canvas).toBe("boolean");
  });
});
