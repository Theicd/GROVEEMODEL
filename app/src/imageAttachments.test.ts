import { describe, expect, it } from "vitest";
import { defaultVisionPrompt, isAcceptedImageFile } from "./imageAttachments";

describe("imageAttachments", () => {
  it("accepts common image types", () => {
    expect(isAcceptedImageFile({ type: "image/png", name: "x.png" } as File)).toBe(true);
    expect(isAcceptedImageFile({ type: "application/pdf", name: "x.pdf" } as File)).toBe(false);
  });

  it("defaultVisionPrompt is localized", () => {
    expect(defaultVisionPrompt(true)).toMatch(/תמונה/);
    expect(defaultVisionPrompt(false)).toMatch(/image/i);
  });
});
