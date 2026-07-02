import { describe, expect, it } from "vitest";
import {
  extractImageDescribeSubject,
  extractImageGenerateSubject,
  isImageDescribeRequest,
  isImageFromPreviousRequest,
  isImageGenerateRequest,
  resolveImagePromptFromHistory,
} from "./imageIntent";

describe("imageIntent", () => {
  it("detects describe requests", () => {
    expect(isImageDescribeRequest("תאר לי נוסח חייזר")).toBe(true);
    expect(isImageDescribeRequest("תאר לי תמונה של פיל עם כנפיים")).toBe(true);
    expect(isImageDescribeRequest("describe an alien on Mars")).toBe(true);
    expect(isImageDescribeRequest("שלום")).toBe(false);
  });

  it("extracts describe subject", () => {
    expect(extractImageDescribeSubject("תאר לי נוסח חייזר ירוק")).toContain("נוסח");
    expect(extractImageDescribeSubject("תאר לי תמונה של פיל עם כנפיים")).toContain("פיל");
  });

  it("extracts generate subject in one step", () => {
    expect(extractImageGenerateSubject("צור תמונה של מאדים")).toBe("מאדים");
    expect(extractImageGenerateSubject("צור תמונה של פיל עם כנפיים")).toContain("פיל");
    expect(extractImageGenerateSubject("צור תמונה")).toBeNull();
  });

  it("detects generate requests", () => {
    expect(isImageGenerateRequest("צור תמונה")).toBe(true);
    expect(isImageGenerateRequest("צור מזה תמונה")).toBe(true);
    expect(isImageFromPreviousRequest("תייצר תמונה לפי התיאור")).toBe(true);
  });

  it("resolves prompt from prior assistant turn", () => {
    const prompt = resolveImagePromptFromHistory(
      "צור מזה תמונה",
      null,
      [
        { role: "user", content: "תאר נוסח" },
        { role: "assistant", content: "A green alien with large eyes on Mars." },
      ],
    );
    expect(prompt).toContain("green alien");
  });

  it("uses pending prompt first", () => {
    expect(
      resolveImagePromptFromHistory("צור תמונה", "pending scene", []),
    ).toBe("pending scene");
  });

  it("prefers explicit generate subject over stale pending", () => {
    expect(
      resolveImagePromptFromHistory(
        "צור תמונה של מאדים",
        "elephant with wings refusal text",
        [],
      ),
    ).toBe("מאדים");
  });
});
