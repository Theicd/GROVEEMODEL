import { describe, expect, it } from "vitest";
import { buildNewsPanelGuideReply } from "./newsPanelGuideReply";

describe("buildNewsPanelGuideReply", () => {
  it("explains panel usage when cards exist", () => {
    const text = buildNewsPanelGuideReply("חפש חדשות בנושא סייבר", {
      mode: "search",
      cardCount: 8,
    });
    expect(text).toContain("8 כרטיסיות");
    expect(text).toContain("מימין");
    expect(text).toContain("סכם כתבה");
    expect(text).toContain("מקור");
    expect(text).not.toMatch(/ANSWER \(headline\)/i);
    expect(text).not.toMatch(/Qwen|Gemma/i);
  });

  it("handles empty results", () => {
    const text = buildNewsPanelGuideReply("חפש חדשות בנושא X", { mode: "search", cardCount: 0 });
    expect(text).toContain("לא נמצאו");
  });
});
