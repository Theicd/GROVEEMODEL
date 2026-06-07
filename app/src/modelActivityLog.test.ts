import { describe, expect, it } from "vitest";
import { appendModelActivity, directionLabel, formatActivityLogForCopy } from "./modelActivityLog";

describe("modelActivityLog", () => {
  it("prepends entries and caps length", () => {
    let log = appendModelActivity([], {
      direction: "out",
      kind: "test",
      title: "A",
      detail: "one",
    });
    log = appendModelActivity(log, {
      direction: "in",
      kind: "test",
      title: "B",
      detail: "two",
    });
    expect(log[0].title).toBe("B");
    expect(log[1].title).toBe("A");
  });

  it("direction labels", () => {
    expect(directionLabel("out")).toContain("מודל");
  });

  it("formatActivityLogForCopy orders oldest first", () => {
    const text = formatActivityLogForCopy([
      {
        id: "2",
        ts: 2000,
        direction: "in",
        kind: "generate",
        title: "B",
        detail: "answer",
      },
      {
        id: "1",
        ts: 1000,
        direction: "out",
        kind: "generate",
        title: "A",
        detail: "prompt",
        meta: { tokens: 128 },
      },
    ]);
    expect(text).toMatch(/GROVEE — Model Activity Log/);
    expect(text.indexOf("#1")).toBeLessThan(text.indexOf("#2"));
    expect(text).toContain("Meta:");
    expect(text).toContain("tokens: 128");
  });
});
