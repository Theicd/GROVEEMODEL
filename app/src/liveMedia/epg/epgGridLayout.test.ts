import { describe, expect, it } from "vitest";
import { buildEpgGridWindow, layoutProgramsInWindow } from "./epgGridLayout";
import type { EpgProgram } from "./types";

describe("epgGridLayout", () => {
  it("places programme blocks on a timeline", () => {
    const now = new Date("2026-06-25T22:15:00Z");
    const window = buildEpgGridWindow(now);
    const programs: EpgProgram[] = [
      {
        channelId: "c1",
        title: "Show A",
        start: new Date("2026-06-25T22:00:00Z"),
        end: new Date("2026-06-25T23:00:00Z"),
      },
    ];
    const blocks = layoutProgramsInWindow(programs, window, now);
    expect(blocks).toHaveLength(1);
    expect(blocks[0].widthPx).toBeGreaterThan(40);
    expect(blocks[0].live).toBe(true);
  });
});
