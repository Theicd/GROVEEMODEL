import { describe, expect, it } from "vitest";
import { parseSceneAnalysisJson, WorldMemory } from "./environmentMemory";
import { frameDiffScore, isSignificantChange } from "./sceneChangeDetector";

describe("environmentMemory / worldMemory", () => {
  it("parses scene analysis JSON", () => {
    const raw = `{"current":["Person","Laptop"],"events":["Phone picked up"],"interesting":true,"summary":"User at desk"}`;
    const p = parseSceneAnalysisJson(raw);
    expect(p?.current).toContain("Person");
    expect(p?.events).toContain("Phone picked up");
    expect(p?.interesting).toBe(true);
  });

  it("tracks baseline then post-baseline events", () => {
    const mem = new WorldMemory();
    const baseline = mem.applyVision({
      objects: ["person", "black shirt"],
      events: ["Person detected"],
      interesting: true,
      summary: "User at desk",
    });
    expect(baseline.isBaselineCapture).toBe(true);
    expect(baseline.newEvents).toHaveLength(0);
    expect(mem.hasData()).toBe(true);

    const update = mem.applyVision({
      objects: ["person", "black shirt", "phone"],
      events: ["Phone picked up"],
      interesting: true,
      summary: "User picked up phone",
    });
    expect(update.newEvents.some((e) => e.type === "object_appeared")).toBe(true);
    const block = mem.toPromptBlock();
    expect(block).toMatch(/World memory/);
    expect(block).toMatch(/phone/);
  });
});

describe("sceneChangeDetector", () => {
  it("detects significant pixel change", () => {
    const make = (fill: number) => ({ data: new Uint8ClampedArray(16).fill(fill), width: 4, height: 1 } as ImageData);
    const a = make(10);
    const b = make(10);
    expect(isSignificantChange(frameDiffScore(a, b))).toBe(false);
    b.data[0] = 255;
    b.data[1] = 255;
    b.data[2] = 255;
    expect(frameDiffScore(a, b)).toBeGreaterThan(0);
  });
});
