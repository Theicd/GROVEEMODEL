import { describe, expect, it } from "vitest";
import { HalMoodEngine } from "./halMoodEngine";

describe("halMoodEngine", () => {
  it("updates mood from body language signals", () => {
    const engine = new HalMoodEngine();
    const hal = engine.update({
      human: {
        posture: "sitting",
        attention: "screen",
        activity: "working",
        energy: "medium",
        engagement: 0.7,
        updatedAt: Date.now(),
      },
      body: {
        focused: 0.7,
        thinking: 0.2,
        stressed: 0.1,
        bored: 0.05,
        ageSec: 5,
        updatedAt: Date.now(),
      },
      situation: {
        primary: "working",
        confidence: 0.8,
        description: "Focused work",
        updatedAt: Date.now(),
      },
      personPresent: true,
    });
    expect(hal.personPresent).toBe(true);
    expect(hal.mood).toBeTruthy();
    expect(hal.tone).toBeTruthy();
  });
});
