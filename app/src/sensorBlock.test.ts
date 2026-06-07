import { describe, expect, it } from "vitest";
import {
  buildProactiveSensorBlock,
  buildSensorBlock,
  describeSituationSubject,
} from "./sensorBlock";
import { WorldMemory } from "./worldMemory";

describe("sensorBlock", () => {
  it("includes latest situation in base block", () => {
    const world = new WorldMemory();
    world.personPresent = true;
    world.poseState = "standing";
    world.holding = ["cup"];
    world.focusHint = "standing with a drink in hand";
    world.lastSituationSubject = "stood_with_drink";
    world.lastSituationAt = Date.now();

    const block = buildSensorBlock(world, {
      poseState: "standing",
      confidence: 0.8,
      gestures: [],
      holding: ["cup"],
      focusHint: "standing with a drink in hand",
    });

    expect(block).toMatch(/holding: cup/);
    expect(block).toMatch(/Latest action.*stood up while holding cup/i);
  });

  it("adds trigger context for proactive Gemma", () => {
    const world = new WorldMemory();
    world.personPresent = true;
    world.holding = ["cup"];

    const block = buildProactiveSensorBlock(
      world,
      { poseState: "standing", confidence: 0.8, gestures: [], holding: ["cup"], focusHint: "" },
      {
        reason: "situation:object_held:cup",
        topic: "object_held_cup",
        fallbackHint: "שמתי לב לכוס ביד — הפסקת קפה?",
      },
    );

    expect(block).toMatch(/Why speaking now.*picked up cup/i);
    expect(block).toMatch(/Template intent.*כוס/);
  });

  it("describes stood_with_drink from holding", () => {
    const world = new WorldMemory();
    world.holding = ["bottle"];
    expect(describeSituationSubject("stood_with_drink", world)).toMatch(/bottle/);
  });
});
