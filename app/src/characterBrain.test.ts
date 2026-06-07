import { describe, expect, it, vi } from "vitest";
import { CharacterBrain, CHARACTER_CONFIG } from "./characterBrain";
import { WorldMemory, makeSemanticEvent } from "./worldMemory";

describe("worldMemory light detection", () => {
  it("person_entered not user_returned on first person sighting", () => {
    const w = new WorldMemory();
    w.applyLightDetection({ objects: ["clock"], people: [] });
    const r = w.applyLightDetection({ objects: ["clock"], people: ["person"] });
    expect(r.newEvents.some((e) => e.type === "person_entered")).toBe(true);
    expect(r.newEvents.some((e) => e.type === "user_returned")).toBe(false);
  });

  it("suppresses events on camera churn", () => {
    const w = new WorldMemory();
    w.applyLightDetection({ objects: ["guitar", "clock"], people: [] });
    const r = w.applyLightDetection({
      objects: ["television", "chair", "laptop"],
      people: [],
    });
    expect(r.suppressedAsChurn).toBe(true);
    expect(r.newEvents).toHaveLength(0);
  });
});

describe("worldMemory", () => {
  it("first scan establishes baseline without events", () => {
    const w = new WorldMemory();
    const baseline = w.applyVision({
      objects: ["guitar", "window", "clock"],
      events: ["Guitar visible", "Clock in frame"],
      interesting: true,
      summary: "room",
    });
    expect(baseline.isBaselineCapture).toBe(true);
    expect(baseline.newEvents).toHaveLength(0);
    expect(w.baselineObjects).toEqual(["guitar", "window", "clock"]);
  });

  it("diff detects object appeared after baseline", () => {
    const w = new WorldMemory();
    w.applyVision({ objects: ["laptop"], events: [], interesting: false, summary: "desk" });
    const { newEvents, appeared } = w.applyVision({
      objects: ["laptop", "guitar"],
      events: [],
      interesting: true,
      summary: "desk with guitar",
    });
    expect(appeared).toContain("guitar");
    expect(newEvents.some((e) => e.type === "object_appeared")).toBe(true);
  });

  it("toPromptBlock includes world memory header", () => {
    const w = new WorldMemory();
    w.applyVision({ objects: ["clock"], events: [], summary: "room", interesting: false });
    expect(w.toPromptBlock()).toMatch(/World memory/);
    expect(w.toPromptBlock()).toMatch(/clock/);
  });
});

describe("characterBrain", () => {
  it("excited on person entered", () => {
    vi.spyOn(Date, "now").mockReturnValue(1_000_000);
    const brain = new CharacterBrain();
    brain.lastProactiveAt = 0;
    const world = new WorldMemory();
    const ev = makeSemanticEvent("person_entered", "Person entered frame", "person");
    const decision = brain.evaluate(world, [ev]);
    expect(decision?.mood).toBe("curious");
    expect(decision?.message).toMatch(/בפריים/);
    vi.restoreAllMocks();
  });

  it("does not repeat guitar topic within TTL", () => {
    const brain = new CharacterBrain();
    brain.topicsMentioned.set("guitar", Date.now());
    brain.lastUserInteractionAt = Date.now() - CHARACTER_CONFIG.curiousAfterMs - 1000;
    brain.lastProactiveAt = 0;
    brain.topicsMentioned.set("scene_general", Date.now());
    const world = new WorldMemory();
    world.objects = ["guitar"];
    const decision = brain.evaluate(world, []);
    expect(decision).toBeNull();
  });

  it("curious after idle with guitar", () => {
    const brain = new CharacterBrain();
    brain.lastUserInteractionAt = Date.now() - CHARACTER_CONFIG.curiousAfterMs - 5000;
    brain.lastProactiveAt = 0;
    const world = new WorldMemory();
    world.objects = ["guitar"];
    const decision = brain.evaluate(world, []);
    expect(decision?.mood).toBe("curious");
    expect(decision?.message).toMatch(/שמתי לב שיש גיטרה/);
  });

  it("baseline curious after user silent ~30s", () => {
    const now = 2_000_000;
    vi.spyOn(Date, "now").mockReturnValue(now);
    const brain = new CharacterBrain();
    brain.baselineSceneAt = now - CHARACTER_CONFIG.baselineCuriousAfterMs - 1000;
    brain.lastUserInteractionAt = now - CHARACTER_CONFIG.baselineCuriousAfterMs - 1000;
    const world = new WorldMemory();
    world.baselineEstablished = true;
    world.objects = ["guitar", "clock"];
    const decision = brain.evaluate(world, []);
    expect(decision?.mood).toBe("curious");
    expect(decision?.reason).toBe("curious:baseline");
    vi.restoreAllMocks();
  });

  it("excited on wave motion", () => {
    const brain = new CharacterBrain();
    brain.lastProactiveAt = 0;
    const world = new WorldMemory();
    world.people = ["person"];
    world.personPresent = true;
    const ev = makeSemanticEvent("activity_change", "Person waving", "wave", true);
    const decision = brain.evaluate(world, [ev]);
    expect(decision?.mood).toBe("excited");
    expect(decision?.reason).toBe("situation:wave");
    expect(decision?.message).toMatch(/תשומת הלב/);
  });

  it("curious on object_held cup", () => {
    const brain = new CharacterBrain();
    brain.lastProactiveAt = 0;
    const world = new WorldMemory();
    world.holding = ["cup"];
    world.personPresent = true;
    const ev = makeSemanticEvent("activity_change", "Person holding cup", "object_held:cup", true);
    const decision = brain.evaluate(world, [ev]);
    expect(decision?.mood).toBe("curious");
    expect(decision?.message).toMatch(/כוס/);
  });

  it("curious on stood_with_drink", () => {
    const brain = new CharacterBrain();
    brain.lastProactiveAt = 0;
    const world = new WorldMemory();
    world.holding = ["cup"];
    const ev = makeSemanticEvent(
      "activity_change",
      "Stood up with cup",
      "stood_with_drink",
      true,
    );
    const decision = brain.evaluate(world, [ev]);
    expect(decision?.mood).toBe("curious");
    expect(decision?.message).toMatch(/קמת|קפה/);
  });

  it("user_returned tiered welcome after long absence", () => {
    vi.spyOn(Date, "now").mockReturnValue(5_000_000);
    const brain = new CharacterBrain();
    brain.lastProactiveAt = 0;
    const world = new WorldMemory();
    world.lastAbsentDurationMs = 400_000;
    const ev = makeSemanticEvent("user_returned", "User returned", "person");
    const decision = brain.evaluate(world, [ev]);
    expect(decision?.message).toMatch(/ברוך שובך/);
    vi.restoreAllMocks();
  });

  it("proactive alone message after long absence", () => {
    const now = 10_000_000;
    vi.spyOn(Date, "now").mockReturnValue(now);
    const brain = new CharacterBrain();
    brain.lastProactiveAt = now - 120_000;
    const world = new WorldMemory();
    world.personPresent = false;
    world.absentSince = now - CHARACTER_CONFIG.absentSpeakAfterMs - 1000;
    const decision = brain.evaluate(world, []);
    expect(decision?.reason).toBe("presence:alone");
    vi.restoreAllMocks();
  });
});
