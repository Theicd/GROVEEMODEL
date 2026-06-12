import { describe, expect, it } from "vitest";
import {
  createPresenceAccumState,
  DEFAULT_PRESENCE_CONFIG,
  tickPresence,
} from "./presenceAccumulator";
import { ConsciousnessEngine } from "./consciousnessEngine";

describe("presenceAccumulator", () => {
  it("does not flip to stable on single-frame detection", () => {
    const state = createPresenceAccumState();
    const r = tickPresence(state, true, 0.08);
    expect(r.soul).not.toBe("STABLE_PRESENCE");
    expect(r.confidence).toBeGreaterThan(0);
  });

  it("reaches stable after sustained detection", () => {
    const state = createPresenceAccumState();
    let soul = "VOID_IDLE";
    for (let i = 0; i < 40; i++) {
      const r = tickPresence(state, true, 0.08, DEFAULT_PRESENCE_CONFIG, 1000 + i * 80);
      soul = r.soul;
    }
    expect(soul).toBe("STABLE_PRESENCE");
  });

  it("tolerates brief flicker without immediate collapse", () => {
    const state = createPresenceAccumState();
    for (let i = 0; i < 40; i++) {
      tickPresence(state, true, 0.08, DEFAULT_PRESENCE_CONFIG, 1000 + i * 80);
    }
    expect(state.prevSoul).toBe("STABLE_PRESENCE");
    const afterMiss = tickPresence(state, false, 0.08, DEFAULT_PRESENCE_CONFIG, 5000);
    expect(afterMiss.soul).not.toBe("VOID_IDLE");
    expect(state.confidence).toBeGreaterThan(DEFAULT_PRESENCE_CONFIG.absentConfidenceMax);
    for (let i = 0; i < 40; i++) {
      tickPresence(state, false, 0.08, DEFAULT_PRESENCE_CONFIG, 6000 + i * 80);
    }
    expect(state.prevSoul).not.toBe("STABLE_PRESENCE");
    expect(state.confidence).toBeLessThan(DEFAULT_PRESENCE_CONFIG.stableConfidenceMin);
  });
});

describe("ConsciousnessEngine", () => {
  it("tracks soul transitions in scene memory", () => {
    const engine = new ConsciousnessEngine();
    for (let i = 0; i < 30; i++) engine.tick(true, 1000 + i * 100);
    const snap = engine.getSnapshot();
    expect(snap?.authority.personStable).toBe(true);
    expect(snap?.sceneMemory.some((e) => e.soul === "STABLE_PRESENCE")).toBe(true);
  });
});
