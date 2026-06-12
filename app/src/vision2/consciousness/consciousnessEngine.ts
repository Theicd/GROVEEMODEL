/**
 * HAL Consciousness Layer — temporal single source of truth.
 * Replaces per-frame person truth with accumulated stability.
 */

import { updateAffect, updatePerception } from "./affectEngine";
import { buildPresenceAuthority, formatConsciousnessForGemma } from "./halBrain";
import {
  createPresenceAccumState,
  tickPresence,
} from "./presenceAccumulator";
import { SceneMemory } from "./sceneMemory";
import {
  createInitialGlobalState,
  type ConsciousnessSnapshot,
  type HalGlobalState,
  type PresenceAuthority,
} from "./types";

export class ConsciousnessEngine {
  private global = createInitialGlobalState();
  private presence = createPresenceAccumState();
  private memory = new SceneMemory();
  private lastTickAt = 0;
  private flickerWindow: boolean[] = [];
  private lastAuthority: PresenceAuthority | null = null;

  reset(): void {
    this.global = createInitialGlobalState();
    this.presence = createPresenceAccumState();
    this.memory.reset();
    this.lastTickAt = 0;
    this.flickerWindow = [];
    this.lastAuthority = null;
  }

  getState(): HalGlobalState {
    return this.global;
  }

  getAuthority(): PresenceAuthority | null {
    return this.lastAuthority;
  }

  getSnapshot(): ConsciousnessSnapshot | null {
    if (!this.lastAuthority) return null;
    return {
      ...this.global,
      sceneMemory: this.memory.recent(),
      gemmaBlock: formatConsciousnessForGemma(
        this.global,
        this.memory.recent(),
        this.lastAuthority,
      ),
      authority: this.lastAuthority,
    };
  }

  /** Call each vision tick with raw YOLO person detection. */
  tick(rawPersonDetected: boolean, now = Date.now()): PresenceAuthority {
    const dtSec =
      this.lastTickAt > 0 ? Math.min(0.5, Math.max(0.04, (now - this.lastTickAt) / 1000)) : 0.08;
    this.lastTickAt = now;

    this.flickerWindow.push(rawPersonDetected);
    if (this.flickerWindow.length > 12) this.flickerWindow.shift();
    const flickerRate =
      this.flickerWindow.length >= 4
        ? this.flickerWindow.filter((v, i) => i > 0 && v !== this.flickerWindow[i - 1]).length /
          this.flickerWindow.length
        : 0;

    const prevSoul = this.global.presence.soul;
    const pt = tickPresence(this.presence, rawPersonDetected, dtSec, undefined, now);

    if (pt.transition) {
      this.memory.push({
        t: now,
        soul: pt.soul,
        transition: pt.transition,
        confidence: pt.confidence,
        rawDetected: rawPersonDetected,
      });
    }

    const perception = updatePerception(pt.confidence, pt.soul, flickerRate);
    const affect = updateAffect(this.global.affect, perception, pt.soul, pt.stabilitySec, dtSec);

    const continuity =
      pt.soul === "STABLE_PRESENCE"
        ? Math.min(1, pt.stabilitySec / 10)
        : Math.max(0, this.global.worldModel.continuity - 0.05);

    this.global = {
      presence: {
        phase: pt.phase,
        soul: pt.soul,
        confidence: pt.confidence,
        stabilitySec: pt.stabilitySec,
        stableSince: pt.stableSince,
        lastSeenAt: this.presence.lastSeenAt,
        lastTransition: pt.transition,
        lastTransitionAt: pt.transition ? now : this.global.presence.lastTransitionAt,
      },
      perception,
      worldModel: {
        continuity,
        entityCount: pt.soul === "STABLE_PRESENCE" ? 1 : 0,
      },
      affect,
      updatedAt: now,
    };

    const authority = buildPresenceAuthority(this.global, rawPersonDetected, prevSoul);
    this.lastAuthority = authority;
    return authority;
  }
}

export { SOUL_LABEL_HE } from "./types";
