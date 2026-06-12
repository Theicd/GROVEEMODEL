/**
 * Vision 2.0 orchestrator — L2–L8 + Phase 7 capabilities per tick.
 * VisionResult stays private to sensor UI; this engine produces WorldSnapshot + DialogueContext.
 */

import type { CharacterBrain } from "../characterBrain";
import type { VisionResult } from "../vision-lab/core/types";
import type { WorldMemory } from "../worldMemory";
import {
  evaluateEmotionalCoach,
  evaluateEmotionalState,
} from "./capabilities/emotionalCompanion";
import {
  createProductivityTracker,
  evaluateProductivity,
  updateProductivityTracker,
} from "./capabilities/productivityCoach";
import { evaluateSocialAwareness } from "./capabilities/socialAwarenessModel";
import {
  createTeachingTracker,
  evaluateTeachingState,
  resetTeachingTracker,
} from "./capabilities/teachingAssistant";
import {
  createBodyLanguageModelState,
  resetBodyLanguageModel,
  updateBodyLanguage,
} from "./bodyLanguageModel";
import {
  buildDialogueContext,
  buildRecentChanges,
  eventsToSpeakReason,
} from "./dialogueContext";
import { EpisodicMemory } from "./episodicMemory";
import {
  createHumanStateEngineState,
  resetHumanStateEngine,
  updateHumanState,
} from "./humanStateEngine";
import { perceiveFromVisionResult } from "./perceptionEngine";
import { AudioSensor, audioSummaryForContext } from "./sensors/audioSensor";
import { createFrameBundle } from "./sensors/frameBundle";
import {
  createSituationEngineState,
  resetSituationEngine,
  updateSituation,
} from "./situationEngine";
import {
  createEntityProfileState,
  updateEntityProfile,
  type EntityProfile,
} from "./entityProfile";
import { BoolTracker } from "./temporalTracker";
import type { DialogueContext, HumanState, WorldSnapshot } from "./types";
import type { HalMoodState } from "./halMoodEngine";
import {
  buildWorldSnapshot,
  createSessionTracker,
  updateSessionTracker,
} from "./worldModel";

export class Vision2Engine {
  private humanEngine = createHumanStateEngineState();
  private bodyState = createBodyLanguageModelState();
  private situationState = createSituationEngineState();
  private session = createSessionTracker();
  private productivity = createProductivityTracker();
  private teaching = createTeachingTracker();
  private audio = new AudioSensor();
  private episodic = new EpisodicMemory();
  private faceTouch = new BoolTracker();
  private handsOnHead = new BoolTracker();
  private waving = new BoolTracker();
  private entityState = createEntityProfileState();
  private lastEntity: EntityProfile | null = null;
  private prevSituationPrimary: string | null = null;
  private lastHuman: HumanState | null = null;
  private lastSnapshot: WorldSnapshot | null = null;
  private lastDialogue: DialogueContext | null = null;
  private lastMotionLevel = 0;
  private lastFaceTouchSec = 0;
  private lastWaveRising = false;

  reset(): void {
    resetHumanStateEngine(this.humanEngine);
    resetBodyLanguageModel(this.bodyState);
    resetSituationEngine(this.situationState);
    this.session = createSessionTracker();
    this.productivity = createProductivityTracker();
    resetTeachingTracker(this.teaching);
    this.audio.reset();
    this.episodic.reset();
    this.faceTouch.reset();
    this.handsOnHead.reset();
    this.waving.reset();
    this.entityState = createEntityProfileState();
    this.lastEntity = null;
    this.prevSituationPrimary = null;
    this.lastHuman = null;
    this.lastSnapshot = null;
    this.lastDialogue = null;
    this.lastMotionLevel = 0;
    this.lastFaceTouchSec = 0;
    this.lastWaveRising = false;
  }

  getSnapshot(): WorldSnapshot | null {
    return this.lastSnapshot;
  }

  getDialogueContext(): DialogueContext | null {
    return this.lastDialogue;
  }

  getHumanState(): HumanState | null {
    return this.lastHuman;
  }

  /** Patch HAL mood layer after situation-pack peek (UI + chat). */
  applyHalLayer(hal: HalMoodState): void {
    if (!this.lastDialogue) return;
    this.lastDialogue.hal = {
      mood: hal.mood,
      tone: hal.tone,
      moodLabelHe: hal.moodLabelHe,
      personPresent: hal.personPresent,
      sceneLabel: hal.sceneLabel,
      activePackId: hal.activePackId,
      interpretation: hal.interpretation,
      stressLevel: hal.stressLevel,
      engagement: hal.engagement,
    };
    this.lastDialogue.character.mood = hal.mood;
  }

  applyInterpretationLayer(layer: import("./types").InterpretationLayer): void {
    if (!this.lastDialogue) return;
    this.lastDialogue.interpretation = layer;
  }

  applyConsciousnessLayer(snapshot: import("./consciousness/types").ConsciousnessSnapshot | null): void {
    if (!this.lastDialogue || !snapshot) return;
    const auth = snapshot.authority;
    this.lastDialogue.consciousness = {
      soul: auth.soul,
      phase: auth.phase,
      confidence: auth.confidence,
      stabilitySec: auth.stabilitySec,
      personStable: auth.personStable,
      rawDetected: auth.rawDetected,
      interpretation: auth.interpretation,
      evolution: snapshot.sceneMemory.map((e) => e.soul.replace(/_/g, " ")).join(" → ") || "VOID_IDLE",
      affect: snapshot.affect,
      perception: snapshot.perception,
      gemmaBlock: snapshot.gemmaBlock,
    };
    this.lastDialogue.worldState.person.present = auth.personStable;
    this.lastDialogue.hal.personPresent = auth.personStable;
  }

  getProcessMeta(): { faceTouchSec: number; waveRising: boolean } {
    return { faceTouchSec: this.lastFaceTouchSec, waveRising: this.lastWaveRising };
  }

  /** Called each UI tick after WorldMemory sync. */
  process(
    result: VisionResult,
    world: WorldMemory,
    character: CharacterBrain,
    options?: { freshEvents?: import("../worldMemory").SemanticEvent[] },
  ): { snapshot: WorldSnapshot; dialogue: DialogueContext } {
    const now = Date.now();
    const personPresent = world.personPresent;
    const motionLevel = Math.max(this.lastMotionLevel, world.lastMotionLevel);
    this.lastMotionLevel = world.lastMotionLevel;

    const audioSample = this.audio.sample(now);
    void createFrameBundle(result, audioSample, now);

    const obs = perceiveFromVisionResult(result, personPresent, motionLevel, now);
    const faceTrack = this.faceTouch.update(obs.touchingFace || obs.handOnChin, now);
    const headTrack = this.handsOnHead.update(obs.handsOnHead || obs.touchingHead, now);
    const waveTrack = this.waving.update(obs.waving, now);

    const human = updateHumanState(obs, world, this.humanEngine, now);
    const body = updateBodyLanguage(
      obs,
      human,
      { faceTouchSec: faceTrack.durationSec, handsOnHeadSec: headTrack.durationSec },
      this.bodyState,
      now,
    );
    const situation = updateSituation(obs, human, this.situationState, now);
    updateSessionTracker(this.session, situation, obs, now);
    updateProductivityTracker(this.productivity, situation, obs.personPresent, now);

    this.episodic.tickEpisode(
      "face_touch",
      faceTrack.value && faceTrack.durationSec >= 3,
      body,
      now,
    );
    this.episodic.tickEpisode(
      "stress_episode",
      body.stressed >= 0.65 && headTrack.durationSec >= 5,
      body,
      now,
    );
    this.episodic.tickEpisode(
      "focus_block",
      situation.primary === "working" && body.focused >= 0.65,
      body,
      now,
    );
    if (waveTrack.rising && obs.personPresent) {
      this.episodic.recordInstant("greeting", now);
    }
    if (situation.primary === "drinking" && obs.holdingCup) {
      this.episodic.recordInstant("break", now);
    }

    const snapshot = buildWorldSnapshot(
      world,
      human,
      body,
      situation,
      this.session,
      faceTrack.durationSec,
      now,
    );

    const social = evaluateSocialAwareness(obs, human, body, audioSample, now);
    const productivityState = evaluateProductivity(
      snapshot.session,
      this.productivity,
      body,
      human,
      situation,
      now,
    );
    const teachingState = evaluateTeachingState(obs, human, body, this.teaching, now);
    const emotional = evaluateEmotionalState(human, body, social, productivityState, now);
    const coach = evaluateEmotionalCoach(
      human,
      body,
      situation,
      snapshot.session,
      social,
      emotional,
      productivityState,
    );

    const recentChanges = buildRecentChanges(world, snapshot, this.prevSituationPrimary, now);
    if (teachingState.likelyDistracted) {
      recentChanges.unshift({
        kind: "shifted_focus",
        ageSec: Math.round(teachingState.ageSec),
        confidence: teachingState.attentionLoss,
        text: "Attention may have drifted from task.",
      });
    }

    const speakReason = options?.freshEvents?.length
      ? eventsToSpeakReason(options.freshEvents)
      : undefined;

    const face = result.faces[0] ?? null;
    this.lastEntity = updateEntityProfile(
      this.entityState,
      {
        face,
        emotion: result.emotion ?? null,
        body,
        human,
        personStable: personPresent,
      },
      now,
    );

    const dialogue = buildDialogueContext({
      world,
      snapshot,
      human,
      coach,
      capabilities: {
        social,
        productivity: productivityState,
        teaching: teachingState,
        emotional,
      },
      audio: audioSummaryForContext(audioSample),
      episodicSummary: this.episodic.summarize(),
      character,
      shouldSpeak: !!speakReason || coach.intent !== "none",
      speakReason: speakReason ?? (coach.intent !== "none" ? `coach:${coach.intent}` : undefined),
      recentChanges: recentChanges.slice(0, 8),
      entity: this.lastEntity,
    });

    this.prevSituationPrimary = situation.primary;
    this.lastHuman = human;
    this.lastSnapshot = snapshot;
    this.lastFaceTouchSec = faceTrack.durationSec;
    this.lastWaveRising = waveTrack.rising;
    this.lastDialogue = dialogue;

    return { snapshot, dialogue };
  }
}

export const isVision2Enabled = (settings?: { vision2Enabled?: boolean }): boolean =>
  settings?.vision2Enabled !== false;
