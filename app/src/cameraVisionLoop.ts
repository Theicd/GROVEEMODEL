import { captureVideoFrame } from "./cameraMode";
import { CharacterBrain, moodStatusLine, type CharacterDecision, type CharacterMood } from "./characterBrain";
import { FrameStabilityScorer } from "./frameStability";
import { LightSceneDetector, preloadLightDetector } from "./lightSceneDetector";
import { motionFromSamples, type MotionSnapshot } from "./motionLayer";
import { PoseDetectorLoop, preloadPoseDetector, type PoseSource } from "./poseDetector";
import {
  captureDownscaledFrame,
  frameDiffScore,
  isSignificantChange,
  type FrameSample,
} from "./sceneChangeDetector";
import { formatFreshPersonBlock, type PersonFocusSnapshot } from "./personFocus";
import { buildSensorBlock, poseFromWorld } from "./sensorBlock";
import { SituationEngine } from "./situationEngine";
import { isTfVisionPaused } from "./visionCoordination";
import {
  deepVisionBackoffMs,
  detectVisionBudget,
  mergeCameraLoopTiming,
  type VisionBudgetProfile,
} from "./visionBudget";
import type { SemanticEvent, WorldMemory, WorldUpdateResult } from "./worldMemory";

export const CAMERA_LOOP_CONFIG = {
  pollIntervalMs: 4000,
  motionThreshold: 0.05,
  deepVisionMinIntervalMs: 120_000,
  deepVisionMaxIntervalMs: 300_000,
  postChatCooldownMs: 45_000,
  chatHoldMs: 45_000,
  analysisMaxWidth: 512,
  minFrameStabilityForDeep: 0.35,
} as const;

export type CameraLoopRuntimeConfig = typeof CAMERA_LOOP_CONFIG;

export type SceneAnalysisRequest = {
  bytes: ArrayBuffer;
  previousSummary: string;
  reason: string;
  sensorBlock?: string;
};

export type SceneAnalysisResult = {
  objects: string[];
  people: string[];
  current: string[];
  events: string[];
  interesting: boolean;
  summary: string;
};

export type CameraVisionLoopCallbacks = {
  requestAnalysis: (req: SceneAnalysisRequest) => Promise<SceneAnalysisResult | null>;
  resolveUtterance?: (decision: CharacterDecision) => Promise<CharacterDecision>;
  /** When false, proactive lines use CharacterBrain fallback (no extra Gemma call). */
  useLlmProactiveUtterance?: () => boolean;
  isWorkerBusy: () => boolean;
  onCameraStatus?: (text: string) => void;
  onCharacterSpeak?: (decision: CharacterDecision) => void;
  onMoodChange?: (mood: CharacterMood) => void;
  onObservingChange?: (observing: boolean) => void;
  onAnalysisScheduled?: (reason: string, stage?: "light" | "deep" | "pose" | "situation") => void;
  onLightDetection?: (payload: {
    objects: string[];
    rawPeople: number;
    debouncedPeople: string[];
    personJustConfirmed: boolean;
    personJustLeft: boolean;
    worldUpdate: WorldUpdateResult;
  }) => void;
  onMotionUpdate?: (payload: {
    motionLevel: number;
    upperMotion: number;
    armMovement: boolean;
    waveLike: boolean;
    personPresent: boolean;
    newEvents: SemanticEvent[];
  }) => void;
  onSituationUpdate?: (payload: {
    poseState: string;
    gestures: string[];
    holding: string[];
    focusHint: string;
    poseSource: PoseSource;
    poseConfidence?: number;
    newEvents: SemanticEvent[];
  }) => void;
};

export type { PersonFocusSnapshot };
export { formatFreshPersonBlock };

export class CameraVisionLoop {
  private timer: ReturnType<typeof setInterval> | null = null;
  private paused = false;
  private analyzing = false;
  private utterancePending = false;
  private holdUntil = 0;
  private motionFrame: FrameSample | null = null;
  private lastFrame: FrameSample | null = null;
  private lastMotionSnap: MotionSnapshot | null = null;
  private lastFrameScore = 0;
  private lastDeepVisionAt = 0;
  private deepBaselineDone = false;
  private deepVisionFailures = 0;
  private deepVisionBackoffUntil = 0;
  private deepVisionDegraded = false;
  private readonly budget: VisionBudgetProfile;
  private readonly loopConfig: CameraLoopRuntimeConfig;
  private readonly lightDetector = new LightSceneDetector();
  private readonly poseDetector = new PoseDetectorLoop();
  private readonly situationEngine = new SituationEngine();
  private readonly frameScorer = new FrameStabilityScorer();
  private readonly world: WorldMemory;
  private readonly character: CharacterBrain;
  private readonly video: HTMLVideoElement;
  private readonly callbacks: CameraVisionLoopCallbacks;

  constructor(
    video: HTMLVideoElement,
    world: WorldMemory,
    character: CharacterBrain,
    callbacks: CameraVisionLoopCallbacks,
    budget: VisionBudgetProfile = detectVisionBudget(),
  ) {
    this.video = video;
    this.world = world;
    this.character = character;
    this.callbacks = callbacks;
    this.budget = budget;
    this.loopConfig = mergeCameraLoopTiming(CAMERA_LOOP_CONFIG, budget);
  }

  isDeepVisionDegraded(): boolean {
    return this.deepVisionDegraded;
  }

  getBudgetTier(): VisionBudgetProfile["tier"] {
    return this.budget.tier;
  }

  start(): void {
    this.stop();
    this.paused = false;
    this.holdUntil = 0;
    this.motionFrame = null;
    this.lastFrame = null;
    this.lastMotionSnap = null;
    this.lastFrameScore = 0;
    this.lastDeepVisionAt = 0;
    this.deepBaselineDone = false;
    this.deepVisionFailures = 0;
    this.deepVisionBackoffUntil = 0;
    this.deepVisionDegraded = false;
    this.lightDetector.reset();
    this.poseDetector.reset();
    this.situationEngine.reset();
    this.frameScorer.reset();
    this.character.mood = "observing";
    this.callbacks.onObservingChange?.(true);
    this.callbacks.onMoodChange?.("observing");
    void preloadLightDetector().catch((e) => console.warn("[GROVEE] COCO-SSD preload", e));
    if (this.budget.preloadPose) {
      void preloadPoseDetector()
        .then((ok) => {
          if (ok) this.callbacks.onCameraStatus?.("👁 MoveNet · מוכן");
        })
        .catch((e) => console.warn("[GROVEE] MoveNet preload", e));
    }
    const tierNote =
      this.budget.tier === "low"
        ? ` · חיסכון (${this.budget.reason})`
        : "";
    this.callbacks.onCameraStatus?.(`👁 Character · תצפית${tierNote}`);
    this.timer = setInterval(() => void this.tick(), this.loopConfig.pollIntervalMs);
    document.addEventListener("visibilitychange", this.onVisibility);
  }

  stop(): void {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
    document.removeEventListener("visibilitychange", this.onVisibility);
    this.motionFrame = null;
    this.lastFrame = null;
    this.lastMotionSnap = null;
    this.holdUntil = 0;
    this.lightDetector.reset();
    this.poseDetector.reset();
    this.situationEngine.reset();
    this.frameScorer.reset();
    this.callbacks.onObservingChange?.(false);
    this.callbacks.onCameraStatus?.("");
  }

  pause(): void {
    this.paused = true;
    this.callbacks.onObservingChange?.(false);
  }

  resume(): void {
    if (!this.timer) return;
    this.paused = false;
    this.callbacks.onObservingChange?.(true);
  }

  holdForChat(ms = CAMERA_LOOP_CONFIG.chatHoldMs): void {
    this.holdUntil = Math.max(this.holdUntil, Date.now() + ms);
    this.callbacks.onCameraStatus?.("👁 Character · ממתין לצ'אט");
  }

  releaseAfterChat(cooldownMs = CAMERA_LOOP_CONFIG.postChatCooldownMs): void {
    this.holdUntil = Math.max(this.holdUntil, Date.now() + cooldownMs);
    this.character.recordUserInteraction();
    this.callbacks.onMoodChange?.("observing");
    this.callbacks.onCameraStatus?.(moodStatusLine("observing"));
  }

  isAnalyzing(): boolean {
    return this.analyzing;
  }

  async captureFreshSnapshot(): Promise<ArrayBuffer> {
    return captureVideoFrame(this.video, CAMERA_LOOP_CONFIG.analysisMaxWidth);
  }

  /**
   * Focus On Person — fresh COCO + MoveNet + situation before answering posture/activity questions.
   */
  async refreshPersonFocus(): Promise<PersonFocusSnapshot | null> {
    if (this.video.readyState < 2) return null;
    if (this.callbacks.isWorkerBusy() || isTfVisionPaused()) return null;

    const sample = captureDownscaledFrame(this.video);
    if (sample && this.lastFrame) {
      this.lastMotionSnap = motionFromSamples(this.lastFrame, sample);
    }
    if (sample) this.lastFrame = sample;

    const light = await this.lightDetector.run(this.video, { force: true });
    if (!light) return null;

    this.world.applyLightDetection({
      objects: light.objects,
      people: light.debouncedPeople,
    });

    const personInFrame = light.debouncedPeople.length > 0;
    if (!personInFrame) {
      return {
        personPresent: false,
        poseState: "unknown",
        confidence: 0,
        holding: [],
        gestures: [],
        focusHint: "",
        poseSource: "none",
        capturedAt: Date.now(),
        validationFrames: 1,
      };
    }

    await this.runSituationPass(light, true, { skipCharacterDispatch: true });

    const raw = this.situationEngine.getLastRawInference();
    this.world.poseState = raw.poseState;
    this.world.poseConfidence = raw.confidence;
    this.world.poseUpdatedAt = Date.now();

    return {
      personPresent: true,
      poseState: raw.poseState,
      confidence: raw.confidence,
      holding: [...this.world.holding],
      gestures: [...this.world.gestures],
      focusHint: this.world.focusHint,
      poseSource: this.poseDetector.lastSource,
      capturedAt: Date.now(),
      validationFrames: 1,
    };
  }

  private onVisibility = (): void => {
    if (document.hidden) this.pause();
    else this.resume();
  };

  private msSinceDeepVision(): number {
    if (this.lastDeepVisionAt <= 0) return Number.POSITIVE_INFINITY;
    return Date.now() - this.lastDeepVisionAt;
  }

  private emitCharacter(decision: CharacterDecision): void {
    this.character.markSpoke(decision);
    this.callbacks.onMoodChange?.(decision.mood);

    const llmUtterance =
      this.callbacks.useLlmProactiveUtterance?.() ?? this.budget.useLlmProactiveUtterance;

    if (
      this.callbacks.resolveUtterance &&
      llmUtterance &&
      !this.utterancePending &&
      !this.callbacks.isWorkerBusy() &&
      Date.now() >= this.holdUntil
    ) {
      this.utterancePending = true;
      void this.callbacks
        .resolveUtterance(decision)
        .then((final) => {
          this.utterancePending = false;
          this.callbacks.onCharacterSpeak?.(final);
          this.callbacks.onCameraStatus?.(moodStatusLine(final.mood));
        })
        .catch(() => {
          this.utterancePending = false;
          this.callbacks.onCharacterSpeak?.(decision);
          this.callbacks.onCameraStatus?.(moodStatusLine(decision.mood));
        });
      return;
    }

    this.callbacks.onCharacterSpeak?.(decision);
    this.callbacks.onCameraStatus?.(moodStatusLine(decision.mood));
  }

  private dispatchCharacter(newEvents: SemanticEvent[]): void {
    if (!newEvents.length) return;
    const decision = this.character.evaluate(this.world, newEvents);
    if (!decision) return;
    this.emitCharacter(decision);
  }

  private async runSituationPass(
    light: NonNullable<Awaited<ReturnType<LightSceneDetector["run"]>>>,
    personInFrame: boolean,
    options?: { skipCharacterDispatch?: boolean },
  ): Promise<void> {
    this.callbacks.onAnalysisScheduled?.("pose_detect", "pose");
    const personBbox = personInFrame ? (light.personBoxes[0] ?? null) : null;
    const poseResult = personInFrame
      ? await this.poseDetector.run(this.video, personBbox, this.lastMotionSnap)
      : { keypoints: null, source: "none" as const };
    const { keypoints, source: poseSource } = poseResult;
    const objectBoxes = light.boxes
      .filter((b) => b.label !== "person")
      .map((b) => ({ label: b.label, bbox: b }));

    const motion =
      this.lastMotionSnap ??
      ({
        motionLevel: 0,
        upperMotion: 0,
        lowerMotion: 0,
        armMovement: false,
        waveLike: false,
        bodyMovement: false,
      } satisfies MotionSnapshot);

    const { pose, events } = this.situationEngine.analyze({
      keypoints,
      personBbox,
      objectBoxes,
      motion,
      personInFrame,
    });
    this.situationEngine.applyToWorld(this.world, pose, events, poseSource);

    this.callbacks.onSituationUpdate?.({
      poseState: pose.poseState,
      gestures: pose.gestures,
      holding: pose.holding,
      focusHint: pose.focusHint,
      poseSource,
      poseConfidence: pose.confidence,
      newEvents: events,
    });

    if (events.length && !options?.skipCharacterDispatch) this.dispatchCharacter(events);
  }

  private async tick(): Promise<void> {
    if (this.paused || this.analyzing) return;
    if (Date.now() < this.holdUntil) return;
    if (this.callbacks.isWorkerBusy() || isTfVisionPaused()) {
      this.callbacks.onCameraStatus?.("👁 Character · Gemma פעיל — ממתין");
      return;
    }

    const sample = captureDownscaledFrame(this.video);
    if (!sample) return;

    this.lastFrameScore = this.frameScorer.scoreFrame(sample.imageData);

    if (this.lastFrame) {
      const motionSnap = motionFromSamples(this.lastFrame, sample);
      this.lastMotionSnap = motionSnap;
      const motionResult = this.world.applyMotion(motionSnap, this.world.people.length > 0);
      this.callbacks.onMotionUpdate?.({
        motionLevel: motionSnap.motionLevel,
        upperMotion: motionSnap.upperMotion,
        armMovement: motionSnap.armMovement,
        waveLike: motionSnap.waveLike,
        personPresent: this.world.people.length > 0,
        newEvents: motionResult.newEvents,
      });
      if (motionResult.newEvents.length) {
        this.dispatchCharacter(motionResult.newEvents);
      }
    }
    this.lastFrame = sample;

    const isFirst = !this.motionFrame;
    let hasMotion = isFirst;

    if (this.motionFrame && !isFirst) {
      const diff = frameDiffScore(this.motionFrame.imageData, sample.imageData);
      hasMotion = isSignificantChange(diff, CAMERA_LOOP_CONFIG.motionThreshold);
    }

    const urgentMotion =
      !!this.lastMotionSnap &&
      (this.lastMotionSnap.waveLike ||
        this.lastMotionSnap.armMovement ||
        this.lastMotionSnap.motionLevel >= 0.22);
    const canRunLight = isFirst || this.lightDetector.canRun(undefined, { urgent: urgentMotion });
    if (!canRunLight) {
      this.maybeCharacterTick();
      this.callbacks.onCameraStatus?.(moodStatusLine(this.character.mood));
      return;
    }

    if (this.callbacks.isWorkerBusy() || isTfVisionPaused()) {
      this.maybeCharacterTick();
      this.callbacks.onCameraStatus?.("👁 Character · Gemma פעיל");
      return;
    }

    this.callbacks.onCameraStatus?.("👁 COCO · סורק…");
    const light = await this.lightDetector.run(this.video, { force: isFirst });
    if (!light) {
      this.maybeCharacterTick();
      return;
    }

    this.callbacks.onAnalysisScheduled?.("light_detect", "light");

    const lightResult = this.world.applyLightDetection({
      objects: light.objects,
      people: light.debouncedPeople,
    });

    this.callbacks.onLightDetection?.({
      objects: light.objects,
      rawPeople: light.personCount,
      debouncedPeople: light.debouncedPeople,
      personJustConfirmed: light.personJustConfirmed,
      personJustLeft: light.personJustLeft,
      worldUpdate: lightResult,
    });

    const personInFrame = light.debouncedPeople.length > 0;
    if (!personInFrame) {
      this.world.clearPersonLayer();
      this.world.pruneStalePersonActivity();
    }

    await this.runSituationPass(light, personInFrame);

    if (lightResult.isBaselineCapture) {
      this.character.noteBaselineScene();
      this.callbacks.onMoodChange?.("observing");
      if (hasMotion) this.motionFrame = sample;
      this.maybeCharacterTick();
      if (!this.deepBaselineDone) {
        if (this.budget.useLlmDeepVision) {
          await this.runDeepVision("first_deep_summary");
        }
        if (!this.deepBaselineDone) {
          this.finishSensorOnlyBaseline();
        }
      }
      return;
    }

    if (!lightResult.suppressedAsChurn) {
      this.dispatchCharacter(lightResult.newEvents);
    }

    if (hasMotion) this.motionFrame = sample;

    const deepReason = this.pickDeepVisionReason(lightResult);
    if (deepReason && !this.callbacks.isWorkerBusy() && !isTfVisionPaused()) {
      await this.runDeepVision(deepReason);
    } else {
      this.maybeCharacterTick();
    }
  }

  private pickDeepVisionReason(lightResult: WorldUpdateResult): string | null {
    if (!this.budget.useLlmDeepVision || this.deepVisionDegraded) return null;
    if (Date.now() < this.deepVisionBackoffUntil) return null;

    const since = this.msSinceDeepVision();
    if (!this.deepBaselineDone) return "first_deep_summary";
    if (since < this.loopConfig.deepVisionMinIntervalMs) return null;
    if (this.lastFrameScore < this.loopConfig.minFrameStabilityForDeep && since < this.loopConfig.deepVisionMaxIntervalMs) {
      return null;
    }
    if (this.world.hasSignificantLightEvent(lightResult)) {
      const personEv = lightResult.newEvents.some(
        (e) => e.type === "person_entered" || e.type === "user_returned",
      );
      if (personEv) return "person_detected";
      return "significant_object";
    }
    if (since >= this.loopConfig.deepVisionMaxIntervalMs) return "heartbeat";
    return null;
  }

  private finishSensorOnlyBaseline(): void {
    const sensorBlock = buildSensorBlock(this.world, poseFromWorld(this.world));
    this.world.applySensorBaseline(sensorBlock);
    this.deepBaselineDone = true;
    this.character.noteBaselineScene();
    this.callbacks.onCameraStatus?.("👁 Character · תצפית (קל)");
    this.maybeCharacterTick();
  }

  private noteDeepVisionFailure(reason: string): void {
    this.deepVisionFailures++;
    this.deepVisionBackoffUntil =
      Date.now() + deepVisionBackoffMs(this.budget, this.deepVisionFailures);
    if (this.deepVisionFailures >= this.budget.maxDeepVisionFailures) {
      this.deepVisionDegraded = true;
      this.deepBaselineDone = true;
      const sensorBlock = buildSensorBlock(this.world, poseFromWorld(this.world));
      this.world.applySensorBaseline(sensorBlock);
      this.callbacks.onCameraStatus?.("👁 Character · מצב קל (ללא Gemma עמוק)");
      console.warn(
        `[GROVEE camera] deep vision paused after ${this.deepVisionFailures} failures (${reason})`,
      );
      this.maybeCharacterTick();
    }
  }

  private async runDeepVision(reason: string): Promise<void> {
    if (this.analyzing || this.callbacks.isWorkerBusy() || Date.now() < this.holdUntil) return;
    this.analyzing = true;
    this.callbacks.onAnalysisScheduled?.(reason, "deep");
    this.callbacks.onCameraStatus?.(`👁 Gemma · ${reason}…`);

    try {
      const bytes = await captureVideoFrame(this.video, this.loopConfig.analysisMaxWidth);
      if (bytes.byteLength < this.budget.minSnapshotBytes) {
        console.warn(`[GROVEE camera] skip deep vision — snapshot too small (${bytes.byteLength}B)`);
        if (reason === "first_deep_summary") this.finishSensorOnlyBaseline();
        return;
      }

      const sensorBlock = buildSensorBlock(this.world, poseFromWorld(this.world));
      const result = await this.callbacks.requestAnalysis({
        bytes,
        previousSummary: this.world.lastSummary,
        reason,
        sensorBlock,
      });

      this.lastDeepVisionAt = Date.now();

      if (!result) {
        this.noteDeepVisionFailure(reason);
        this.callbacks.onCameraStatus?.(moodStatusLine(this.character.mood));
        return;
      }

      this.deepVisionFailures = 0;
      this.deepVisionBackoffUntil = 0;
      this.deepVisionDegraded = false;

      this.world.applyDeepVision({
        objects: result.objects.length ? result.objects : result.current,
        summary: result.summary,
      });

      if (reason === "first_deep_summary") {
        this.deepBaselineDone = true;
      }

      this.callbacks.onCameraStatus?.(moodStatusLine(this.character.mood));
      this.maybeCharacterTick();
    } catch (e) {
      console.warn("[GROVEE camera deep]", e);
      this.callbacks.onCameraStatus?.("👁 Character · שגיאה");
    } finally {
      this.analyzing = false;
    }
  }

  private maybeCharacterTick(): void {
    if (this.paused || this.analyzing) return;
    if (Date.now() < this.holdUntil) return;
    if (this.callbacks.isWorkerBusy()) return;
    if (!this.world.hasSensorContext()) return;

    const decision = this.character.evaluate(this.world, []);
    if (!decision) {
      this.callbacks.onCameraStatus?.(moodStatusLine(this.character.mood));
      return;
    }

    this.emitCharacter(decision);
  }
}

export type { CharacterDecision, CharacterMood };
