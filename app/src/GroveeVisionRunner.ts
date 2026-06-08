import { captureVideoFrame } from "./cameraMode";
import { CharacterBrain, moodStatusLine, type CharacterDecision, type CharacterMood } from "./characterBrain";
import { formatFreshPersonBlock, type PersonFocusSnapshot } from "./personFocus";
import { buildRichSensorBlock, mapLabGestures, mapLabHolding, mapPoseActionToState } from "./visionBridge";
import {
  createGroveeVisionSyncState,
  syncVisionResultToWorld,
  type GroveeVisionSyncState,
} from "./groveeVisionSync";
import { isTfVisionPaused } from "./visionCoordination";
import {
  deepVisionBackoffMs,
  mergeCameraLoopTiming,
  type VisionBudgetProfile,
} from "./visionBudget";
import type { SemanticEvent, WorldMemory } from "./worldMemory";
import { VisionPipeline } from "./vision-lab/core/VisionPipeline";
import { ensureVisionLabConfig } from "./vision-lab/core/configStorage";
import type { PipelineConfig, VisionResult } from "./vision-lab/core/types";

export const GROVEE_VISION_LOOP_CONFIG = {
  deepVisionMinIntervalMs: 120_000,
  deepVisionMaxIntervalMs: 300_000,
  postChatCooldownMs: 45_000,
  chatHoldMs: 45_000,
  analysisMaxWidth: 512,
} as const;

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

export type GroveeVisionCallbacks = {
  requestAnalysis: (req: SceneAnalysisRequest) => Promise<SceneAnalysisResult | null>;
  resolveUtterance?: (decision: CharacterDecision) => Promise<CharacterDecision>;
  useLlmProactiveUtterance?: () => boolean;
  useBootDeepSnapshot?: () => boolean;
  isWorkerBusy: () => boolean;
  onCameraStatus?: (text: string) => void;
  onCharacterSpeak?: (decision: CharacterDecision) => void;
  onMoodChange?: (mood: CharacterMood) => void;
  onObservingChange?: (observing: boolean) => void;
  onVisionResult?: (result: VisionResult) => void;
  onPipelineProgress?: (msg: string) => void;
  onLightDetection?: (payload: {
    objects: string[];
    rawPeople: number;
    debouncedPeople: string[];
    personJustConfirmed: boolean;
    personJustLeft: boolean;
    worldUpdate: import("./worldMemory").WorldUpdateResult;
  }) => void;
  onSituationUpdate?: (payload: {
    poseState: string;
    gestures: string[];
    holding: string[];
    focusHint: string;
    poseSource: string;
    poseConfidence?: number;
    newEvents: SemanticEvent[];
  }) => void;
  onLabEvents?: (events: SemanticEvent[]) => void;
};

export type { PersonFocusSnapshot };
export { formatFreshPersonBlock };
export type { CharacterDecision, CharacterMood };

export class GroveeVisionRunner {
  private readonly pipeline = new VisionPipeline();
  private readonly world: WorldMemory;
  private readonly character: CharacterBrain;
  private video: HTMLVideoElement | null = null;
  private readonly callbacks: GroveeVisionCallbacks;
  private readonly budget: VisionBudgetProfile;
  private readonly syncState: GroveeVisionSyncState = createGroveeVisionSyncState();

  private holdUntil = 0;
  private utterancePending = false;
  private analyzing = false;
  private deepBaselineDone = false;
  private deepVisionFailures = 0;
  private deepVisionBackoffUntil = 0;
  private deepVisionDegraded = false;
  private lastDeepVisionAt = 0;
  private latest: VisionResult | null = null;
  private uiFlushTimer: ReturnType<typeof setInterval> | null = null;
  private watchdogTimer: ReturnType<typeof setInterval> | null = null;
  private visibilityBound = false;
  private lastUiEmit = 0;

  constructor(
    world: WorldMemory,
    character: CharacterBrain,
    callbacks: GroveeVisionCallbacks,
    budget: VisionBudgetProfile,
    initialConfig: PipelineConfig,
  ) {
    this.world = world;
    this.character = character;
    this.callbacks = callbacks;
    this.budget = budget;

    this.applyPipelineConfig(initialConfig);
    this.pipeline.setOnUpdate((r) => {
      this.latest = r;
      const now = performance.now();
      const uiMs = this.pipeline.getUiUpdateMs();
      if (now - this.lastUiEmit < uiMs) return;
      this.lastUiEmit = now;
      this.callbacks.onVisionResult?.({ ...r });
    });
    this.pipeline.setOnProgress((msg) => this.callbacks.onPipelineProgress?.(msg));
  }

  bindVideo(video: HTMLVideoElement): void {
    this.video = video;
  }

  applyPipelineConfig(config: PipelineConfig): void {
    const tier = this.budget.tier === "low" ? "low" : "normal";
    this.pipeline.setConfig(ensureVisionLabConfig(config, tier));
    this.lastUiEmit = 0;
  }

  private requireVideo(): HTMLVideoElement {
    if (!this.video) throw new Error("Camera video element not bound");
    return this.video;
  }

  getPipeline(): VisionPipeline {
    return this.pipeline;
  }

  getLatestResult(): VisionResult | null {
    return this.latest ?? this.pipeline.getLatest();
  }

  async start(): Promise<void> {
    this.callbacks.onObservingChange?.(true);
    this.callbacks.onCameraStatus?.("🔬 Vision Lab · טוען מודלים…");
    const video = this.requireVideo();
    await this.pipeline.init();
    this.pipeline.start(video);

    const uiMs = this.pipeline.getUiUpdateMs();
    this.uiFlushTimer = setInterval(() => this.onUiTick(), uiMs);
    this.watchdogTimer = setInterval(() => this.watchdogPipeline(), 2500);

    if (!this.visibilityBound) {
      document.addEventListener("visibilitychange", this.onVisibility);
      this.visibilityBound = true;
    }

    this.callbacks.onCameraStatus?.("👁 Character · צופה");
  }

  stop(): void {
    if (this.uiFlushTimer) {
      clearInterval(this.uiFlushTimer);
      this.uiFlushTimer = null;
    }
    if (this.watchdogTimer) {
      clearInterval(this.watchdogTimer);
      this.watchdogTimer = null;
    }
    this.pipeline.stop();
    if (this.visibilityBound) {
      document.removeEventListener("visibilitychange", this.onVisibility);
      this.visibilityBound = false;
    }
    this.callbacks.onObservingChange?.(false);
  }

  dispose(): void {
    this.stop();
    this.pipeline.dispose();
  }

  /**
   * Legacy hook — vision models are never auto-paused (user toggles only).
   */
  pauseForChatInference(): void {
    this.callbacks.onCameraStatus?.("👁 Character · צופה (כל המודלים פעילים)");
  }

  /** Legacy hook — no-op; models stay running. */
  resumeAfterChatInference(): void {
    const video = this.video;
    if (video && video.readyState >= 2) {
      const last = this.pipeline.getLastFrameAt();
      if (!last || Date.now() - last > 3000) {
        this.pipeline.start(video);
      }
    }
    if (!this.uiFlushTimer) {
      const uiMs = this.pipeline.getUiUpdateMs();
      this.uiFlushTimer = setInterval(() => this.onUiTick(), uiMs);
    }
    if (!this.watchdogTimer) {
      this.watchdogTimer = setInterval(() => this.watchdogPipeline(), 2500);
    }
    this.callbacks.onCameraStatus?.("👁 Character · צופה");
  }

  private watchdogPipeline(): void {
    if (!this.video) return;
    const last = this.pipeline.getLastFrameAt();
    if (!last) return;
    const staleMs = Date.now() - last;
    if (staleMs < 4000) return;

    console.warn(`[GROVEE vision] pipeline stalled ${staleMs}ms — restarting`);
    this.pipeline.stop();
    if (this.video.readyState >= 2) {
      this.pipeline.start(this.video);
    }
    this.callbacks.onCameraStatus?.("👁 Vision · הופעל מחדש (ניטור נעצר)");
  }

  isPipelinePaused(): boolean {
    return false;
  }

  holdForChat(ms = GROVEE_VISION_LOOP_CONFIG.chatHoldMs): void {
    this.holdUntil = Math.max(this.holdUntil, Date.now() + ms);
    this.callbacks.onCameraStatus?.("👁 Character · ממתין לצ'אט");
  }

  releaseAfterChat(cooldownMs = GROVEE_VISION_LOOP_CONFIG.postChatCooldownMs): void {
    this.holdUntil = Math.max(this.holdUntil, Date.now() + cooldownMs);
    this.character.recordUserInteraction();
    this.callbacks.onMoodChange?.("observing");
    this.callbacks.onCameraStatus?.(moodStatusLine("observing"));
  }

  isAnalyzing(): boolean {
    return this.analyzing;
  }

  isDeepVisionDegraded(): boolean {
    return this.deepVisionDegraded;
  }

  isDeepBaselineDone(): boolean {
    return this.deepBaselineDone;
  }

  async captureFreshSnapshot(): Promise<ArrayBuffer> {
    return captureVideoFrame(this.requireVideo(), GROVEE_VISION_LOOP_CONFIG.analysisMaxWidth);
  }

  async refreshPersonFocus(): Promise<PersonFocusSnapshot | null> {
    const result = this.getLatestResult();
    const video = this.video;
    if (!result || !video || video.readyState < 2) return null;
    if (isTfVisionPaused()) return null;

    const hasPerson = result.objects.some((o) => o.label === "person" && o.confidence >= 0.45);
    if (!hasPerson) {
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

    const pose = mapPoseActionToState(result.poseActions);
    return {
      personPresent: true,
      poseState: pose.poseState,
      confidence: pose.confidence,
      holding: mapLabHolding(result),
      gestures: mapLabGestures(result),
      focusHint: result.interactions.map((i) => i.name).join(", "),
      poseSource: "vision-lab",
      capturedAt: Date.now(),
      validationFrames: 1,
    };
  }

  private onVisibility = (): void => {
    // Keep all vision models running when tab is hidden — no auto-stop.
    if (document.hidden) return;
    const video = this.video;
    if (video && video.readyState >= 2) {
      const last = this.pipeline.getLastFrameAt();
      if (!last || Date.now() - last > 3000) {
        this.pipeline.start(video);
      }
    }
  };

  private onUiTick(): void {
    const result = this.getLatestResult();
    if (!result) return;

    this.callbacks.onVisionResult?.(result);

    const sync = syncVisionResultToWorld(this.world, result, this.syncState);
    const objectLabels = result.objects
      .filter((o) => o.label !== "person")
      .map((o) => o.displayLabel || o.label);
    const rawPeople = result.objects.filter((o) => o.label === "person").length;

    this.callbacks.onLightDetection?.({
      objects: objectLabels,
      rawPeople,
      debouncedPeople: sync.personPresent ? ["person"] : [],
      personJustConfirmed: sync.personJustConfirmed,
      personJustLeft: sync.personJustLeft,
      worldUpdate: sync.worldUpdate,
    });

    if (sync.labEvents.length) {
      this.callbacks.onSituationUpdate?.({
        poseState: this.world.poseState,
        gestures: this.world.gestures,
        holding: this.world.holding,
        focusHint: this.world.focusHint,
        poseSource: "vision-lab",
        poseConfidence: this.world.poseConfidence,
        newEvents: sync.labEvents,
      });
      this.callbacks.onLabEvents?.(sync.labEvents);
      this.dispatchCharacter(sync.labEvents);
    }

    if (sync.worldUpdate.isBaselineCapture && !this.deepBaselineDone) {
      this.character.noteBaselineScene();
      this.callbacks.onMoodChange?.("observing");
      const wantBoot =
        (this.callbacks.useBootDeepSnapshot?.() ?? true) && this.budget.useLlmDeepVision;
      if (wantBoot) {
        void this.runDeepVision("first_deep_summary");
      } else {
        this.finishSensorOnlyBaseline(result);
      }
      return;
    }

    if (!sync.worldUpdate.suppressedAsChurn && sync.worldUpdate.newEvents.length) {
      this.dispatchCharacter(sync.worldUpdate.newEvents);
    }

    void this.maybeDeepVision(sync.worldUpdate.newEvents.length > 0);

    if (Date.now() >= this.holdUntil && !this.callbacks.isWorkerBusy()) {
      this.maybeCharacterTick();
    }
  }

  private finishSensorOnlyBaseline(result?: VisionResult | null): void {
    const sensorBlock = buildRichSensorBlock(this.world, result);
    this.world.applySensorBaseline(sensorBlock);
    if (result?.sceneDescription?.trim() && !this.world.bootContext.trim()) {
      this.world.bootContext = result.sceneDescription.trim().slice(0, 320);
    }
    this.deepBaselineDone = true;
    this.character.noteBaselineScene();
    this.callbacks.onCameraStatus?.("👁 Character · תצפית (Vision Lab)");
    this.maybeCharacterTick();
  }

  private maybeDeepVision(significantChange: boolean): void {
    if (!this.budget.useLlmDeepVision || this.deepVisionDegraded) return;
    if (Date.now() < this.deepVisionBackoffUntil) return;
    if (this.analyzing || isTfVisionPaused()) return;

    const timing = mergeCameraLoopTiming(
      {
        pollIntervalMs: this.budget.pollIntervalMs,
        deepVisionMinIntervalMs: GROVEE_VISION_LOOP_CONFIG.deepVisionMinIntervalMs,
        deepVisionMaxIntervalMs: GROVEE_VISION_LOOP_CONFIG.deepVisionMaxIntervalMs,
        minFrameStabilityForDeep: 0.35,
      },
      this.budget,
    );
    const since = this.lastDeepVisionAt <= 0 ? Number.POSITIVE_INFINITY : Date.now() - this.lastDeepVisionAt;

    let reason: string | null = null;
    if (!this.deepBaselineDone) {
      reason = "first_deep_summary";
    } else if (significantChange && since >= timing.deepVisionMinIntervalMs) {
      reason = "significant_change";
    } else if (since >= timing.deepVisionMaxIntervalMs) {
      reason = "heartbeat";
    }

    if (reason) void this.runDeepVision(reason);
  }

  private noteDeepVisionFailure(reason: string): void {
    this.deepVisionFailures++;
    this.deepVisionBackoffUntil =
      Date.now() + deepVisionBackoffMs(this.budget, this.deepVisionFailures);
    if (this.deepVisionFailures >= this.budget.maxDeepVisionFailures) {
      this.deepVisionDegraded = true;
      this.deepBaselineDone = true;
      this.finishSensorOnlyBaseline(this.getLatestResult());
      this.callbacks.onCameraStatus?.("👁 Character · מצב קל (ללא Gemma עמוק)");
      console.warn(
        `[GROVEE vision] deep vision paused after ${this.deepVisionFailures} failures (${reason})`,
      );
    }
  }

  private async runDeepVision(reason: string): Promise<void> {
    if (this.analyzing) return;
    if (!this.budget.useLlmDeepVision) {
      this.finishSensorOnlyBaseline(this.getLatestResult());
      return;
    }

    this.analyzing = true;
    this.callbacks.onCameraStatus?.(`👁 Gemma · ${reason}… (YOLO + face keep running)`);

    try {
      const bytes = await captureVideoFrame(this.requireVideo(), GROVEE_VISION_LOOP_CONFIG.analysisMaxWidth);
      if (bytes.byteLength < this.budget.minSnapshotBytes) {
        if (reason === "first_deep_summary") this.finishSensorOnlyBaseline(this.getLatestResult());
        return;
      }

      const sensorBlock = buildRichSensorBlock(this.world, this.getLatestResult());
      const result = await this.callbacks.requestAnalysis({
        bytes,
        previousSummary: this.world.bootContext || this.world.lastSummary,
        reason,
        sensorBlock,
      });

      this.lastDeepVisionAt = Date.now();

      if (!result) {
        this.noteDeepVisionFailure(reason);
        return;
      }

      this.deepVisionFailures = 0;
      this.deepVisionBackoffUntil = 0;
      this.deepVisionDegraded = false;
      this.world.applyDeepVision({
        objects: result.objects.length ? result.objects : result.current,
        summary: result.summary,
      });

      if (reason === "first_deep_summary") this.deepBaselineDone = true;
      this.callbacks.onCameraStatus?.(moodStatusLine(this.character.mood));
      this.maybeCharacterTick();
    } catch (e) {
      console.warn("[GROVEE vision deep]", e);
      this.callbacks.onCameraStatus?.("👁 Character · שגיאה");
    } finally {
      this.analyzing = false;
    }
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

  private maybeCharacterTick(): void {
    if (this.analyzing) return;
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
