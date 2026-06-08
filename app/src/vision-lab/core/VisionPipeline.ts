import { YoloDetector } from '../detectors/YoloDetector';
import { PoseDetector } from '../detectors/PoseDetector';
import { HandsDetector } from '../detectors/HandsDetector';
import { FaceEmotionDetector } from '../detectors/FaceEmotionDetector';
import { recognizeStaticGestures, getFingerState } from '../analyzers/GestureRecognizer';
import { MotionGestureDetector } from '../analyzers/MotionGestureDetector';
import { PoseActionClassifier } from '../analyzers/PoseActionClassifier';
import { analyzeInteractions } from '../analyzers/InteractionAnalyzer';
import { evaluateEvents } from '../analyzers/EventRuleEngine';
import { interpretBodyLanguage } from '../analyzers/BodyLanguageInterpreter';
import { classifyEnvironment } from '../analyzers/EnvironmentClassifier';
import { buildRuleBasedDescription } from '../analyzers/SceneDescriptionGenerator';
import { VlmSceneDescriber } from '../analyzers/VlmSceneDescriber';
import {
  getModuleOverdue,
  intervalsFromMode,
  isModuleDue,
  pickHeavyModule,
  resolveSchedule,
} from './schedule';
import type { PipelineConfig, VisionResult } from './types';
import { DEFAULT_TOGGLES } from './types';

const EMPTY_RESULT: VisionResult = {
  objects: [],
  poseLandmarks: [],
  poseActions: [],
  hands: [],
  fingerStates: [],
  staticGestures: [],
  motionGestures: [],
  faces: [],
  emotion: null,
  interactions: [],
  events: [],
  bodyLanguage: [],
  environment: 'Unknown',
  sceneDescription: 'Start the camera to begin analysis.',
  vlmDescription: '',
  fps: 0,
  backend: 'wasm',
};

type FrameLayers = {
  objects: VisionResult['objects'];
  poseLandmarks: VisionResult['poseLandmarks'];
  hands: VisionResult['hands'];
  faces: VisionResult['faces'];
  emotion: VisionResult['emotion'];
};

export class VisionPipeline {
  private yolo = new YoloDetector();
  private pose = new PoseDetector();
  private hands = new HandsDetector();
  private faceEmotion = new FaceEmotionDetector();
  private motionGestures = new MotionGestureDetector();
  private poseActions = new PoseActionClassifier();
  private vlm = new VlmSceneDescriber();

  private loopFrames = 0;
  private lastFpsTime = performance.now();
  private fps = 0;
  private running = false;
  private rafId = 0;
  private pipelineStart = 0;
  private lastModuleRun = {
    yolo: 0,
    pose: 0,
    hands: 0,
    face: 0,
    emotion: 0,
    vlm: 0,
  };
  private config: PipelineConfig = {
    performanceMode: 'balanced',
    toggles: { ...DEFAULT_TOGGLES },
    sampleIntervals: intervalsFromMode('balanced'),
  };

  private latest: VisionResult = { ...EMPTY_RESULT };
  private layers: FrameLayers = {
    objects: [],
    poseLandmarks: [],
    hands: [],
    faces: [],
    emotion: null,
  };
  private vlmDescription = '';
  private onUpdate: ((result: VisionResult) => void) | null = null;
  private onProgress: ((msg: string) => void) | null = null;
  private initialized = false;
  /** Skip YOLO/face/VLM only — hands + pose keep running (chat / Gemma). */
  private heavyPaused = false;
  private heavyBusy = false;
  private lastPublishAt = 0;
  private mediaPipeTs = 0;

  setConfig(config: Partial<PipelineConfig>): void {
    const nextMode = config.performanceMode ?? this.config.performanceMode;
    const modeChanged = config.performanceMode !== undefined
      && config.performanceMode !== this.config.performanceMode;

    this.config = {
      ...this.config,
      ...config,
      performanceMode: nextMode,
      toggles: { ...this.config.toggles, ...config.toggles },
      sampleIntervals: modeChanged && !config.sampleIntervals
        ? intervalsFromMode(nextMode)
        : {
          ...this.config.sampleIntervals,
          ...config.sampleIntervals,
        },
    };
  }

  getConfig(): PipelineConfig {
    return this.config;
  }

  getUiUpdateMs(): number {
    return resolveSchedule(this.config).uiUpdateMs;
  }

  getLastFrameAt(): number {
    return this.lastPublishAt;
  }

  setHeavyPaused(paused: boolean): void {
    this.heavyPaused = paused;
  }

  isHeavyPaused(): boolean {
    return this.heavyPaused;
  }

  setOnUpdate(cb: (result: VisionResult) => void): void {
    this.onUpdate = cb;
  }

  setOnProgress(cb: (msg: string) => void): void {
    this.onProgress = cb;
  }

  async init(): Promise<void> {
    if (this.initialized) return;
    const progress = (msg: string) => this.onProgress?.(msg);
    const { toggles } = this.config;

    if (toggles.yolo) await this.yolo.init(progress);
    if (toggles.pose) await this.pose.init(progress);
    if (toggles.hands) await this.hands.init(progress);
    if (toggles.face || toggles.emotion) await this.faceEmotion.init(progress);
    if (toggles.vlm) {
      void this.vlm.init(progress).catch(() => progress('VLM model unavailable'));
    }

    this.initialized = true;
    this.latest.backend = this.yolo.getBackend();
  }

  start(video: HTMLVideoElement): void {
    this.stop();
    this.running = true;
    this.loopFrames = 0;
    this.pipelineStart = performance.now();
    this.lastModuleRun = { yolo: 0, pose: 0, hands: 0, face: 0, emotion: 0, vlm: 0 };
    this.mediaPipeTs = 0;
    this.motionGestures.reset();
    this.poseActions.reset();
    this.layers = {
      objects: [],
      poseLandmarks: [],
      hands: [],
      faces: [],
      emotion: null,
    };

    const loop = (timestamp: number) => {
      if (!this.running) return;
      try {
        this.processFrame(video, timestamp);
      } catch (err) {
        console.warn('[VisionPipeline] frame error', err);
      }
      this.rafId = requestAnimationFrame(loop);
    };
    this.rafId = requestAnimationFrame(loop);
  }

  stop(): void {
    this.running = false;
    cancelAnimationFrame(this.rafId);
    this.heavyBusy = false;
  }

  getLatest(): VisionResult {
    return this.latest;
  }

  private nextMediaPipeTs(): number {
    this.mediaPipeTs += 33;
    return this.mediaPipeTs;
  }

  private processFrame(video: HTMLVideoElement, _timestamp: number): void {
    if (video.readyState < 2) return;

    this.loopFrames++;
    const now = performance.now();
    const schedule = resolveSchedule(this.config);
    const { toggles } = this.config;
    const ts = this.nextMediaPipeTs();

    let hands = this.layers.hands;
    let poseLandmarks = this.layers.poseLandmarks;

    try {
      if (
        toggles.hands
        && isModuleDue(now, this.pipelineStart, this.lastModuleRun.hands, schedule.hands)
      ) {
        hands = this.hands.detect(video, ts);
        this.lastModuleRun.hands = now;
        this.layers.hands = hands;
      }
    } catch (err) {
      console.warn('[VisionPipeline] hands detect failed', err);
    }

    try {
      if (
        toggles.pose
        && isModuleDue(now, this.pipelineStart, this.lastModuleRun.pose, schedule.pose)
      ) {
        poseLandmarks = this.pose.detect(video, ts);
        this.lastModuleRun.pose = now;
        this.layers.poseLandmarks = poseLandmarks;
      }
    } catch (err) {
      console.warn('[VisionPipeline] pose detect failed', err);
    }

    this.publishFrame(now, hands, poseLandmarks);

    if (!this.heavyPaused && !this.heavyBusy) {
      void this.runHeavyModule(video, now);
    }
  }

  private async runHeavyModule(video: HTMLVideoElement, now: number): Promise<void> {
    const schedule = resolveSchedule(this.config);
    const { toggles } = this.config;

    const faceOverdue = toggles.face
      ? getModuleOverdue(now, this.pipelineStart, this.lastModuleRun.face, schedule.face)
      : 0;
    const emotionOverdue = toggles.emotion
      ? getModuleOverdue(now, this.pipelineStart, this.lastModuleRun.emotion, schedule.emotion)
      : 0;
    const faceDue = faceOverdue > 0;
    const emotionDue = emotionOverdue > 0;

    let heavy = pickHeavyModule(now, this.pipelineStart, this.lastModuleRun, schedule, toggles);
    const faceStarved =
      (faceDue || emotionDue)
      && Math.max(faceOverdue, emotionOverdue)
        > Math.min(schedule.face.intervalMs, schedule.emotion.intervalMs) * 1.15;
    if (heavy === 'yolo' && faceStarved) {
      heavy = 'faceEmotion';
    }

    if (!heavy) return;
    if (heavy === 'faceEmotion' && !faceDue && !emotionDue) return;

    this.heavyBusy = true;
    try {
      if (heavy === 'yolo' && toggles.yolo) {
        const objects = await this.yolo.detect(video);
        this.layers.objects = objects;
        this.lastModuleRun.yolo = performance.now();
        this.publishFrame(performance.now(), this.layers.hands, this.layers.poseLandmarks);
      } else if (heavy === 'faceEmotion' && (faceDue || emotionDue)) {
        const result = await this.faceEmotion.detect(video, {
          face: toggles.face,
          emotion: toggles.emotion,
        });
        if (faceDue) {
          this.layers.faces = result.faces;
          this.lastModuleRun.face = performance.now();
        }
        if (emotionDue) {
          this.layers.emotion = result.emotion;
          this.lastModuleRun.emotion = performance.now();
        }
        this.publishFrame(performance.now(), this.layers.hands, this.layers.poseLandmarks);
      } else if (heavy === 'vlm' && toggles.vlm) {
        const vlmText = await this.vlm.describe(video, true);
        if (vlmText) this.vlmDescription = vlmText;
        this.lastModuleRun.vlm = performance.now();
        this.publishFrame(performance.now(), this.layers.hands, this.layers.poseLandmarks);
      }
    } catch (err) {
      console.warn('[VisionPipeline] heavy module failed', heavy, err);
    } finally {
      this.heavyBusy = false;
    }
  }

  private publishFrame(
    now: number,
    hands: VisionResult['hands'],
    poseLandmarks: VisionResult['poseLandmarks'],
  ): void {
    const { objects, faces, emotion } = this.layers;

    const fingerStates = hands.map((h) => ({
      hand: h.handedness,
      ...getFingerState(h),
    }));
    const staticGestures = recognizeStaticGestures(hands);
    const motionGestures = this.motionGestures.update(hands, Math.round(now));
    const poseActionList = this.poseActions.classify(poseLandmarks, Math.round(now));
    const interactions = analyzeInteractions(objects, hands, faces);
    const events = evaluateEvents(
      objects.map((o) => o.label),
      poseActionList,
      motionGestures,
      interactions,
      staticGestures,
    );
    const bodyLanguage = interpretBodyLanguage({
      staticGestures,
      motionGestures,
      poseActions: poseActionList,
      hands,
      poseLandmarks,
      faces,
    });
    const environment = classifyEnvironment(objects);

    const sceneDescription = buildRuleBasedDescription({
      objects,
      poseActions: poseActionList,
      staticGestures,
      motionGestures,
      interactions,
      events,
      environment,
      emotionDominant: emotion?.dominant,
      bodyLanguage,
    });

    if (now - this.lastFpsTime >= 1000) {
      this.fps = this.loopFrames;
      this.loopFrames = 0;
      this.lastFpsTime = now;
    }

    this.latest = {
      objects,
      poseLandmarks,
      poseActions: poseActionList,
      hands,
      fingerStates,
      staticGestures,
      motionGestures,
      faces,
      emotion,
      interactions,
      events,
      bodyLanguage,
      environment,
      sceneDescription,
      vlmDescription: this.vlmDescription,
      fps: this.fps,
      backend: this.yolo.getBackend(),
    };

    this.lastPublishAt = Date.now();
    this.onUpdate?.(this.latest);
  }

  dispose(): void {
    this.stop();
    this.yolo.dispose();
    this.pose.dispose();
    this.hands.dispose();
    this.faceEmotion.dispose();
    this.vlm.dispose();
    this.initialized = false;
  }
}
