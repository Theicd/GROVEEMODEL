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
  private vlmDescription = '';
  private onUpdate: ((result: VisionResult) => void) | null = null;
  private onProgress: ((msg: string) => void) | null = null;
  private initialized = false;

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
    if (this.running) return;
    this.running = true;
    this.loopFrames = 0;
    this.pipelineStart = performance.now();
    this.lastModuleRun = { yolo: 0, pose: 0, hands: 0, face: 0, emotion: 0, vlm: 0 };
    this.motionGestures.reset();
    this.poseActions.reset();
    const loop = async (timestamp: number) => {
      if (!this.running) return;
      await this.processFrame(video, timestamp);
      this.rafId = requestAnimationFrame(loop);
    };
    this.rafId = requestAnimationFrame(loop);
  }

  stop(): void {
    this.running = false;
    cancelAnimationFrame(this.rafId);
  }

  getLatest(): VisionResult {
    return this.latest;
  }

  private async processFrame(video: HTMLVideoElement, timestamp: number): Promise<void> {
    if (video.readyState < 2) return;

    this.loopFrames++;
    const now = performance.now();
    const schedule = resolveSchedule(this.config);
    const { toggles } = this.config;
    const ts = Math.round(timestamp);

    let objects = this.latest.objects;
    let poseLandmarks = this.latest.poseLandmarks;
    let hands = this.latest.hands;
    let faces = this.latest.faces;
    let emotion = this.latest.emotion;

    // Light modules — fast sync inference, may run together without heavy burst impact.
    if (
      toggles.hands
      && isModuleDue(now, this.pipelineStart, this.lastModuleRun.hands, schedule.hands)
    ) {
      hands = this.hands.detect(video, ts);
      this.lastModuleRun.hands = now;
    }

    if (
      toggles.pose
      && isModuleDue(now, this.pipelineStart, this.lastModuleRun.pose, schedule.pose)
    ) {
      poseLandmarks = this.pose.detect(video, ts);
      this.lastModuleRun.pose = now;
    }

    const faceDue = toggles.face
      && getModuleOverdue(now, this.pipelineStart, this.lastModuleRun.face, schedule.face) > 0;
    const emotionDue = toggles.emotion
      && getModuleOverdue(now, this.pipelineStart, this.lastModuleRun.emotion, schedule.emotion) > 0;

    // Heavy modules — at most one async inference per frame to spread CPU/GPU load.
    const heavy = pickHeavyModule(now, this.pipelineStart, this.lastModuleRun, schedule, toggles);

    if (heavy === 'yolo') {
      objects = await this.yolo.detect(video);
      this.lastModuleRun.yolo = now;
    } else if (heavy === 'faceEmotion' && (faceDue || emotionDue)) {
      const result = await this.faceEmotion.detect(video, {
        face: toggles.face,
        emotion: toggles.emotion,
      });
      if (faceDue) {
        faces = result.faces;
        this.lastModuleRun.face = now;
      }
      if (emotionDue) {
        emotion = result.emotion;
        this.lastModuleRun.emotion = now;
      }
    } else if (heavy === 'vlm') {
      const vlmText = await this.vlm.describe(video, true);
      if (vlmText) this.vlmDescription = vlmText;
      this.lastModuleRun.vlm = now;
    }

    const fingerStates = hands.map((h) => ({
      hand: h.handedness,
      ...getFingerState(h),
    }));
    const staticGestures = recognizeStaticGestures(hands);
    const motionGestures = this.motionGestures.update(hands, ts);
    const poseActionList = this.poseActions.classify(poseLandmarks, ts);
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
