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
  intervalsFromMode,
  isModuleDue,
  resolveSchedule,
} from './schedule';
import type { FaceModuleDiagnostics, PipelineConfig, VisionResult } from './types';
import { DEFAULT_FACE_MODULE, DEFAULT_TOGGLES } from './types';

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
  faceModule: { ...DEFAULT_FACE_MODULE },
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
  private yoloBusy = false;
  private faceBusy = false;
  private faceModule: FaceModuleDiagnostics = { ...DEFAULT_FACE_MODULE };
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

    if (toggles.face || toggles.emotion) {
      this.faceModule = {
        status: 'loading',
        message: 'Loading face & emotion models…',
        lastScanAt: 0,
        lastFaceCount: 0,
        modelSource: 'local',
      };
      try {
        await this.faceEmotion.init(progress);
        this.faceModule = {
          status: 'ready',
          message: 'Face & emotion models ready — scanning',
          lastScanAt: 0,
          lastFaceCount: 0,
          modelSource: 'local',
        };
      } catch (err) {
        console.warn('[VisionPipeline] face/emotion init failed', err);
        this.faceModule = {
          status: 'error',
          message: err instanceof Error ? err.message : 'Face model load failed',
          lastScanAt: 0,
          lastFaceCount: 0,
          modelSource: 'local',
        };
      }
    } else {
      this.faceModule = {
        status: 'disabled',
        message: 'Face / emotion toggles off',
        lastScanAt: 0,
        lastFaceCount: 0,
        modelSource: 'none',
      };
    }

    if (toggles.yolo) await this.yolo.init(progress);
    if (toggles.pose) await this.pose.init(progress);
    if (toggles.hands) await this.hands.init(progress);
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
    this.yoloBusy = false;
    this.faceBusy = false;
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

    // Face + emotion — own lane, never blocked by YOLO or Gemma pause.
    if (!this.faceBusy && (toggles.face || toggles.emotion)) {
      const faceDue = toggles.face
        && isModuleDue(now, this.pipelineStart, this.lastModuleRun.face, schedule.face);
      const emotionDue = toggles.emotion
        && isModuleDue(now, this.pipelineStart, this.lastModuleRun.emotion, schedule.emotion);
      if (faceDue || emotionDue) {
        void this.runFaceModule(video, faceDue, emotionDue);
      }
    }

    // YOLO / VLM — never auto-paused; user toggles only.
    if (!this.yoloBusy) {
      if (toggles.yolo && isModuleDue(now, this.pipelineStart, this.lastModuleRun.yolo, schedule.yolo)) {
        void this.runYoloModule(video);
      } else if (
        toggles.vlm
        && isModuleDue(now, this.pipelineStart, this.lastModuleRun.vlm, schedule.vlm)
      ) {
        void this.runVlmModule(video);
      }
    }
  }

  private async runFaceModule(
    video: HTMLVideoElement,
    updateFace: boolean,
    updateEmotion: boolean,
  ): Promise<void> {
    const { toggles } = this.config;
    this.faceBusy = true;
    this.faceModule = {
      ...this.faceModule,
      status: 'scanning',
      message: 'Scanning face & expression…',
    };
    try {
      const result = await this.faceEmotion.detect(video, {
        face: toggles.face,
        emotion: toggles.emotion,
      });
      const ts = performance.now();
      if (toggles.face) {
        this.layers.faces = result.faces;
        if (updateFace) this.lastModuleRun.face = ts;
      }
      if (toggles.emotion) {
        this.layers.emotion = result.emotion;
        if (updateEmotion) this.lastModuleRun.emotion = ts;
      }
      const count = result.faces.length;
      this.faceModule = {
        status: 'ready',
        message: count
          ? `${count} face(s) · ${result.emotion?.dominant ?? 'no emotion'}`
          : 'No face in frame',
        lastScanAt: Date.now(),
        lastFaceCount: count,
        modelSource: 'local',
      };
      this.publishFrame(ts, this.layers.hands, this.layers.poseLandmarks);
    } catch (err) {
      console.warn('[VisionPipeline] face/emotion detect failed', err);
      this.faceModule = {
        status: 'error',
        message: err instanceof Error ? err.message : 'Face detect failed',
        lastScanAt: Date.now(),
        lastFaceCount: this.layers.faces.length,
        modelSource: 'local',
      };
    } finally {
      this.faceBusy = false;
    }
  }

  private async runYoloModule(video: HTMLVideoElement): Promise<void> {
    this.yoloBusy = true;
    try {
      const objects = await this.yolo.detect(video);
      this.layers.objects = objects;
      const ts = performance.now();
      this.lastModuleRun.yolo = ts;
      this.publishFrame(ts, this.layers.hands, this.layers.poseLandmarks);
    } catch (err) {
      console.warn('[VisionPipeline] YOLO detect failed', err);
    } finally {
      this.yoloBusy = false;
    }
  }

  private async runVlmModule(video: HTMLVideoElement): Promise<void> {
    this.yoloBusy = true;
    try {
      const vlmText = await this.vlm.describe(video, true);
      if (vlmText) this.vlmDescription = vlmText;
      this.lastModuleRun.vlm = performance.now();
      this.publishFrame(performance.now(), this.layers.hands, this.layers.poseLandmarks);
    } catch (err) {
      console.warn('[VisionPipeline] VLM describe failed', err);
    } finally {
      this.yoloBusy = false;
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
      faceModule: { ...this.faceModule },
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
