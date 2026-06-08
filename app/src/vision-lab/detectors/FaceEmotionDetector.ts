import type { DetectedFace, EmotionScores } from '../core/types';
import { bboxIoU } from '../utils/geometry';
import { createOffscreenCanvas, modelUrl } from '../utils/helpers';

type FaceApiModule = typeof import('face-api.js');

const EXPRESSION_MAP: Record<string, keyof Omit<EmotionScores, 'dominant' | 'dominantScore'>> = {
  happy: 'happy',
  sad: 'sad',
  angry: 'angry',
  surprised: 'surprised',
  fearful: 'fearful',
  neutral: 'neutral',
  disgusted: 'neutral',
};

interface FaceExpressionDetection {
  detection: { box: { x: number; y: number; width: number; height: number } };
  landmarks: {
    getNose(): Array<{ x: number; y: number }>;
    getLeftEye(): Array<{ x: number; y: number }>;
    getRightEye(): Array<{ x: number; y: number }>;
  };
  age?: number;
  gender?: string;
  expressions?: Record<string, number>;
}

let faceApiPromise: Promise<FaceApiModule> | null = null;

const loadFaceApi = (): Promise<FaceApiModule> => {
  if (!faceApiPromise) {
    faceApiPromise = import('face-api.js');
  }
  return faceApiPromise;
};

export class FaceEmotionDetector {
  private loaded = false;
  private nextId = 1;
  private tracks = new Map<number, DetectedFace>();
  private faceapi: FaceApiModule | null = null;

  /** face-api.js ships TF 1.x — init before any @tensorflow/tfjs 4.x backend. */
  private async ensureFaceTfBackend(faceapi: FaceApiModule): Promise<void> {
    await faceapi.tf.ready();
    if (faceapi.tf.getBackend()) return;
    const preferCpu =
      typeof navigator !== 'undefined' &&
      (navigator.webdriver ||
        !document.createElement('canvas').getContext('webgl'));
    if (preferCpu) {
      await faceapi.tf.setBackend('cpu');
    } else {
      try {
        await faceapi.tf.setBackend('webgl');
      } catch {
        await faceapi.tf.setBackend('cpu');
      }
    }
    await faceapi.tf.ready();
  }

  async init(onProgress?: (msg: string) => void): Promise<void> {
    if (this.loaded) return;
    onProgress?.('Loading face & emotion models...');
    const faceapi = await loadFaceApi();
    this.faceapi = faceapi;
    await this.ensureFaceTfBackend(faceapi);
    const base = modelUrl('models/face-api');
    await Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(base),
      faceapi.nets.ageGenderNet.loadFromUri(base),
      faceapi.nets.faceLandmark68Net.loadFromUri(base),
      faceapi.nets.faceExpressionNet.loadFromUri(base),
    ]);
    this.loaded = true;
    onProgress?.('Face & emotion models ready');
  }

  async detect(
    source: HTMLVideoElement,
    options: { face: boolean; emotion: boolean },
  ): Promise<{ faces: DetectedFace[]; emotion: EmotionScores | null }> {
    if (!this.loaded || (!options.face && !options.emotion)) {
      return { faces: [], emotion: null };
    }

    const faceapi = this.faceapi ?? (await loadFaceApi());
    this.faceapi = faceapi;

    const vw = source.videoWidth || source.clientWidth || 640;
    const vh = source.videoHeight || source.clientHeight || 480;
    if (vw < 64 || vh < 64) {
      return { faces: [], emotion: null };
    }

    const canvas = createOffscreenCanvas(source, 640);
    const detectPromise = faceapi
      .detectAllFaces(canvas, new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 }))
      .withFaceLandmarks()
      .withAgeAndGender()
      .withFaceExpressions();

    const detections = (await Promise.race([
      detectPromise,
      new Promise<never>((_, reject) => {
        setTimeout(() => reject(new Error('Face detect timeout (30s)')), 30_000);
      }),
    ])) as FaceExpressionDetection[];

    const width = canvas.width;
    const height = canvas.height;
    const faces: DetectedFace[] = [];

    if (options.face) {
      for (const det of detections) {
        const bbox = {
          x: det.detection.box.x / width,
          y: det.detection.box.y / height,
          width: det.detection.box.width / width,
          height: det.detection.box.height / height,
        };

        const landmarks = det.landmarks;
        const nose = landmarks.getNose()[3];
        const leftEye = landmarks.getLeftEye()[0];
        const rightEye = landmarks.getRightEye()[3];
        const eyeMidX = (leftEye.x + rightEye.x) / 2;
        const gazeX = nose.x - eyeMidX;
        let gazeDirection = 'Center';
        if (gazeX < -8) gazeDirection = 'Left';
        else if (gazeX > 8) gazeDirection = 'Right';

        const face: DetectedFace = {
          id: this.matchTrack(bbox),
          bbox,
          estimatedAge: Math.round(det.age ?? 0),
          estimatedGender: det.gender === 'male' ? 'Male' : 'Female',
          gazeDirection,
        };
        faces.push(face);
        this.tracks.set(face.id, face);
      }
    }

    let emotion: EmotionScores | null = null;
    if (options.emotion && detections.length) {
      const scores: EmotionScores = {
        happy: 0,
        sad: 0,
        angry: 0,
        surprised: 0,
        fearful: 0,
        neutral: 0,
        dominant: 'neutral',
        dominantScore: 0,
      };

      for (const det of detections) {
        if (!det.expressions) continue;
        for (const [expr, prob] of Object.entries(det.expressions)) {
          const key = EXPRESSION_MAP[expr];
          if (key) scores[key] += prob;
        }
      }

      const count = detections.length;
      for (const key of ['happy', 'sad', 'angry', 'surprised', 'fearful', 'neutral'] as const) {
        scores[key] /= count;
      }

      let dominant: keyof typeof scores = 'neutral';
      let dominantScore = 0;
      for (const key of ['happy', 'sad', 'angry', 'surprised', 'fearful', 'neutral'] as const) {
        if (scores[key] > dominantScore) {
          dominantScore = scores[key];
          dominant = key;
        }
      }

      scores.dominant = dominant.charAt(0).toUpperCase() + dominant.slice(1);
      scores.dominantScore = dominantScore;
      emotion = scores;
    }

    return { faces, emotion };
  }

  private matchTrack(bbox: DetectedFace['bbox']): number {
    let bestId = this.nextId++;
    let bestIoU = 0;
    for (const [id, track] of this.tracks) {
      const iou = bboxIoU(bbox, track.bbox);
      if (iou > bestIoU && iou > 0.3) {
        bestIoU = iou;
        bestId = id;
      }
    }
    return bestId;
  }

  dispose(): void {
    this.loaded = false;
    this.faceapi = null;
    this.tracks.clear();
  }
}
