import * as faceapi from 'face-api.js';
import type { DetectedFace, EmotionScores } from '../core/types';
import { bboxIoU } from '../utils/geometry';
import { createOffscreenCanvas, modelUrl } from '../utils/helpers';

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
  detection: faceapi.FaceDetection;
  landmarks: faceapi.FaceLandmarks68;
  age?: number;
  gender?: string;
  expressions?: faceapi.FaceExpressions;
}

export class FaceEmotionDetector {
  private loaded = false;
  private loadFailed = false;
  private nextId = 1;
  private tracks = new Map<number, DetectedFace>();
  private lastError = '';

  async init(onProgress?: (msg: string) => void): Promise<void> {
    if (this.loaded || this.loadFailed) return;
    onProgress?.('Loading face & emotion models...');
    const base = modelUrl('models/face-api');

    try {
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri(base),
        faceapi.nets.ageGenderNet.loadFromUri(base),
        faceapi.nets.faceLandmark68Net.loadFromUri(base),
        faceapi.nets.faceExpressionNet.loadFromUri(base),
      ]);
      this.loaded = true;
      this.lastError = '';
      onProgress?.('Face & emotion models ready');
    } catch (err) {
      this.loadFailed = true;
      this.lastError = err instanceof Error ? err.message : String(err);
      console.error('[FaceEmotionDetector] model load failed', err);
      onProgress?.(`Face models failed: ${this.lastError}`);
    }
  }

  isReady(): boolean {
    return this.loaded;
  }

  getLastError(): string {
    return this.lastError;
  }

  async detect(
    source: HTMLVideoElement,
    options: { face: boolean; emotion: boolean },
  ): Promise<{ faces: DetectedFace[]; emotion: EmotionScores | null }> {
    if (!this.loaded || this.loadFailed || (!options.face && !options.emotion)) {
      return { faces: [], emotion: null };
    }

    const vw = source.videoWidth;
    const vh = source.videoHeight;
    if (!vw || !vh) {
      return { faces: [], emotion: null };
    }

    try {
      const canvas = createOffscreenCanvas(source, 640);
      const cw = canvas.width;
      const ch = canvas.height;

      const detections = (await faceapi
        .detectAllFaces(
          canvas as unknown as HTMLVideoElement,
          new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: 0.25 }),
        )
        .withFaceLandmarks()
        .withAgeAndGender()
        .withFaceExpressions()) as FaceExpressionDetection[];

      const faces: DetectedFace[] = [];

      if (options.face) {
        for (const det of detections) {
          const box = det.detection.box;
          const bbox = {
            x: box.x / cw,
            y: box.y / ch,
            width: box.width / cw,
            height: box.height / ch,
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

      this.lastError = '';
      return { faces, emotion };
    } catch (err) {
      this.lastError = err instanceof Error ? err.message : String(err);
      console.warn('[FaceEmotionDetector] detect failed', err);
      return { faces: [], emotion: null };
    }
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
    this.loadFailed = false;
    this.tracks.clear();
    this.lastError = '';
  }
}
