import * as faceapi from 'face-api.js';
import type { DetectedFace, EmotionScores } from '../core/types';
import { bboxIoU } from '../utils/geometry';
import { modelUrl } from '../utils/helpers';

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
  private nextId = 1;
  private tracks = new Map<number, DetectedFace>();

  async init(onProgress?: (msg: string) => void): Promise<void> {
    if (this.loaded) return;
    onProgress?.('Loading face & emotion models...');
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

    const detections = (await faceapi
      .detectAllFaces(source, new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 }))
      .withFaceLandmarks()
      .withAgeAndGender()
      .withFaceExpressions()) as FaceExpressionDetection[];

    const width = source.videoWidth;
    const height = source.videoHeight;
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
    this.tracks.clear();
  }
}
