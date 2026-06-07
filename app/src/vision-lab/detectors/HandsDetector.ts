import {
  FilesetResolver,
  HandLandmarker,
  type HandLandmarkerResult,
} from '@mediapipe/tasks-vision';
import type { DetectedHand } from '../core/types';
import { bboxFromPoints } from '../utils/geometry';
import { pickMediapipeDelegate } from '../utils/mediapipeDelegate';

const HAND_MODEL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

export class HandsDetector {
  private landmarker: HandLandmarker | null = null;

  async init(onProgress?: (msg: string) => void): Promise<void> {
    onProgress?.('Loading MediaPipe Hands...');
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm',
    );
    const delegate = await pickMediapipeDelegate();
    try {
      this.landmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: HAND_MODEL, delegate },
        runningMode: 'VIDEO',
        numHands: 2,
      });
    } catch (err) {
      console.warn('[GROVEE hands] delegate failed, retrying CPU', err);
      this.landmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: HAND_MODEL, delegate: 'CPU' },
        runningMode: 'VIDEO',
        numHands: 2,
      });
    }
    onProgress?.(`Hands model ready (${delegate})`);
  }

  detect(source: HTMLVideoElement, timestampMs: number): DetectedHand[] {
    if (!this.landmarker) return [];
    const result: HandLandmarkerResult = this.landmarker.detectForVideo(source, timestampMs);
    return result.landmarks.map((landmarks, i) => {
      const normalized = landmarks.map((lm) => ({
        x: lm.x,
        y: lm.y,
        z: lm.z,
      }));
      const handedness = result.handednesses[i]?.[0]?.categoryName === 'Left' ? 'Left' : 'Right';
      return {
        handedness,
        landmarks: normalized,
        bbox: bboxFromPoints(normalized),
      };
    });
  }

  dispose(): void {
    this.landmarker?.close();
    this.landmarker = null;
  }
}
