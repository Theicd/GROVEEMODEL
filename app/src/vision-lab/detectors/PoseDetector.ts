import {
  FilesetResolver,
  PoseLandmarker,
  type PoseLandmarkerResult,
} from '@mediapipe/tasks-vision';
import type { PoseLandmark } from '../core/types';
import { pickMediapipeDelegate } from '../utils/mediapipeDelegate';

const POSE_MODEL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

export class PoseDetector {
  private landmarker: PoseLandmarker | null = null;

  async init(onProgress?: (msg: string) => void): Promise<void> {
    onProgress?.('Loading MediaPipe Pose...');
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm',
    );
    const delegate = await pickMediapipeDelegate();
    try {
      this.landmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: POSE_MODEL, delegate },
        runningMode: 'VIDEO',
        numPoses: 2,
      });
    } catch (err) {
      console.warn('[GROVEE pose] delegate failed, retrying CPU', err);
      this.landmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: POSE_MODEL, delegate: 'CPU' },
        runningMode: 'VIDEO',
        numPoses: 2,
      });
    }
    onProgress?.(`Pose model ready (${delegate})`);
  }

  detect(source: HTMLVideoElement, timestampMs: number): PoseLandmark[] {
    if (!this.landmarker) return [];
    const result: PoseLandmarkerResult = this.landmarker.detectForVideo(source, timestampMs);
    if (!result.landmarks.length) return [];

    return result.landmarks[0].map((lm) => ({
      x: lm.x,
      y: lm.y,
      z: lm.z,
      visibility: lm.visibility,
    }));
  }

  dispose(): void {
    this.landmarker?.close();
    this.landmarker = null;
  }
}
