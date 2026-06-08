declare module 'face-api.js' {
  export const tf: {
    ready(): Promise<void>;
    getBackend(): string;
    setBackend(name: string): Promise<boolean>;
  };

  export interface FaceDetection {
    box: { x: number; y: number; width: number; height: number };
  }

  export interface FaceLandmarks68 {
    getNose(): Array<{ x: number; y: number }>;
    getLeftEye(): Array<{ x: number; y: number }>;
    getRightEye(): Array<{ x: number; y: number }>;
  }

  export interface FaceExpressions {
    [key: string]: number;
  }

  export interface WithFaceLandmarks<T> {
    detection: FaceDetection;
    landmarks: FaceLandmarks68;
  }

  export interface WithAgeGender<T> extends WithFaceLandmarks<T> {
    age: number;
    gender: string;
  }

  export interface WithExpressions<T> extends WithFaceLandmarks<T> {
    expressions: FaceExpressions;
  }

  export class TinyFaceDetectorOptions {
    constructor(options?: { inputSize?: number; scoreThreshold?: number });
  }

  export const nets: {
    tinyFaceDetector: { loadFromUri(uri: string): Promise<void> };
    ageGenderNet: { loadFromUri(uri: string): Promise<void> };
    faceLandmark68Net: { loadFromUri(uri: string): Promise<void> };
    faceExpressionNet: { loadFromUri(uri: string): Promise<void> };
  };

  export type FaceDetectTask = {
    withFaceLandmarks(): {
      withAgeAndGender(): {
        withFaceExpressions(): Promise<Array<WithAgeGender<unknown> & WithExpressions<unknown>>>;
      };
      withFaceExpressions(): Promise<Array<WithAgeGender<unknown> & WithExpressions<unknown>>>;
    };
  };

  export function detectAllFaces(
    input: HTMLVideoElement | HTMLCanvasElement,
    options: TinyFaceDetectorOptions,
  ): FaceDetectTask & Promise<FaceDetection[]>;
}
