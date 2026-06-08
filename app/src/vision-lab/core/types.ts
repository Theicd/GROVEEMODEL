export interface Point2D {
  x: number;
  y: number;
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface DetectedObject {
  label: string;
  displayLabel: string;
  confidence: number;
  bbox: BoundingBox;
}

export interface PoseLandmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

export interface HandLandmark {
  x: number;
  y: number;
  z: number;
}

export interface DetectedHand {
  handedness: 'Left' | 'Right';
  landmarks: HandLandmark[];
  bbox: BoundingBox;
}

export interface FingerState {
  thumb: 'Open' | 'Closed';
  index: 'Open' | 'Closed';
  middle: 'Open' | 'Closed';
  ring: 'Open' | 'Closed';
  pinky: 'Open' | 'Closed';
}

export interface StaticGesture {
  name: string;
  confidence: number;
  hand: 'Left' | 'Right';
}

export interface MotionGesture {
  name: string;
  confidence: number;
}

export interface PoseAction {
  name: string;
  confidence: number;
}

export interface DetectedFace {
  id: number;
  bbox: BoundingBox;
  estimatedAge: number;
  estimatedGender: string;
  gazeDirection: string;
}

export interface EmotionScores {
  happy: number;
  sad: number;
  angry: number;
  surprised: number;
  fearful: number;
  neutral: number;
  dominant: string;
  dominantScore: number;
}

export interface Interaction {
  name: string;
  confidence: number;
}

export interface DetectedEvent {
  name: string;
  confidence: number;
}

/** Semantic interpretation of gestures, posture, and self-touch cues. */
export interface BodyLanguageCue {
  signal: string;
  meaning: string;
  category: 'sign' | 'motion' | 'posture' | 'self-touch';
  confidence: number;
}

export type EnvironmentType =
  | 'Office'
  | 'Kitchen'
  | 'Living Room'
  | 'Bedroom'
  | 'Classroom'
  | 'Vehicle'
  | 'Unknown';

export type FaceModuleStatus = 'idle' | 'loading' | 'ready' | 'error' | 'scanning' | 'disabled';

export interface FaceModuleDiagnostics {
  status: FaceModuleStatus;
  message: string;
  lastScanAt: number;
  lastFaceCount: number;
  modelSource: 'local' | 'cdn' | 'none';
}

export const DEFAULT_FACE_MODULE: FaceModuleDiagnostics = {
  status: 'idle',
  message: 'Face module not started',
  lastScanAt: 0,
  lastFaceCount: 0,
  modelSource: 'none',
};

export interface VisionResult {
  objects: DetectedObject[];
  poseLandmarks: PoseLandmark[];
  poseActions: PoseAction[];
  hands: DetectedHand[];
  fingerStates: Array<{ hand: 'Left' | 'Right'; fingers: FingerState; count: number }>;
  staticGestures: StaticGesture[];
  motionGestures: MotionGesture[];
  faces: DetectedFace[];
  emotion: EmotionScores | null;
  faceModule: FaceModuleDiagnostics;
  interactions: Interaction[];
  events: DetectedEvent[];
  bodyLanguage: BodyLanguageCue[];
  environment: EnvironmentType;
  sceneDescription: string;
  vlmDescription: string;
  fps: number;
  backend: 'webgpu' | 'wasm';
}

export type PerformanceMode = 'lite' | 'balanced' | 'full';

export interface ModelToggles {
  yolo: boolean;
  pose: boolean;
  hands: boolean;
  face: boolean;
  emotion: boolean;
  vlm: boolean;
}

export interface PipelineConfig {
  performanceMode: PerformanceMode;
  toggles: ModelToggles;
  sampleIntervals: SampleIntervals;
}

export interface SampleIntervals {
  yolo: number;
  pose: number;
  hands: number;
  face: number;
  emotion: number;
  vlm: number;
  uiUpdate: number;
}

export const DEFAULT_TOGGLES: ModelToggles = {
  yolo: true,
  pose: true,
  hands: true,
  face: true,
  emotion: true,
  vlm: true,
};

export const POSE_CONNECTIONS: Array<[number, number]> = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24], [23, 25], [25, 27],
  [24, 26], [26, 28], [15, 17], [15, 19], [15, 21],
  [16, 18], [16, 20], [16, 22], [27, 29], [27, 31],
  [28, 30], [28, 32],
];

export const HAND_CONNECTIONS: Array<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15],
  [15, 16], [0, 17], [17, 18], [18, 19], [19, 20], [5, 9], [9, 13], [13, 17],
];

export const TARGET_COCO_LABELS = new Set([
  'person', 'cup', 'bottle', 'wine glass', 'cell phone', 'backpack',
  'handbag', 'laptop', 'chair', 'car', 'cat', 'dog', 'bird', 'horse',
  'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'tv', 'bed',
  'dining table', 'couch', 'keyboard', 'mouse', 'book', 'clock',
  'refrigerator', 'oven', 'sink', 'potted plant', 'tie', 'suitcase',
]);

export const LABEL_DISPLAY: Record<string, string> = {
  person: 'Person',
  cup: 'Coffee Cup',
  bottle: 'Bottle',
  'wine glass': 'Glass',
  'cell phone': 'Phone',
  backpack: 'Backpack',
  handbag: 'Bag',
  laptop: 'Laptop',
  chair: 'Chair',
  car: 'Car',
  cat: 'Cat',
  dog: 'Dog',
  bird: 'Bird',
  horse: 'Horse',
  tv: 'TV',
  bed: 'Bed',
  'dining table': 'Table',
  couch: 'Sofa',
  keyboard: 'Keyboard',
  mouse: 'Mouse',
};
