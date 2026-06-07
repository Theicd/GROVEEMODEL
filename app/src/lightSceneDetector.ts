/**
 * Stage 2 — lightweight browser object detection (COCO-SSD).
 * Updates world state; does not run Gemma.
 */

import * as cocoSsd from "@tensorflow-models/coco-ssd";
import { ensureTfBackend } from "./browserVision";

export const LIGHT_DETECTOR_CONFIG = {
  minObjectScore: 0.5,
  minPersonScore: 0.42,
  throttleMs: 4_000,
  /** Faster COCO pass when wave/arm/high motion detected. */
  urgentThrottleMs: 1_500,
  detectMaxWidth: 416,
} as const;

export type DetectionBox = {
  label: string;
  score: number;
  x: number;
  y: number;
  width: number;
  height: number;
};

export type LightDetectionResult = {
  objects: string[];
  people: string[];
  personCount: number;
  labels: string[];
  boxes: DetectionBox[];
  personBoxes: DetectionBox[];
};

/** Map COCO class names to short world-memory labels. */
const COCO_LABEL_MAP: Record<string, string> = {
  person: "person",
  "cell phone": "phone",
  laptop: "laptop",
  tv: "television",
  chair: "chair",
  couch: "couch",
  bottle: "bottle",
  cup: "cup",
  book: "book",
  clock: "clock",
  keyboard: "keyboard",
  mouse: "mouse",
  "dining table": "table",
  "potted plant": "plant",
  backpack: "backpack",
  handbag: "bag",
  umbrella: "umbrella",
  bicycle: "bicycle",
  dog: "dog",
  cat: "cat",
};

const normalizeCocoClass = (className: string): string => {
  const key = className.trim().toLowerCase();
  return COCO_LABEL_MAP[key] ?? key.replace(/\s+/g, "_");
};

let modelPromise: Promise<cocoSsd.ObjectDetection> | null = null;

export const preloadLightDetector = (): Promise<cocoSsd.ObjectDetection> => {
  if (!modelPromise) {
    modelPromise = ensureTfBackend(true).then(() => cocoSsd.load({ base: "lite_mobilenet_v2" }));
  }
  return modelPromise;
};

export const isLightDetectorReady = (): boolean => modelPromise !== null;

const captureDetectCanvas = (
  video: HTMLVideoElement,
  maxWidth: number,
): { canvas: HTMLCanvasElement; scale: number } | null => {
  if (video.readyState < 2 || video.videoWidth <= 0) return null;
  const scale = Math.min(1, maxWidth / Math.max(video.videoWidth, video.videoHeight));
  const w = Math.max(1, Math.round(video.videoWidth * scale));
  const h = Math.max(1, Math.round(video.videoHeight * scale));
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.drawImage(video, 0, 0, w, h);
  return { canvas, scale };
};

export const detectLightScene = async (
  video: HTMLVideoElement,
): Promise<LightDetectionResult | null> => {
  const capture = captureDetectCanvas(video, LIGHT_DETECTOR_CONFIG.detectMaxWidth);
  if (!capture) return null;

  const model = await preloadLightDetector();
  const predictions = await model.detect(capture.canvas);

  const objectSet = new Set<string>();
  let personCount = 0;
  const boxes: DetectionBox[] = [];
  const personBoxes: DetectionBox[] = [];

  for (const pred of predictions) {
    const label = normalizeCocoClass(pred.class);
    const [x, y, width, height] = pred.bbox;
    const box: DetectionBox = { label, score: pred.score, x, y, width, height };
    boxes.push(box);
    if (label === "person") {
      if (pred.score >= LIGHT_DETECTOR_CONFIG.minPersonScore) {
        personCount++;
        personBoxes.push(box);
      }
      continue;
    }
    if (pred.score >= LIGHT_DETECTOR_CONFIG.minObjectScore) {
      objectSet.add(label);
    }
  }

  const objects = [...objectSet].sort();
  const people = personCount > 0 ? (personCount === 1 ? ["person"] : [`person x${personCount}`]) : [];

  return {
    objects,
    people,
    personCount,
    labels: [...objects, ...people],
    boxes,
    personBoxes,
  };
};

export class LightSceneDetector {
  private lastRunAt = 0;
  private running = false;
  /** Debounce: consecutive detections with person before confirming. */
  private personStreak = 0;
  private noPersonStreak = 0;
  private confirmedPerson = false;

  reset(): void {
    this.lastRunAt = 0;
    this.running = false;
    this.personStreak = 0;
    this.noPersonStreak = 0;
    this.confirmedPerson = false;
  }

  /** Apply debounce — returns stable person presence for world memory. */
  debouncedPeople(raw: LightDetectionResult): {
    people: string[];
    justConfirmed: boolean;
    justLeft: boolean;
  } {
    const wasConfirmed = this.confirmedPerson;

    if (raw.personCount > 0) {
      this.personStreak++;
      this.noPersonStreak = 0;
    } else {
      this.noPersonStreak++;
      this.personStreak = 0;
    }

    if (!this.confirmedPerson && this.personStreak >= 2) {
      this.confirmedPerson = true;
    }
    if (this.confirmedPerson && this.noPersonStreak >= 2) {
      this.confirmedPerson = false;
    }

    const people = this.confirmedPerson
      ? raw.personCount >= 1
        ? raw.personCount === 1
          ? ["person"]
          : [`person x${raw.personCount}`]
        : ["person"]
      : [];

    return {
      people,
      justConfirmed: !wasConfirmed && this.confirmedPerson,
      justLeft: wasConfirmed && !this.confirmedPerson,
    };
  }

  canRun(now = Date.now(), options?: { urgent?: boolean; force?: boolean }): boolean {
    if (options?.force) return true;
    const throttle = options?.urgent
      ? LIGHT_DETECTOR_CONFIG.urgentThrottleMs
      : LIGHT_DETECTOR_CONFIG.throttleMs;
    return now - this.lastRunAt >= throttle;
  }

  async run(
    video: HTMLVideoElement,
    options?: { force?: boolean },
  ): Promise<
    (LightDetectionResult & {
      debouncedPeople: string[];
      personJustConfirmed: boolean;
      personJustLeft: boolean;
    }) | null
  > {
    if (this.running) return null;
    const now = Date.now();
    if (!this.canRun(now, { force: options?.force })) return null;

    this.running = true;
    this.lastRunAt = now;
    try {
      const raw = await detectLightScene(video);
      if (!raw) return null;
      const debounce = this.debouncedPeople(raw);
      return {
        ...raw,
        debouncedPeople: debounce.people,
        personJustConfirmed: debounce.justConfirmed,
        personJustLeft: debounce.justLeft,
      };
    } finally {
      this.running = false;
    }
  }
}
