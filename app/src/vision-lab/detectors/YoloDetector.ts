import * as ort from 'onnxruntime-web';
import {
  FilesetResolver,
  ObjectDetector,
  type ObjectDetectorResult,
} from '@mediapipe/tasks-vision';
import type { DetectedObject } from '../core/types';
import { LABEL_DISPLAY, TARGET_COCO_LABELS } from '../core/types';
import { modelUrl } from '../utils/helpers';

const COCO_LABELS = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
  'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
  'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
  'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
  'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
  'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
  'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
  'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
  'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
];

const MP_OBJECT_MODEL =
  'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite';

const YOLO_MODEL_CANDIDATES = [
  'models/yolo11n.onnx',
  'models/yolov8n.onnx',
];

type DetectorMode = 'yolo' | 'mediapipe';

export class YoloDetector {
  private session: ort.InferenceSession | null = null;
  private mpDetector: ObjectDetector | null = null;
  private mode: DetectorMode = 'yolo';
  private modelName = 'yolo11n';
  private inputSize = 640;
  private backend: 'webgpu' | 'wasm' = 'wasm';
  private timestamp = 0;

  async init(onProgress?: (msg: string) => void): Promise<void> {
    // Keep YOLO on WASM so ORT WebGPU stays free for Gemma (shared GPU / EP conflicts).
    this.backend = 'wasm';

    try {
      onProgress?.('Loading YOLO11n ONNX...');
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.0/dist/';
      const providers = ['wasm'];
      let loaded = false;

      for (const relativePath of YOLO_MODEL_CANDIDATES) {
        try {
          this.session = await ort.InferenceSession.create(modelUrl(relativePath), {
            executionProviders: providers,
            graphOptimizationLevel: 'all',
          });
          this.modelName = relativePath.includes('yolo11') ? 'yolo11n' : 'yolov8n';
          loaded = true;
          break;
        } catch {
          // try next candidate
        }
      }

      if (!loaded || !this.session) {
        throw new Error('No YOLO ONNX model available');
      }

      this.mode = 'yolo';
      onProgress?.(`YOLO11 ready (${this.modelName}, ${this.backend})`);
    } catch {
      onProgress?.('YOLO ONNX unavailable — using MediaPipe Object Detector...');
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm',
      );
      this.mpDetector = await ObjectDetector.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MP_OBJECT_MODEL, delegate: 'GPU' },
        runningMode: 'VIDEO',
        scoreThreshold: 0.35,
        maxResults: 20,
      });
      this.mode = 'mediapipe';
      onProgress?.('MediaPipe object detector ready');
    }
  }

  getBackend(): 'webgpu' | 'wasm' {
    return this.backend;
  }

  getMode(): DetectorMode {
    return this.mode;
  }

  getModelName(): string {
    return this.modelName;
  }

  async detect(source: HTMLVideoElement | HTMLCanvasElement): Promise<DetectedObject[]> {
    if (this.mode === 'mediapipe' && this.mpDetector && source instanceof HTMLVideoElement) {
      return this.detectMediaPipe(source);
    }
    if (!this.session) return [];
    return this.detectYolo(source);
  }

  private detectMediaPipe(source: HTMLVideoElement): DetectedObject[] {
    if (!this.mpDetector) return [];
    this.timestamp += 33;
    const result: ObjectDetectorResult = this.mpDetector.detectForVideo(source, this.timestamp);
    const width = source.videoWidth || 1;
    const height = source.videoHeight || 1;

    return result.detections
      .map((det) => {
        const category = det.categories[0];
        if (!category) return null;
        const label = category.categoryName.toLowerCase();
        if (!TARGET_COCO_LABELS.has(label)) return null;
        const box = det.boundingBox;
        if (!box) return null;
        return {
          label,
          displayLabel: LABEL_DISPLAY[label] ?? category.displayName ?? label,
          confidence: category.score,
          bbox: {
            x: box.originX / width,
            y: box.originY / height,
            width: box.width / width,
            height: box.height / height,
          },
        } satisfies DetectedObject;
      })
      .filter((item): item is DetectedObject => item !== null);
  }

  private async detectYolo(source: HTMLVideoElement | HTMLCanvasElement): Promise<DetectedObject[]> {
    if (!this.session) return [];

    try {
    const canvas = document.createElement('canvas');
    const width = source instanceof HTMLVideoElement ? source.videoWidth : source.width;
    const height = source instanceof HTMLVideoElement ? source.videoHeight : source.height;
    canvas.width = this.inputSize;
    canvas.height = this.inputSize;
    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(source, 0, 0, this.inputSize, this.inputSize);

    const imageData = ctx.getImageData(0, 0, this.inputSize, this.inputSize);
    const input = new Float32Array(3 * this.inputSize * this.inputSize);
    for (let i = 0; i < imageData.data.length; i += 4) {
      const idx = i / 4;
      input[idx] = imageData.data[i] / 255;
      input[idx + this.inputSize * this.inputSize] = imageData.data[i + 1] / 255;
      input[idx + 2 * this.inputSize * this.inputSize] = imageData.data[i + 2] / 255;
    }

    const tensor = new ort.Tensor('float32', input, [1, 3, this.inputSize, this.inputSize]);
    const inputName = this.session.inputNames[0] ?? 'images';
    const outputs = await this.session.run({ [inputName]: tensor });
    const output = outputs[Object.keys(outputs)[0]] as ort.Tensor;
    return this.parseOutput(output.data as Float32Array, output.dims, width, height);
    } catch {
      return [];
    }
  }

  private parseOutput(
    data: Float32Array,
    dims: readonly number[],
    srcW: number,
    srcH: number,
  ): DetectedObject[] {
    const numClasses = COCO_LABELS.length;
    const channels = 4 + numClasses;

    let channelMajor = true;
    if (dims.length === 3) {
      channelMajor = dims[1] === channels;
    }

    const results = channelMajor
      ? this.parseChannelMajor(data, channels, srcW, srcH)
      : this.parseRowMajor(data, channels, srcW, srcH);

    if (results.length === 0) {
      return channelMajor
        ? this.parseRowMajor(data, channels, srcW, srcH)
        : this.parseChannelMajor(data, channels, srcW, srcH);
    }

    return results;
  }

  private parseChannelMajor(
    data: Float32Array,
    channels: number,
    srcW: number,
    srcH: number,
  ): DetectedObject[] {
    const numBoxes = data.length / channels;
    return this.collectDetections(
      numBoxes,
      (boxIndex, classIndex) => data[(4 + classIndex) * numBoxes + boxIndex],
      (boxIndex) => ({
        cx: data[0 * numBoxes + boxIndex],
        cy: data[1 * numBoxes + boxIndex],
        w: data[2 * numBoxes + boxIndex],
        h: data[3 * numBoxes + boxIndex],
      }),
      srcW,
      srcH,
    );
  }

  private parseRowMajor(
    data: Float32Array,
    channels: number,
    srcW: number,
    srcH: number,
  ): DetectedObject[] {
    const numBoxes = data.length / channels;
    return this.collectDetections(
      numBoxes,
      (boxIndex, classIndex) => {
        const offset = boxIndex * channels;
        return data[offset + 4 + classIndex];
      },
      (boxIndex) => {
        const offset = boxIndex * channels;
        return {
          cx: data[offset],
          cy: data[offset + 1],
          w: data[offset + 2],
          h: data[offset + 3],
        };
      },
      srcW,
      srcH,
    );
  }

  private collectDetections(
    numBoxes: number,
    scoreAt: (boxIndex: number, classIndex: number) => number,
    boxAt: (boxIndex: number) => { cx: number; cy: number; w: number; h: number },
    srcW: number,
    srcH: number,
  ): DetectedObject[] {
    const numClasses = COCO_LABELS.length;
    const results: DetectedObject[] = [];
    const scaleX = srcW / this.inputSize;
    const scaleY = srcH / this.inputSize;

    for (let i = 0; i < numBoxes; i++) {
      let maxScore = 0;
      let maxClass = -1;
      for (let c = 0; c < numClasses; c++) {
        const score = scoreAt(i, c);
        if (score > maxScore) {
          maxScore = score;
          maxClass = c;
        }
      }
      if (maxScore < 0.35) continue;

      const label = COCO_LABELS[maxClass];
      if (!TARGET_COCO_LABELS.has(label)) continue;

      const { cx, cy, w, h } = boxAt(i);
      const normalized = Math.max(Math.abs(cx), Math.abs(cy), Math.abs(w), Math.abs(h)) <= 1.5;

      let x: number;
      let y: number;
      let bw: number;
      let bh: number;

      if (normalized) {
        x = cx - w / 2;
        y = cy - h / 2;
        bw = w;
        bh = h;
      } else {
        const absW = Math.abs(w * scaleX);
        const absH = Math.abs(h * scaleY);
        if (absW < 2 || absH < 2) continue;
        x = (cx * scaleX - absW / 2) / srcW;
        y = (cy * scaleY - absH / 2) / srcH;
        bw = absW / srcW;
        bh = absH / srcH;
        results.push({
          label,
          displayLabel: LABEL_DISPLAY[label] ?? label,
          confidence: maxScore,
          bbox: { x, y, width: bw, height: bh },
        });
        continue;
      }

      results.push({
        label,
        displayLabel: LABEL_DISPLAY[label] ?? label,
        confidence: maxScore,
        bbox: { x, y, width: bw, height: bh },
      });
    }

    return this.nms(results, 0.5).slice(0, 20);
  }

  private nms(boxes: DetectedObject[], threshold: number): DetectedObject[] {
    const sorted = [...boxes].sort((a, b) => b.confidence - a.confidence);
    const kept: DetectedObject[] = [];

    for (const box of sorted) {
      let overlap = false;
      for (const k of kept) {
        if (box.label === k.label && this.iou(box.bbox, k.bbox) > threshold) {
          overlap = true;
          break;
        }
      }
      if (!overlap) kept.push(box);
    }
    return kept;
  }

  private iou(a: DetectedObject['bbox'], b: DetectedObject['bbox']): number {
    const x1 = Math.max(a.x, b.x);
    const y1 = Math.max(a.y, b.y);
    const x2 = Math.min(a.x + a.width, b.x + b.width);
    const y2 = Math.min(a.y + a.height, b.y + b.height);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const union = a.width * a.height + b.width * b.height - inter;
    return union > 0 ? inter / union : 0;
  }

  dispose(): void {
    this.session = null;
    this.mpDetector?.close();
    this.mpDetector = null;
  }
}
