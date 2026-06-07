import type { BoundingBox, Point2D } from '../core/types';

export function distance(a: Point2D, b: Point2D): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

export function angle(a: Point2D, b: Point2D, c: Point2D): number {
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };
  const dot = ab.x * cb.x + ab.y * cb.y;
  const mag = Math.hypot(ab.x, ab.y) * Math.hypot(cb.x, cb.y);
  if (mag === 0) return 0;
  return (Math.acos(Math.max(-1, Math.min(1, dot / mag))) * 180) / Math.PI;
}

export function bboxCenter(bbox: BoundingBox): Point2D {
  return { x: bbox.x + bbox.width / 2, y: bbox.y + bbox.height / 2 };
}

export function bboxFromPoints(points: Point2D[], padding = 0.02): BoundingBox {
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const minX = Math.min(...xs) - padding;
  const minY = Math.min(...ys) - padding;
  const maxX = Math.max(...xs) + padding;
  const maxY = Math.max(...ys) + padding;
  return {
    x: minX,
    y: minY,
    width: maxX - minX,
    height: maxY - minY,
  };
}

export function bboxIoU(a: BoundingBox, b: BoundingBox): number {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width, b.x + b.width);
  const y2 = Math.min(a.y + a.height, b.y + b.height);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a.width * a.height + b.width * b.height - inter;
  return union > 0 ? inter / union : 0;
}

export function bboxDistance(a: BoundingBox, b: BoundingBox): number {
  return distance(bboxCenter(a), bboxCenter(b));
}

export function normalizeLandmarks<T extends Point2D>(
  landmarks: T[],
  width: number,
  height: number,
): T[] {
  return landmarks.map((lm) => ({
    ...lm,
    x: lm.x / width,
    y: lm.y / height,
  }));
}

export function countDirectionChanges(values: number[], threshold: number): number {
  let changes = 0;
  let lastDir = 0;
  for (let i = 1; i < values.length; i++) {
    const delta = values[i] - values[i - 1];
    if (Math.abs(delta) < threshold) continue;
    const dir = delta > 0 ? 1 : -1;
    if (lastDir !== 0 && dir !== lastDir) changes++;
    lastDir = dir;
  }
  return changes;
}

export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export function percent(value: number): number {
  return Math.round(value * 100);
}
