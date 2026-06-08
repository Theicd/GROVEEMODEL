import type { PerformanceMode, PipelineConfig, SampleIntervals } from './types';

/** Spacing between module phase offsets to reduce same-frame bursts. */
const STAGGER_SLOT_MS = 400;

export interface ModuleSchedule {
  intervalMs: number;
  phaseOffsetMs: number;
}

export interface ResolvedSchedule {
  hands: ModuleSchedule;
  pose: ModuleSchedule;
  yolo: ModuleSchedule;
  face: ModuleSchedule;
  emotion: ModuleSchedule;
  vlm: ModuleSchedule;
  uiUpdateMs: number;
}

export function getSchedule(mode: PerformanceMode): ResolvedSchedule {
  switch (mode) {
    case 'lite':
      return {
        hands: { intervalMs: 150, phaseOffsetMs: 0 },
        pose: { intervalMs: 3000, phaseOffsetMs: 0 },
        yolo: { intervalMs: 550, phaseOffsetMs: 0 },
        face: { intervalMs: 3000, phaseOffsetMs: 0 },
        emotion: { intervalMs: 3000, phaseOffsetMs: 0 },
        vlm: { intervalMs: 5000, phaseOffsetMs: 0 },
        uiUpdateMs: 120,
      };
    case 'full':
      return {
        hands: { intervalMs: 80, phaseOffsetMs: 0 },
        pose: { intervalMs: 3000, phaseOffsetMs: 0 },
        yolo: { intervalMs: 350, phaseOffsetMs: 0 },
        face: { intervalMs: 3000, phaseOffsetMs: 0 },
        emotion: { intervalMs: 3000, phaseOffsetMs: 0 },
        vlm: { intervalMs: 2000, phaseOffsetMs: 0 },
        uiUpdateMs: 80,
      };
    default:
      return {
        hands: { intervalMs: 100, phaseOffsetMs: 0 },
        pose: { intervalMs: 500, phaseOffsetMs: 0 },
        yolo: { intervalMs: 600, phaseOffsetMs: 0 },
        face: { intervalMs: 2000, phaseOffsetMs: 0 },
        emotion: { intervalMs: 2000, phaseOffsetMs: 0 },
        vlm: { intervalMs: 4000, phaseOffsetMs: 0 },
        uiUpdateMs: 100,
      };
  }
}

export function intervalsFromMode(mode: PerformanceMode): SampleIntervals {
  const schedule = getSchedule(mode);
  return {
    yolo: schedule.yolo.intervalMs,
    pose: schedule.pose.intervalMs,
    hands: schedule.hands.intervalMs,
    face: schedule.face.intervalMs,
    emotion: schedule.emotion.intervalMs,
    vlm: schedule.vlm.intervalMs,
    uiUpdate: schedule.uiUpdateMs,
  };
}

/**
 * Builds a staggered schedule from user intervals and enabled modules.
 * Light modules (hands/pose) may co-run; heavy modules get spread phase offsets.
 * Face and emotion are offset by half the shorter interval when both are enabled.
 */
export function resolveSchedule(config: PipelineConfig): ResolvedSchedule {
  const { sampleIntervals: intervals, toggles } = config;
  let slot = 1;

  const nextSlot = (): number => {
    const offset = slot * STAGGER_SLOT_MS;
    slot += 1;
    return offset;
  };

  const hands: ModuleSchedule = {
    intervalMs: intervals.hands,
    phaseOffsetMs: 0,
  };

  const pose: ModuleSchedule = {
    intervalMs: intervals.pose,
    phaseOffsetMs: toggles.pose ? nextSlot() : 0,
  };

  const yolo: ModuleSchedule = {
    intervalMs: intervals.yolo,
    phaseOffsetMs: toggles.yolo ? nextSlot() : 0,
  };

  let faceOffset = 0;
  if (toggles.face) {
    faceOffset = nextSlot();
  }

  let emotionOffset = 0;
  if (toggles.emotion) {
    if (toggles.face) {
      emotionOffset = faceOffset + Math.round(Math.min(intervals.face, intervals.emotion) / 2);
    } else {
      emotionOffset = nextSlot();
    }
  }

  const face: ModuleSchedule = {
    intervalMs: intervals.face,
    phaseOffsetMs: faceOffset,
  };

  const emotion: ModuleSchedule = {
    intervalMs: intervals.emotion,
    phaseOffsetMs: emotionOffset,
  };

  const vlm: ModuleSchedule = {
    intervalMs: intervals.vlm,
    phaseOffsetMs: toggles.vlm ? nextSlot() : 0,
  };

  return {
    hands,
    pose,
    yolo,
    face,
    emotion,
    vlm,
    uiUpdateMs: intervals.uiUpdate,
  };
}

export function clampInterval(ms: number): number {
  if (!Number.isFinite(ms)) return 1000;
  return Math.max(100, Math.min(60000, Math.round(ms)));
}

export function isModuleDue(
  now: number,
  pipelineStart: number,
  lastRun: number,
  schedule: ModuleSchedule,
): boolean {
  return getModuleOverdue(now, pipelineStart, lastRun, schedule) > 0;
}

/** Milliseconds past the scheduled run time (0 if not yet due). */
export function getModuleOverdue(
  now: number,
  pipelineStart: number,
  lastRun: number,
  schedule: ModuleSchedule,
): number {
  if (lastRun === 0) {
    const firstDueAt = pipelineStart + schedule.phaseOffsetMs;
    if (now < firstDueAt) return 0;
    return now - firstDueAt;
  }
  const nextDueAt = lastRun + schedule.intervalMs;
  if (now < nextDueAt) return 0;
  return now - nextDueAt;
}

export type HeavyModuleKey = 'yolo' | 'faceEmotion' | 'vlm';

export function pickHeavyModule(
  now: number,
  pipelineStart: number,
  lastRun: Record<'yolo' | 'face' | 'emotion' | 'vlm', number>,
  schedule: ResolvedSchedule,
  toggles: PipelineConfig['toggles'],
): HeavyModuleKey | null {
  const candidates: Array<{ key: HeavyModuleKey; overdue: number }> = [];

  if (toggles.yolo) {
    const overdue = getModuleOverdue(now, pipelineStart, lastRun.yolo, schedule.yolo);
    if (overdue > 0) candidates.push({ key: 'yolo', overdue });
  }

  const faceOverdue = toggles.face
    ? getModuleOverdue(now, pipelineStart, lastRun.face, schedule.face)
    : 0;
  const emotionOverdue = toggles.emotion
    ? getModuleOverdue(now, pipelineStart, lastRun.emotion, schedule.emotion)
    : 0;
  const faceEmotionOverdue = Math.max(faceOverdue, emotionOverdue);
  if (faceEmotionOverdue > 0) {
    candidates.push({ key: 'faceEmotion', overdue: faceEmotionOverdue });
  }

  if (toggles.vlm) {
    const overdue = getModuleOverdue(now, pipelineStart, lastRun.vlm, schedule.vlm);
    if (overdue > 0) candidates.push({ key: 'vlm', overdue });
  }

  if (candidates.length === 0) return null;

  candidates.sort((a, b) => b.overdue - a.overdue);
  return candidates[0].key;
}
