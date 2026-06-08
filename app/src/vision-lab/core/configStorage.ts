import type { PipelineConfig } from './types';
import { DEFAULT_TOGGLES } from './types';
import { intervalsFromMode } from './schedule';

const STORAGE_KEY = 'grovee-vision-pipeline-config-v2';

/** GROVEE: VLM off by default — Gemma handles chat; lab uses rule-based sceneDescription. */
export const GROVEE_DEFAULT_TOGGLES = {
  ...DEFAULT_TOGGLES,
  vlm: false,
};

export const DEFAULT_PIPELINE_CONFIG: PipelineConfig = {
  performanceMode: 'balanced',
  toggles: { ...GROVEE_DEFAULT_TOGGLES },
  sampleIntervals: intervalsFromMode('balanced'),
};

export function loadPipelineConfig(): PipelineConfig {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_PIPELINE_CONFIG;

    const parsed = JSON.parse(raw) as Partial<PipelineConfig>;
    return {
      performanceMode: parsed.performanceMode ?? DEFAULT_PIPELINE_CONFIG.performanceMode,
      toggles: { ...GROVEE_DEFAULT_TOGGLES, ...parsed.toggles },
      sampleIntervals: {
        ...DEFAULT_PIPELINE_CONFIG.sampleIntervals,
        ...parsed.sampleIntervals,
      },
    };
  } catch {
    return DEFAULT_PIPELINE_CONFIG;
  }
}

/** Ensure core lab modules stay on — matches browser-vision-lab behavior. */
export function ensureVisionLabConfig(
  config: PipelineConfig,
  tier: 'low' | 'normal' = 'normal',
): PipelineConfig {
  const mode = tier === 'low' ? 'lite' : config.performanceMode;
  const defaults = intervalsFromMode(mode === 'lite' ? 'lite' : mode);
  const intervals = { ...defaults, ...config.sampleIntervals };
  return {
    ...config,
    performanceMode: mode,
    toggles: {
      ...GROVEE_DEFAULT_TOGGLES,
      ...config.toggles,
      yolo: true,
      pose: true,
      hands: true,
      vlm: false,
      face: tier === 'low' ? false : config.toggles.face !== false,
      emotion: tier === 'low' ? false : config.toggles.emotion !== false,
    },
    sampleIntervals: {
      ...intervals,
      hands: Math.min(intervals.hands, mode === 'lite' ? 150 : 100),
      pose: Math.min(intervals.pose, mode === 'lite' ? 3000 : 500),
      face: Math.min(intervals.face, mode === 'lite' ? 5000 : 2000),
      emotion: Math.min(intervals.emotion, mode === 'lite' ? 5000 : 2000),
    },
  };
}

export function savePipelineConfig(config: PipelineConfig): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
  } catch {
    // ignore quota / private mode errors
  }
}
