/** L1 — audio sensor stub (mic hook for future; no LLM exposure). */

export type AudioSample = {
  timestamp: number;
  /** RMS level 0–1 */
  level: number;
  /** Heuristic speech detection */
  speechDetected: boolean;
  /** Ambient noise floor */
  noiseFloor: number;
};

export type AudioSensorState = {
  available: boolean;
  lastSample: AudioSample | null;
  error: string | null;
};

export class AudioSensor {
  private state: AudioSensorState = {
    available: false,
    lastSample: null,
    error: null,
  };

  /** Stub — returns silence until Web Audio mic is wired. */
  sample(now = Date.now()): AudioSample {
    const sample: AudioSample = {
      timestamp: now,
      level: 0,
      speechDetected: false,
      noiseFloor: 0.02,
    };
    this.state.lastSample = sample;
    return sample;
  }

  getState(): AudioSensorState {
    return { ...this.state, lastSample: this.state.lastSample ? { ...this.state.lastSample } : null };
  }

  /** Future: attach MediaStream from getUserMedia audio track. */
  attachStream(_stream: MediaStream | null): void {
    this.state.available = false;
    this.state.error = null;
  }

  reset(): void {
    this.state = { available: false, lastSample: null, error: null };
  }
}

export const audioSummaryForContext = (sample: AudioSample | null): {
  available: boolean;
  level: number;
  speechDetected: boolean;
} => ({
  available: sample !== null,
  level: sample?.level ?? 0,
  speechDetected: sample?.speechDetected ?? false,
});
