/** Tracks boolean / numeric signals over time (duration, EMA). */

export class BoolTracker {
  private activeSince = 0;
  private lastValue = false;

  update(value: boolean, now = Date.now()): { value: boolean; durationSec: number; rising: boolean } {
    const rising = value && !this.lastValue;
    if (value && !this.lastValue) {
      this.activeSince = now;
    } else if (!value) {
      this.activeSince = 0;
    }
    this.lastValue = value;
    const durationSec =
      value && this.activeSince ? Math.max(0, (now - this.activeSince) / 1000) : 0;
    return { value, durationSec, rising };
  }

  reset(): void {
    this.activeSince = 0;
    this.lastValue = false;
  }
}

export class EmaTracker {
  private value = 0;
  private initialized = false;

  update(sample: number, alpha = 0.25): number {
    if (!this.initialized) {
      this.value = sample;
      this.initialized = true;
    } else {
      this.value = alpha * sample + (1 - alpha) * this.value;
    }
    return this.value;
  }

  get(): number {
    return this.value;
  }

  reset(): void {
    this.value = 0;
    this.initialized = false;
  }
}

export class DominantStateTracker<T extends string> {
  private since = 0;
  private current: T | null = null;

  update(next: T, now = Date.now()): { value: T; ageSec: number } {
    if (this.current !== next) {
      this.current = next;
      this.since = now;
    }
    const ageSec = this.since ? Math.max(0, (now - this.since) / 1000) : 0;
    return { value: next, ageSec };
  }

  reset(): void {
    this.since = 0;
    this.current = null;
  }
}
