import type { DetectedHand, MotionGesture } from '../core/types';
import { countDirectionChanges } from '../utils/geometry';

interface HandHistory {
  timestamps: number[];
  xPositions: number[];
  yPositions: number[];
}

export class MotionGestureDetector {
  private history = new Map<string, HandHistory>();
  private clapHistory: number[] = [];

  update(hands: DetectedHand[], timestamp: number): MotionGesture[] {
    const gestures: MotionGesture[] = [];

    if (hands.length === 2) {
      const dist = Math.hypot(
        hands[0].landmarks[0].x - hands[1].landmarks[0].x,
        hands[0].landmarks[0].y - hands[1].landmarks[0].y,
      );
      this.clapHistory.push(dist);
      if (this.clapHistory.length > 20) this.clapHistory.shift();

      const changes = countDirectionChanges(this.clapHistory, 0.02);
      const minDist = Math.min(...this.clapHistory);
      if (changes >= 2 && minDist < 0.08) {
        gestures.push({ name: 'Clapping', confidence: 0.9 });
      }
    }

    for (const hand of hands) {
      const key = hand.handedness;
      const wrist = hand.landmarks[0];
      let hist = this.history.get(key);
      if (!hist) {
        hist = { timestamps: [], xPositions: [], yPositions: [] };
        this.history.set(key, hist);
      }

      hist.timestamps.push(timestamp);
      hist.xPositions.push(wrist.x);
      hist.yPositions.push(wrist.y);

      if (hist.timestamps.length > 20) {
        hist.timestamps.shift();
        hist.xPositions.shift();
        hist.yPositions.shift();
      }

      const xChanges = countDirectionChanges(hist.xPositions, 0.015);
      const xRange = Math.max(...hist.xPositions) - Math.min(...hist.xPositions);

      if (xChanges >= 3 && xRange > 0.08) {
        gestures.push({ name: 'Waving', confidence: Math.min(0.95, 0.7 + xChanges * 0.05) });
      }

      const yChanges = countDirectionChanges(hist.yPositions, 0.01);
      if (yChanges >= 2 && wrist.y < 0.4) {
        gestures.push({ name: 'Come Here', confidence: 0.75 });
      }

      const isRaised = wrist.y < 0.35;
      const isOpen = hand.landmarks[8].y < hand.landmarks[6].y;
      const xStable = xRange < 0.03;
      if (isRaised && isOpen && xStable && hist.timestamps.length > 10) {
        gestures.push({ name: 'Stop Sign', confidence: 0.82 });
      }
    }

    return gestures;
  }

  reset(): void {
    this.history.clear();
    this.clapHistory = [];
  }
}
