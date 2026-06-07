import type { DetectedHand, FingerState, StaticGesture } from '../core/types';
import { distance } from '../utils/geometry';

function isFingerExtended(
  landmarks: DetectedHand['landmarks'],
  tip: number,
  pip: number,
  mcp: number,
  wrist: number,
): boolean {
  return distance(landmarks[tip], landmarks[wrist]) > distance(landmarks[pip], landmarks[wrist]) * 1.05
    && distance(landmarks[tip], landmarks[mcp]) > distance(landmarks[pip], landmarks[mcp]) * 0.9;
}

export function getFingerState(hand: DetectedHand): { fingers: FingerState; count: number } {
  const lm = hand.landmarks;
  const wrist = 0;
  const isRight = hand.handedness === 'Right';

  const thumbExtended = isRight
    ? lm[4].x < lm[3].x
    : lm[4].x > lm[3].x;

  const indexExtended = isFingerExtended(lm, 8, 6, 5, wrist);
  const middleExtended = isFingerExtended(lm, 12, 10, 9, wrist);
  const ringExtended = isFingerExtended(lm, 16, 14, 13, wrist);
  const pinkyExtended = isFingerExtended(lm, 20, 18, 17, wrist);

  const fingers: FingerState = {
    thumb: thumbExtended ? 'Open' : 'Closed',
    index: indexExtended ? 'Open' : 'Closed',
    middle: middleExtended ? 'Open' : 'Closed',
    ring: ringExtended ? 'Open' : 'Closed',
    pinky: pinkyExtended ? 'Open' : 'Closed',
  };

  const count = [thumbExtended, indexExtended, middleExtended, ringExtended, pinkyExtended]
    .filter(Boolean).length;

  return { fingers, count };
}

export function recognizeStaticGestures(hands: DetectedHand[]): StaticGesture[] {
  const gestures: StaticGesture[] = [];

  for (const hand of hands) {
    const { fingers, count } = getFingerState(hand);
    const lm = hand.landmarks;

    if (count === 0) {
      gestures.push({ name: 'Fist', confidence: 0.92, hand: hand.handedness });
      continue;
    }

    if (count === 5) {
      gestures.push({ name: 'Open Palm', confidence: 0.9, hand: hand.handedness });
    }

    if (count === 1 && fingers.thumb === 'Open') {
      const thumbUp = lm[4].y < lm[3].y;
      gestures.push({
        name: thumbUp ? 'Thumbs Up' : 'Thumbs Down',
        confidence: 0.88,
        hand: hand.handedness,
      });
    }

    if (fingers.index === 'Open' && fingers.middle === 'Closed' && count <= 2) {
      gestures.push({ name: 'Pointing', confidence: 0.85, hand: hand.handedness });
    }

    if (fingers.index === 'Open' && fingers.middle === 'Open' && fingers.ring === 'Closed') {
      gestures.push({ name: 'Peace Sign', confidence: 0.87, hand: hand.handedness });
    }

    const thumbTip = lm[4];
    const indexTip = lm[8];
    const okDist = distance(thumbTip, indexTip);
    if (okDist < 0.04 && fingers.middle === 'Open') {
      gestures.push({ name: 'OK Sign', confidence: 0.84, hand: hand.handedness });
    }

    if (count === 4) {
      gestures.push({ name: '4 Fingers', confidence: 0.86, hand: hand.handedness });
    }

    gestures.push({
      name: `${count} Finger${count === 1 ? '' : 's'}`,
      confidence: 0.8,
      hand: hand.handedness,
    });
  }

  return gestures;
}
