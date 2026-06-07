import type { BoundingBox, DetectedHand, DetectedObject, Interaction } from '../core/types';
import { bboxDistance, bboxIoU } from '../utils/geometry';

export function analyzeInteractions(
  objects: DetectedObject[],
  hands: DetectedHand[],
  faces: Array<{ bbox: BoundingBox }>,
): Interaction[] {
  const interactions: Interaction[] = [];
  const persons = objects.filter((o) => o.label === 'person');
  const cups = objects.filter((o) => ['cup', 'bottle', 'wine glass'].includes(o.label));
  const backpacks = objects.filter((o) => ['backpack', 'handbag', 'suitcase'].includes(o.label));
  const phones = objects.filter((o) => o.label === 'cell phone');

  for (const cup of cups) {
    for (const hand of hands) {
      const dist = bboxDistance(hand.bbox, cup.bbox);
      if (dist < 0.12 && persons.length > 0) {
        interactions.push({ name: 'Holding Cup', confidence: Math.max(0.7, 1 - dist * 5) });
      }
    }
  }

  for (const person of persons) {
    for (const bp of backpacks) {
      const iou = bboxIoU(person.bbox, bp.bbox);
      if (iou > 0.15 || bboxDistance(person.bbox, bp.bbox) < 0.08) {
        interactions.push({
          name: 'Wearing Backpack',
          confidence: Math.min(0.95, 0.6 + iou * 2),
        });
      }
    }
  }

  for (const phone of phones) {
    for (const face of faces) {
      const faceDist = bboxDistance(phone.bbox, face.bbox);
      for (const hand of hands) {
        const handDist = bboxDistance(hand.bbox, phone.bbox);
        if (faceDist < 0.2 && handDist < 0.15) {
          interactions.push({ name: 'Using Phone', confidence: 0.85 });
        }
      }
    }
  }

  return interactions;
}
