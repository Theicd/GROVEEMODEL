import type {
  DetectedEvent,
  Interaction,
  MotionGesture,
  PoseAction,
  StaticGesture,
} from '../core/types';

export function evaluateEvents(
  objects: string[],
  poseActions: PoseAction[],
  motionGestures: MotionGesture[],
  interactions: Interaction[],
  staticGestures: StaticGesture[] = [],
): DetectedEvent[] {
  const events: DetectedEvent[] = [];
  const poseNames = new Set(poseActions.map((a) => a.name));
  const gestureNames = new Set(motionGestures.map((g) => g.name));
  const staticNames = new Set(staticGestures.map((g) => g.name));
  const interactionNames = new Set(interactions.map((i) => i.name));
  const hasPerson = objects.includes('person');

  if (hasPerson && gestureNames.has('Waving')) {
    events.push({ name: 'Calling for Attention', confidence: 0.92 });
  }

  if (staticNames.has('Thumbs Up')) {
    events.push({ name: 'Like / OK Signal', confidence: 0.9 });
  }

  if (staticNames.has('Thumbs Down')) {
    events.push({ name: 'Dislike Signal', confidence: 0.9 });
  }

  if (hasPerson && interactionNames.has('Holding Cup')) {
    events.push({ name: 'Drinking / Holding Cup', confidence: 0.88 });
  }

  if (hasPerson && (poseNames.has('Right Hand Raised') || poseNames.has('Left Hand Raised')) && gestureNames.has('Waving')) {
    events.push({ name: 'Greeting Gesture', confidence: 0.9 });
  }

  if (hasPerson && interactionNames.has('Using Phone')) {
    events.push({ name: 'Phone Usage', confidence: 0.86 });
  }

  if (hasPerson && poseNames.has('Jumping')) {
    events.push({ name: 'Jumping Activity', confidence: 0.8 });
  }

  if (hasPerson && poseNames.has('Running')) {
    events.push({ name: 'Running Activity', confidence: 0.78 });
  }

  if (gestureNames.has('Clapping')) {
    events.push({ name: 'Applause', confidence: 0.85 });
  }

  if (gestureNames.has('Stop Sign')) {
    events.push({ name: 'Stop Signal', confidence: 0.83 });
  }

  return events;
}
