import type {
  DetectedEvent,
  DetectedObject,
  Interaction,
  MotionGesture,
  PoseAction,
  StaticGesture,
  BodyLanguageCue,
} from '../core/types';

export function buildRuleBasedDescription(params: {
  objects: DetectedObject[];
  poseActions: PoseAction[];
  staticGestures: StaticGesture[];
  motionGestures: MotionGesture[];
  interactions: Interaction[];
  events: DetectedEvent[];
  environment: string;
  emotionDominant?: string;
  bodyLanguage?: BodyLanguageCue[];
}): string {
  const sentences: string[] = [];
  const {
    objects, poseActions, staticGestures, motionGestures,
    interactions, events, environment, emotionDominant,
  } = params;

  const people = objects.filter((o) => o.label === 'person').length;
  if (people === 1) {
    sentences.push('A person is visible.');
  } else if (people > 1) {
    sentences.push(`${people} people are visible.`);
  }

  const primaryPose = poseActions[0]?.name.toLowerCase();
  if (primaryPose && !primaryPose.includes('hand')) {
    sentences.push(`The person is ${primaryPose.toLowerCase()}.`);
  }

  const objectLabels = objects
    .filter((o) => o.label !== 'person')
    .map((o) => o.displayLabel.toLowerCase());
  if (objectLabels.length) {
    sentences.push(`Detected objects include ${objectLabels.slice(0, 4).join(', ')}.`);
  }

  for (const interaction of interactions.slice(0, 2)) {
    sentences.push(`${interaction.name} detected.`);
  }

  const wave = motionGestures.find((g) => g.name === 'Waving');
  if (wave) {
    const hand = staticGestures[0]?.hand?.toLowerCase() ?? 'a';
    sentences.push(`The ${hand} hand appears to be waving.`);
  }

  for (const event of events.slice(0, 2)) {
    sentences.push(`Event: ${event.name}.`);
  }

  if (environment !== 'Unknown') {
    sentences.push(`Environment appears to be an ${environment.toLowerCase()}.`);
  }

  if (emotionDominant) {
    sentences.push(`Estimated emotion: ${emotionDominant}.`);
  }

  for (const cue of (params.bodyLanguage ?? []).slice(0, 3)) {
    sentences.push(`${cue.signal}: ${cue.meaning}.`);
  }

  return sentences.join(' ') || 'Waiting for detections...';
}
