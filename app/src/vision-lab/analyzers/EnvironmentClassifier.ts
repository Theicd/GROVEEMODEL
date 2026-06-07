import type { DetectedObject, EnvironmentType } from '../core/types';

const ENVIRONMENT_RULES: Array<{ env: EnvironmentType; objects: string[]; minMatches: number }> = [
  { env: 'Office', objects: ['laptop', 'keyboard', 'mouse', 'chair', 'tv'], minMatches: 2 },
  { env: 'Kitchen', objects: ['refrigerator', 'oven', 'sink', 'bottle', 'cup'], minMatches: 2 },
  { env: 'Living Room', objects: ['couch', 'tv', 'potted plant'], minMatches: 2 },
  { env: 'Bedroom', objects: ['bed', 'clock'], minMatches: 1 },
  { env: 'Classroom', objects: ['chair', 'laptop', 'book', 'clock'], minMatches: 3 },
  { env: 'Vehicle', objects: ['car'], minMatches: 1 },
];

export function classifyEnvironment(objects: DetectedObject[]): EnvironmentType {
  const labels = new Set(objects.map((o) => o.label));
  let best: EnvironmentType = 'Unknown';
  let bestScore = 0;

  for (const rule of ENVIRONMENT_RULES) {
    const matches = rule.objects.filter((o) => labels.has(o)).length;
    if (matches >= rule.minMatches && matches > bestScore) {
      bestScore = matches;
      best = rule.env;
    }
  }

  return best;
}
