export type IntroCinematicStage =
  | "idle"
  | "eyebrow"
  | "title"
  | "subtitle"
  | "typewriter"
  | "action";

/** Ms from mount → stage (start phase only). */
export const CINEMATIC_TIMELINE: { stage: IntroCinematicStage; at: number }[] = [
  { stage: "eyebrow", at: 350 },
  { stage: "title", at: 1100 },
  { stage: "subtitle", at: 1900 },
  { stage: "typewriter", at: 3100 },
  { stage: "action", at: 5200 },
];

export function stageAtLeast(current: IntroCinematicStage, min: IntroCinematicStage): boolean {
  const order: IntroCinematicStage[] = [
    "idle",
    "eyebrow",
    "title",
    "subtitle",
    "typewriter",
    "action",
  ];
  return order.indexOf(current) >= order.indexOf(min);
}
