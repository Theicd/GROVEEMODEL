/** F1 — audit that LLM prompts contain no raw sensor dumps. */

const FORBIDDEN_PATTERNS = [
  /"fingerStates"\s*:/i,
  /landmarks?\s*[:=]/i,
  /\bbbox\s*[:=]/i,
  /YOLO objects/i,
  /thumb=Open/i,
  /Sensor layers:\s*\n/i,
  /Finger counts:\s/i,
  /Body language cues:\s/i,
  /FRESH FINGER COUNT/i,
  /buildRichSensorBlock/i,
  /Live state \(updated every frame\)/i,
];

export const promptContainsRawSensors = (prompt: string): boolean =>
  FORBIDDEN_PATTERNS.some((re) => re.test(prompt));

export const assertVision2PromptClean = (prompt: string): void => {
  if (promptContainsRawSensors(prompt)) {
    throw new Error("Vision 2.0 prompt audit failed: raw sensor data detected");
  }
};
