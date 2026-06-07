/** System prompts for live camera vision layer (structured world — not character voice). */

export const SCENE_ANALYSIS_SYSTEM_PROMPT = `You are a vision layer building structured world memory from periodic camera snapshots (not video).
Compare to the previous summary when provided.
Return ONLY valid JSON (no markdown):
{"objects":["item",...],"people":["person",...],"events":["semantic change phrase",...],"interesting":true|false,"summary":"one sentence"}
Rules:
- "objects": stable visible items (guitar, laptop, door, clock, phone…) — max 10, English short labels.
- "people": visible people descriptors — max 3.
- "events": ONLY new changes vs previous state (Person entered, Door opened, Object removed…). Empty if nothing new.
- "interesting": true only for meaningful changes worth a character noticing.
- Do not repeat unchanged items in events.
- Be factual; do not invent.`;

export const buildSceneAnalysisUserPrompt = (previousSummary?: string, sensorBlock?: string): string => {
  const sensor = sensorBlock?.trim()
    ? `\n\nStructured sensors (use for reasoning, not verbatim listing):\n${sensorBlock.trim()}`
    : "";
  if (previousSummary?.trim()) {
    return `Previous world summary: ${previousSummary.trim()}${sensor}\nUpdate objects/people/events. Return JSON only.`;
  }
  return `Initialize world memory from this snapshot.${sensor}\nReturn JSON only.`;
};
