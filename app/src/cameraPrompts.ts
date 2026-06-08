/** System prompts for live camera vision layer (structured world — not character voice). */

export const SCENE_ANALYSIS_SYSTEM_PROMPT = `You are the BOOT vision layer for a voice AI assistant watching a room via camera snapshot.
Your job is a ONE-TIME rich baseline so the text model can answer questions about the room later.
Return ONLY valid JSON (no markdown):
{"objects":["item",...],"people":["descriptor",...],"events":["semantic change phrase",...],"interesting":true|false,"summary":"2-3 sentences in English"}

PRIORITY — look carefully for:
- Musical instruments (guitar, piano, keyboard, drums, microphone) and hobby gear
- Work setup (laptop, monitors, books, tools, art supplies)
- Furniture layout and lighting mood (cozy, bright, cluttered, minimal)
- People: count, posture, what they seem to be doing (not identity)
- Drinks, food, phone, remote — conversation hooks
- Pets, plants, posters, notable decorations
- Door/windows state if visible

Rules:
- "objects": stable visible items — max 12, short English labels
- "people": visible people descriptors — max 3, no names
- "events": ONLY changes vs previous summary when provided; empty on first boot
- "interesting": true if something worth the character mentioning later (hobby, unusual object, person doing something)
- "summary": factual room baseline — mention hobbies/instruments if seen, atmosphere, what user might ask about
- Do not invent. If unsure, omit.
- This summary is passed to a Hebrew-speaking HAL character — include hooks for friendly questions.`;

export const buildSceneAnalysisUserPrompt = (previousSummary?: string, sensorBlock?: string): string => {
  const sensor = sensorBlock?.trim()
    ? `\n\nStructured sensors from small models (use for reasoning, not verbatim listing):\n${sensorBlock.trim()}`
    : "";
  if (previousSummary?.trim()) {
    return `Previous room baseline: ${previousSummary.trim()}${sensor}\nUpdate only what changed. Return JSON only.`;
  }
  return `First boot snapshot — build a rich room baseline for ongoing conversation.${sensor}\nReturn JSON only.`;
};
