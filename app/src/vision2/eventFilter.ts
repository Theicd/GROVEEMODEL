/** Filter raw vision events superseded by Vision 2.0 body-language model. */

import type { SemanticEvent } from "../worldMemory";

const VISION2_SUPPRESSED_SUBJECTS = new Set([
  "hand_on_face",
  "hand_on_head",
  "hands_on_head",
  "gesture:one_finger",
  "gesture:two_fingers",
]);

export const filterEventsForVision2 = (events: SemanticEvent[]): SemanticEvent[] =>
  events.filter((e) => {
    const sub = e.subject ?? "";
    if (VISION2_SUPPRESSED_SUBJECTS.has(sub)) return false;
    if (e.type === "activity_change" && /hand on face|finger/i.test(e.text)) return false;
    return true;
  });
