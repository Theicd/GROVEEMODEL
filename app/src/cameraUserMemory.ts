/**
 * HAL user memory — profile, rolling summaries, history search.
 */

import type { CameraMessage, CameraSessionStore, UserProfile } from "./cameraSession";
import { saveCameraSessionStore } from "./cameraSession";

export const emptyUserProfile = (): UserProfile => ({
  name: "",
  hobbies: [],
  notes: "",
  updatedAt: 0,
});

/** Merge hobby chips without duplicates. */
export const mergeHobbies = (existing: string[], incoming: string[]): string[] => {
  const out = [...existing];
  for (const h of incoming) {
    const t = h.trim();
    if (!t) continue;
    if (!out.some((x) => x.toLowerCase() === t.toLowerCase())) out.push(t);
  }
  return out.slice(-24);
};

/** Heuristic: pull name / hobbies from user text into profile. */
export const extractProfileHintsFromUserText = (text: string): Partial<UserProfile> => {
  const t = text.trim();
  if (!t) return {};
  const hobbies: string[] = [];
  if (/מדע בדיונ/i.test(t)) hobbies.push("מדע בדיוני");
  if (/קונספיר/i.test(t)) hobbies.push("קונספירציות");
  if (/ברמודה|משולש/i.test(t)) hobbies.push("ברמודה");
  if (/פירמיד/i.test(t)) hobbies.push("פירמידות");
  if (/ירח|חלל|space/i.test(t)) hobbies.push("חלל");
  if (/משחק|game/i.test(t)) hobbies.push("משחקים");
  if (/סיפור|fiction|story/i.test(t)) hobbies.push("סיפורים");

  let name = "";
  const namePatterns = [
    /(?:קוראים לי|שמי|אני)\s+([^\s,.!?]{2,20})/i,
    /(?:my name is|i'?m|call me)\s+([a-zA-Z\u0590-\u05FF]{2,20})/i,
    /נעים מאוד[,.\s]+(?:אני\s+)?([^\s,.!?]{2,20})/i,
  ];
  for (const re of namePatterns) {
    const m = t.match(re);
    if (m?.[1] && !/מחפש|מעוניין|רוצה|אתה|groovee|גרווי/i.test(m[1])) {
      name = m[1].trim();
      break;
    }
  }

  return { ...(name ? { name } : {}), ...(hobbies.length ? { hobbies } : {}) };
};

export const applyProfileHints = (profile: UserProfile, hints: Partial<UserProfile>): UserProfile => {
  if (!hints.name && !hints.hobbies?.length) return profile;
  return {
    ...profile,
    name: hints.name?.trim() || profile.name,
    hobbies: hints.hobbies?.length ? mergeHobbies(profile.hobbies, hints.hobbies) : profile.hobbies,
    updatedAt: Date.now(),
  };
};

const tokenize = (text: string): string[] =>
  text
    .toLowerCase()
    .split(/[\s,.!?;:()\-—]+/)
    .map((w) => w.trim())
    .filter((w) => w.length >= 2);

/** Score message relevance to query (simple keyword overlap). */
export const scoreMessageRelevance = (msg: CameraMessage, query: string): number => {
  const qTokens = tokenize(query);
  if (!qTokens.length) return 0;
  const body = `${msg.content}`.toLowerCase();
  let score = 0;
  for (const tok of qTokens) {
    if (body.includes(tok)) score += tok.length >= 4 ? 2 : 1;
  }
  if (msg.role === "user") score += 0.5;
  return score;
};

export type HistorySearchHit = {
  message: CameraMessage;
  score: number;
  snippet: string;
};

export const searchCameraHistory = (
  messages: CameraMessage[],
  query: string,
  limit = 8,
): HistorySearchHit[] => {
  const q = query.trim();
  if (!q || q.length < 2) return [];
  const hits: HistorySearchHit[] = [];
  for (const msg of messages) {
    const score = scoreMessageRelevance(msg, q);
    if (score <= 0) continue;
    const snippet =
      msg.content.length > 120 ? `${msg.content.slice(0, 117)}…` : msg.content;
    hits.push({ message: msg, score, snippet });
  }
  return hits.sort((a, b) => b.score - a.score || b.message.ts - a.message.ts).slice(0, limit);
};

/** Find snippets relevant to the current user message (for prompt injection). */
export const findRelevantHistoryForPrompt = (
  messages: CameraMessage[],
  userText: string,
  limit = 4,
): HistorySearchHit[] => {
  const hits = searchCameraHistory(messages.slice(0, -1), userText, limit);
  return hits.filter((h) => h.score >= 2);
};

/** Rolling 2–4 line summary from recent camera turns. */
export const buildRollingSummary = (messages: CameraMessage[], maxLines = 4): string => {
  if (messages.length === 0) return "";
  const recent = messages.slice(-12);
  const lines: string[] = [];
  for (const m of recent) {
    const short =
      m.content.replace(/\s+/g, " ").trim().slice(0, 100) +
      (m.content.length > 100 ? "…" : "");
    if (!short) continue;
    const who = m.role === "user" ? "User" : "HAL";
    lines.push(`${who}: ${short}`);
  }
  return lines.slice(-maxLines).join("\n");
};

export const buildUserMemoryPromptBlock = (params: {
  profile: UserProfile;
  rollingSummary: string;
  relevantSnippets?: HistorySearchHit[];
}): string => {
  const { profile, rollingSummary, relevantSnippets } = params;
  const lines: string[] = [
    "[USER MEMORY — use naturally, do not read aloud as a list]",
  ];
  if (profile.name.trim()) lines.push(`name=${profile.name.trim()}`);
  if (profile.hobbies.length) lines.push(`hobbies=${profile.hobbies.join(", ")}`);
  if (profile.notes.trim()) lines.push(`notes=${profile.notes.trim().slice(0, 280)}`);
  if (rollingSummary.trim()) {
    lines.push("recent_dialogue_summary:");
    lines.push(rollingSummary.trim());
  }
  if (relevantSnippets?.length) {
    lines.push("relevant_past_turns:");
    for (const h of relevantSnippets) {
      const who = h.message.role === "user" ? "User" : "HAL";
      lines.push(`· ${who}: ${h.snippet}`);
    }
  }
  lines.push("[/USER MEMORY]");
  return lines.join("\n");
};

export const patchCameraStoreAfterTurn = (
  store: CameraSessionStore,
  userText: string,
): CameraSessionStore => {
  const hints = extractProfileHintsFromUserText(userText);
  const profile = applyProfileHints(store.profile, hints);
  const rollingSummary = buildRollingSummary(store.messages);
  return {
    ...store,
    profile,
    rollingSummary,
    updatedAt: Date.now(),
  };
};

export const updateUserProfile = (
  store: CameraSessionStore,
  patch: Partial<UserProfile>,
): CameraSessionStore => {
  const hobbies =
    patch.hobbies !== undefined
      ? patch.hobbies.map((h) => h.trim()).filter(Boolean)
      : store.profile.hobbies;
  const next: CameraSessionStore = {
    ...store,
    profile: {
      name: patch.name !== undefined ? patch.name.trim() : store.profile.name,
      hobbies,
      notes: patch.notes !== undefined ? patch.notes.trim() : store.profile.notes,
      updatedAt: Date.now(),
    },
    updatedAt: Date.now(),
  };
  saveCameraSessionStore(next);
  return next;
};
