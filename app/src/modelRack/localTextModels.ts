/**
 * Official ONNX Instruct builds for browser.
 * - 360M: default on capable devices (better coherence).
 * - 135M: fallback for low memory (≤4GB) or weak mobile devices.
 */
export const SMOLLM_HF_MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct";
export const SMOLLM_RACK_ID = `hf--${SMOLLM_HF_MODEL_ID.replace(/\//g, "--")}`;

export const SMOLLM_135M_HF_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct";
export const SMOLLM_135M_RACK_ID = `hf--${SMOLLM_135M_HF_MODEL_ID.replace(/\//g, "--")}`;

export const LOCAL_TEXT_READY_KEY = "grovee_local_text_ready_v1";

import type { RackModelEntry, RackModelStatus } from "./modelRack";

const DOWNLOADABLE_TEXT_BUILTINS: RackModelEntry[] = [
  {
    id: SMOLLM_RACK_ID,
    label: "SmolLM2 360M",
    modality: "text",
    adapter: "hf-local-text",
    status: "not_downloaded",
    source: "builtin",
    hfModelId: SMOLLM_HF_MODEL_ID,
    pipelineTag: "text-generation",
    addedAt: 0,
  },
  {
    id: SMOLLM_135M_RACK_ID,
    label: "SmolLM2 135M",
    modality: "text",
    adapter: "hf-local-text",
    status: "not_downloaded",
    source: "builtin",
    hfModelId: SMOLLM_135M_HF_MODEL_ID,
    pipelineTag: "text-generation",
    addedAt: 1,
  },
];

const RACK_TO_HF = new Map(
  DOWNLOADABLE_TEXT_BUILTINS.map((e) => [e.id, e.hfModelId ?? ""] as const),
);

export function hfModelIdForLocalTextRack(rackId: string): string | null {
  const id = RACK_TO_HF.get(rackId);
  return id || null;
}

export function isLocalTextRackId(rackId: string): boolean {
  return RACK_TO_HF.has(rackId);
}

function readStorage<T>(key: string): T | null {
  if (typeof localStorage === "undefined") return null;
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return null;
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

function writeStorage(key: string, value: unknown): void {
  if (typeof localStorage === "undefined") return;
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* quota */
  }
}

export function readLocalTextReadyIds(): string[] {
  const raw = readStorage<string[]>(LOCAL_TEXT_READY_KEY);
  return Array.isArray(raw) ? raw.filter((id) => typeof id === "string") : [];
}

export function markLocalTextReady(rackId: string): void {
  const ids = new Set(readLocalTextReadyIds());
  ids.add(rackId);
  writeStorage(LOCAL_TEXT_READY_KEY, [...ids]);
}

export function isLocalTextAdapter(entry: RackModelEntry): boolean {
  return entry.adapter === "hf-local-text";
}

export function localTextStatusForRackId(
  rackId: string,
  downloadingId: string | null,
): RackModelStatus {
  if (downloadingId === rackId) return "downloading";
  if (readLocalTextReadyIds().includes(rackId)) return "ready";
  return "not_downloaded";
}

export function applyLocalTextDownloadStates(
  rack: RackModelEntry[],
  downloadingId: string | null = null,
): RackModelEntry[] {
  return rack.map((entry) => {
    if (!isLocalTextAdapter(entry)) return entry;
    return { ...entry, status: localTextStatusForRackId(entry.id, downloadingId) };
  });
}

export function getDownloadableTextBuiltins(): RackModelEntry[] {
  return DOWNLOADABLE_TEXT_BUILTINS.map((b) => ({ ...b }));
}

export function isSelectableInPicker(entry: RackModelEntry): boolean {
  if (entry.adapter === "gemma-local") return true;
  if (entry.adapter === "hf-local-text") return entry.status === "ready";
  return entry.status === "ready";
}

/** Entries in the full model rack (chat + cloud image backends). */
export function isPickerRackEntry(entry: RackModelEntry): boolean {
  if (entry.adapter === "gemma-local" || entry.adapter === "hf-local-text") return true;
  return entry.status === "ready";
}

/** Header chat picker — text chat models only. */
export function isChatPickerRackEntry(entry: RackModelEntry): boolean {
  return entry.modality === "text" && (entry.adapter === "gemma-local" || entry.adapter === "hf-local-text");
}

export const SMOLLM_CHAT_SYSTEM =
  "You are SmolLM, a helpful assistant. Reply clearly and concisely in the same language the user writes.";
