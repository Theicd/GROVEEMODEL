/** Scanned + built-in models available from the chat header picker. */

export type ModelModality = "text" | "image" | "code" | "video" | "audio" | "vision";

export type ModelAdapter =
  | "gemma-local"
  | "hf-local-text"
  | "pollinations"
  | "sd-turbo-local"
  | "hf-chat"
  | "hf-inference-image"
  | "hf-inference"
  | "hf-gradio-space";

export type RackModelStatus = "ready" | "token_required" | "unavailable" | "not_downloaded" | "downloading";

export type RackModelSource = "builtin" | "hf-scan" | "cloud-scan" | "hf-space";

export type RackModelEntry = {
  id: string;
  label: string;
  modality: ModelModality;
  adapter: ModelAdapter;
  status: RackModelStatus;
  source: RackModelSource;
  hfModelId?: string;
  hfSpaceId?: string;
  gradioEndpoint?: string;
  gradioProbeData?: unknown[];
  hfAccessMode?: "FREE" | "TOKEN";
  pollinationsModel?: string;
  pipelineTag?: string;
  addedAt: number;
};

export type RackCountSummary = {
  builtin: number;
  hf: number;
  spaces: number;
  cloud: number;
  total: number;
};

export const GEMMA_RACK_ID = "gemma-local";
export const MODEL_RACK_STORAGE_KEY = "grovee_model_rack_v2";
const LEGACY_RACK_STORAGE_KEY = "grovee_model_rack_v1";
export const CLOUD_HEALTH_CHECKED_KEY = "grovee_cloud_health_checked_v1";
export const SELECTED_MODEL_STORAGE_KEY = "grovee_selected_model_v1";

const BUILTIN_MODELS: RackModelEntry[] = [
  {
    id: GEMMA_RACK_ID,
    label: "Gemma 4 E2B",
    modality: "text",
    adapter: "gemma-local",
    status: "ready",
    source: "builtin",
    addedAt: 0,
  },
];

import { pollinationsDisplayName } from "./modelRackDisplay";
import {
  applyLocalTextDownloadStates,
  getDownloadableTextBuiltins,
  isChatPickerRackEntry,
  isPickerRackEntry,
  isSelectableInPicker,
} from "./localTextModels";

export {
  SMOLLM_RACK_ID,
  SMOLLM_HF_MODEL_ID,
  SMOLLM_135M_RACK_ID,
  SMOLLM_135M_HF_MODEL_ID,
  SMOLLM_CHAT_SYSTEM,
  hfModelIdForLocalTextRack,
  isLocalTextRackId,
} from "./localTextModels";

const CORE_POLLINATIONS = ["flux", "turbo", "sdxl"] as const;

function pollinationsLabel(model: string): string {
  return pollinationsDisplayName(model);
}

export function rackIdFromPollinations(model: string): string {
  const slug = model.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-");
  return `pollinations-${slug || "flux"}`;
}

function defaultCloudEntry(model: string): RackModelEntry {
  return {
    id: rackIdFromPollinations(model),
    label: pollinationsLabel(model),
    modality: "image",
    adapter: "pollinations",
    status: "ready",
    source: "cloud-scan",
    pollinationsModel: model,
    pipelineTag: "text-to-image",
    addedAt: 0,
  };
}

const DEFAULT_CLOUD_FALLBACKS: RackModelEntry[] = CORE_POLLINATIONS.map(defaultCloudEntry);

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

function readStoredScanned(): RackModelEntry[] {
  const v2 = readStorage<RackModelEntry[]>(MODEL_RACK_STORAGE_KEY);
  if (v2?.length) return v2;
  const legacy = readStorage<RackModelEntry[]>(LEGACY_RACK_STORAGE_KEY);
  if (!legacy?.length) return [];
  writeStorage(MODEL_RACK_STORAGE_KEY, legacy);
  return legacy;
}

export function rackIdFromHfModel(modelId: string): string {
  return `hf--${modelId.replace(/\//g, "--")}`;
}

export function rackIdFromHfSpace(spaceId: string): string {
  return `hf-space--${spaceId.replace(/\//g, "--")}`;
}

/** Models shown in the picker — ready only (no token / unavailable). */
export function isPickableRackEntry(entry: RackModelEntry): boolean {
  return entry.status === "ready";
}

export function pickableRackModels(rack: RackModelEntry[]): RackModelEntry[] {
  return rack.filter(isPickableRackEntry);
}

export function pickerRackModels(rack: RackModelEntry[]): RackModelEntry[] {
  return rack.filter(isPickerRackEntry);
}

export { isChatPickerRackEntry, isSelectableInPicker } from "./localTextModels";

export function summarizeRackCounts(rack: RackModelEntry[]): RackCountSummary {
  const pickable = pickableRackModels(rack);
  return {
    builtin: pickable.filter((r) => r.source === "builtin").length,
    hf: pickable.filter((r) => r.source === "hf-scan").length,
    spaces: pickable.filter((r) => r.source === "hf-space").length,
    cloud: pickable.filter((r) => r.source === "cloud-scan").length,
    total: pickable.length,
  };
}

export function rackEntryTagLabel(entry: RackModelEntry): string | null {
  if (entry.source === "builtin" && entry.adapter === "gemma-local") return "מובנה";
  if (entry.adapter === "hf-local-text") {
    if (entry.status === "ready") return "מוכן";
    if (entry.status === "downloading") return "מוריד…";
    return "לא הורד";
  }
  return null;
}

export function markCloudHealthChecked(): void {
  writeStorage(CLOUD_HEALTH_CHECKED_KEY, Date.now());
}

export function hasCloudHealthBeenChecked(): boolean {
  return readStorage<number>(CLOUD_HEALTH_CHECKED_KEY) != null;
}

function isFreeHfRackEntry(r: RackModelEntry): boolean {
  return r.source === "hf-scan" && r.status === "ready" && r.hfAccessMode === "FREE";
}

function isFreeHfSpaceEntry(r: RackModelEntry): boolean {
  return r.source === "hf-space" && r.status === "ready" && r.hfAccessMode === "FREE";
}

function sanitizeStoredScanned(rows: RackModelEntry[]): RackModelEntry[] {
  return rows.filter(
    (r) =>
      r.id !== "sd-turbo-local" &&
      isPickableRackEntry(r) &&
      ((r.source === "cloud-scan" && r.status === "ready") ||
        isFreeHfRackEntry(r) ||
        isFreeHfSpaceEntry(r)),
  );
}

function ensureCloudFallbacks(byId: Map<string, RackModelEntry>, scanned: RackModelEntry[]): void {
  const hasCloudInScan = scanned.some((r) => r.source === "cloud-scan");
  if (hasCloudInScan || hasCloudHealthBeenChecked()) return;
  for (const fallback of DEFAULT_CLOUD_FALLBACKS) {
    if (!byId.has(fallback.id)) byId.set(fallback.id, { ...fallback });
  }
}

export function mergeWithBuiltinRack(scanned: RackModelEntry[]): RackModelEntry[] {
  const byId = new Map<string, RackModelEntry>();
  for (const b of BUILTIN_MODELS) byId.set(b.id, { ...b });
  for (const d of getDownloadableTextBuiltins()) {
    if (!byId.has(d.id)) byId.set(d.id, { ...d });
  }

  const sortedScanned = sanitizeStoredScanned(scanned).sort((a, b) => b.addedAt - a.addedAt);
  for (const row of sortedScanned) {
    if (byId.has(row.id) && byId.get(row.id)?.source === "builtin") continue;
    byId.set(row.id, row);
  }

  ensureCloudFallbacks(byId, sortedScanned);

  const builtins = [...BUILTIN_MODELS, ...getDownloadableTextBuiltins()].map((b) => byId.get(b.id)!);
  const builtinIds = new Set(builtins.map((b) => b.id));
  const extras = [...byId.values()].filter((r) => !builtinIds.has(r.id));
  extras.sort((a, b) => b.addedAt - a.addedAt);
  const merged = [...builtins, ...extras];
  return applyLocalTextDownloadStates(merged.filter(isPickerRackEntry));
}

export function loadModelRack(): RackModelEntry[] {
  const stored = readStoredScanned();
  const cleaned = stored.length ? sanitizeStoredScanned(stored) : [];
  if (stored.length && cleaned.length !== stored.length) {
    writeStorage(MODEL_RACK_STORAGE_KEY, cleaned);
  }
  return mergeWithBuiltinRack(cleaned);
}

export function saveScannedRackModels(scanned: RackModelEntry[]): RackModelEntry[] {
  const ready = sanitizeStoredScanned(scanned);
  writeStorage(MODEL_RACK_STORAGE_KEY, ready);
  return mergeWithBuiltinRack(ready);
}

export function upsertFreeRackModel(entry: RackModelEntry): RackModelEntry[] {
  if (entry.source === "hf-scan" && entry.hfAccessMode !== "FREE") {
    return loadModelRack();
  }
  if (entry.source === "hf-space" && entry.hfAccessMode !== "FREE") {
    return loadModelRack();
  }
  if (
    (entry.source === "hf-scan" || entry.source === "cloud-scan") &&
    !isPickableRackEntry(entry)
  ) {
    return loadModelRack();
  }
  const current = loadModelRack();
  const stored = sanitizeStoredScanned(readStoredScanned());
  const idx = stored.findIndex((r) => r.id === entry.id);
  const nextStored = idx >= 0 ? stored.map((r, i) => (i === idx ? entry : r)) : [...stored, entry];
  saveScannedRackModels(nextStored);
  const idxAll = current.findIndex((r) => r.id === entry.id);
  if (idxAll >= 0) return current.map((r, i) => (i === idxAll ? entry : r));
  return [...current, entry];
}

export function isChatRackEntry(entry: RackModelEntry): boolean {
  return (
    entry.modality === "text" &&
    (entry.adapter === "gemma-local" || entry.adapter === "hf-local-text")
  );
}

/** Header picker: chat models only (image gen is in-chat via settings). */
export function resolveChatModelRackId(rack?: RackModelEntry[]): string {
  const r = rack ?? loadModelRack();
  const stored = readStorage<string>(SELECTED_MODEL_STORAGE_KEY);
  if (stored) {
    const entry = r.find((x) => x.id === stored);
    if (entry && isChatRackEntry(entry) && isSelectableInPicker(entry)) return stored;
  }
  const gemma = r.find((x) => x.id === GEMMA_RACK_ID);
  if (gemma) return GEMMA_RACK_ID;
  const smolReady = r.find((x) => x.adapter === "hf-local-text" && x.status === "ready");
  return smolReady?.id ?? GEMMA_RACK_ID;
}

export function getSelectedModelId(): string {
  return resolveChatModelRackId();
}

export function setSelectedModelId(id: string): void {
  writeStorage(SELECTED_MODEL_STORAGE_KEY, id);
}

export function getRackModelById(id: string, rack?: RackModelEntry[]): RackModelEntry | undefined {
  return (rack ?? loadModelRack()).find((r) => r.id === id);
}

export function modalityIcon(modality: ModelModality): string {
  if (modality === "image") return "🖼";
  if (modality === "code") return "💻";
  if (modality === "video") return "🎬";
  if (modality === "audio") return "🔊";
  if (modality === "vision") return "👁";
  return "🧠";
}

export function modalityFromPipeline(pipelineTag: string, modelId: string): ModelModality {
  const p = (pipelineTag || "").toLowerCase();
  const id = modelId.toLowerCase();
  if (p.includes("text-to-image") || p.includes("image-to-image")) return "image";
  if (p.includes("video")) return "video";
  if (p.includes("speech") || p.includes("audio") || p.includes("text-to-speech")) return "audio";
  if (
    p.includes("image-to-text") ||
    p.includes("visual-question") ||
    p.includes("image-classification") ||
    p.includes("object-detection") ||
    p.includes("depth-estimation")
  ) {
    return "vision";
  }
  if (/coder|code|starcoder|codellama|codegemma|deepseek-coder/i.test(id)) return "code";
  return "text";
}

function adapterForModality(modality: ModelModality): ModelAdapter {
  if (modality === "image") return "hf-inference-image";
  if (modality === "text" || modality === "code") return "hf-chat";
  return "hf-inference";
}

export function rackEntryFromHfHit(input: {
  modelId: string;
  category?: string;
  pipelineTag?: string;
  status: string;
  accessMode?: string;
}): RackModelEntry {
  const pipeline = input.pipelineTag || "";
  const modality = modalityFromPipeline(pipeline, input.modelId);

  const statusUpper = (input.status || "").toUpperCase();
  const access = (input.accessMode || "").toUpperCase();
  const isFreeWorking = statusUpper === "WORKING" && access === "FREE";

  const shortName = input.modelId.split("/").pop() || input.modelId;

  return {
    id: rackIdFromHfModel(input.modelId),
    label: shortName,
    modality,
    adapter: adapterForModality(modality),
    status: isFreeWorking ? "ready" : "token_required",
    source: "hf-scan",
    hfModelId: input.modelId,
    hfAccessMode: isFreeWorking ? "FREE" : undefined,
    pipelineTag: pipeline || input.pipelineTag,
    addedAt: Date.now(),
  };
}
