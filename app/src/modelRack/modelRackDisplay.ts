import { SMOLLM_RACK_ID } from "./localTextModels";
import { GEMMA_RACK_ID, type RackModelEntry } from "./modelRack";

export type ImageModelMeta = {
  badge: string;
  title: string;
  hint: string;
  accent: string;
};

const IMAGE_META: Record<string, ImageModelMeta> = {
  flux: {
    badge: "FX",
    title: "FLUX",
    hint: "איכות גבוהה · מאוזן",
    accent: "#a78bfa",
  },
  turbo: {
    badge: "⚡",
    title: "Turbo",
    hint: "מהיר · קליל",
    accent: "#38bdf8",
  },
  sdxl: {
    badge: "XL",
    title: "SDXL",
    hint: "סגנון קלאסי",
    accent: "#f472b6",
  },
};

const IMAGE_ORDER = ["flux", "turbo", "sdxl"] as const;

export function pollinationsDisplayName(model: string): string {
  return IMAGE_META[model]?.title ?? model.charAt(0).toUpperCase() + model.slice(1);
}

export function rackPickerTitle(entry: RackModelEntry): string {
  if (entry.id === GEMMA_RACK_ID) return "Gemma 4 E2B";
  if (entry.id === SMOLLM_RACK_ID) return "SmolLM2 135M";
  if (entry.adapter === "pollinations" && entry.pollinationsModel) {
    return pollinationsDisplayName(entry.pollinationsModel);
  }
  return entry.label.replace(/\s*\(Pollinations\)\s*/i, "").trim() || entry.label;
}

export function rackPickerHint(entry: RackModelEntry): string | null {
  if (entry.id === GEMMA_RACK_ID) return "שיחה מקומית";
  if (entry.id === SMOLLM_RACK_ID) {
    if (entry.status === "ready") return "מודל קל · שיחה טקסט";
    if (entry.status === "downloading") return "מוריד…";
    return "לחץ הורדה לפני שיחה";
  }
  if (entry.adapter === "pollinations" && entry.pollinationsModel) {
    return IMAGE_META[entry.pollinationsModel]?.hint ?? "יצירת תמונה";
  }
  if (entry.source === "hf-space") return "HF Space";
  if (entry.source === "hf-scan") return "Hugging Face";
  return null;
}

export function rackPickerBadge(entry: RackModelEntry): ImageModelMeta | null {
  if (entry.adapter !== "pollinations" || !entry.pollinationsModel) return null;
  return IMAGE_META[entry.pollinationsModel] ?? null;
}

export function rackPickerShowTag(entry: RackModelEntry): string | null {
  if (entry.source === "builtin") return "מקומי";
  return null;
}

export function sortImageRackEntries(items: RackModelEntry[]): RackModelEntry[] {
  const order = new Map(IMAGE_ORDER.map((id, i) => [id, i]));
  return [...items].sort((a, b) => {
    const ai = order.get(a.pollinationsModel ?? "") ?? 99;
    const bi = order.get(b.pollinationsModel ?? "") ?? 99;
    return ai - bi;
  });
}
