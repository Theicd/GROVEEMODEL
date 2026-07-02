import type { PollinationsModelId } from "../cloudImage";

export type ImageGenSettings = {
  preferredPollinationsModel: PollinationsModelId;
};

export const IMAGE_GEN_SETTINGS_KEY = "grovee_image_gen_settings_v1";

export const DEFAULT_IMAGE_GEN_SETTINGS: ImageGenSettings = {
  preferredPollinationsModel: "flux",
};

export function mergeImageGenSettings(
  partial?: Partial<ImageGenSettings> | null,
): ImageGenSettings {
  const model = partial?.preferredPollinationsModel;
  const allowed = new Set<PollinationsModelId>(["flux", "turbo", "sdxl"]);
  return {
    preferredPollinationsModel:
      model && allowed.has(model) ? model : DEFAULT_IMAGE_GEN_SETTINGS.preferredPollinationsModel,
  };
}

export function loadImageGenSettings(): ImageGenSettings {
  if (typeof localStorage === "undefined") return { ...DEFAULT_IMAGE_GEN_SETTINGS };
  try {
    const raw = localStorage.getItem(IMAGE_GEN_SETTINGS_KEY);
    if (!raw) return { ...DEFAULT_IMAGE_GEN_SETTINGS };
    return mergeImageGenSettings(JSON.parse(raw) as Partial<ImageGenSettings>);
  } catch {
    return { ...DEFAULT_IMAGE_GEN_SETTINGS };
  }
}

export function saveImageGenSettings(settings: ImageGenSettings): void {
  if (typeof localStorage === "undefined") return;
  try {
    localStorage.setItem(IMAGE_GEN_SETTINGS_KEY, JSON.stringify(settings));
  } catch {
    /* quota */
  }
}
