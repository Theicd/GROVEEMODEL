/**
 * Pollinations cloud image URL helper.
 *
 * Why this is its own module:
 *  - It's the single source of truth for the cloud image link, so unit tests can
 *    pin the contract (encoding, model, default size).
 *  - The chat flow + the deployment health-check script both import it instead of
 *    duplicating the format, which is exactly the kind of duplication that caused
 *    earlier `image.pollinations.ai` regressions.
 */

export type PollinationsModelId = "flux" | "turbo" | "sdxl";

export const IMAGE_MODEL_OPTIONS: ReadonlyArray<{ id: PollinationsModelId; label: string }> = [
  { id: "flux", label: "FLUX (balanced)" },
  { id: "turbo", label: "Turbo (faster)" },
  { id: "sdxl", label: "SDXL style" },
] as const;

const KNOWN_MODELS = new Set<PollinationsModelId>(IMAGE_MODEL_OPTIONS.map((m) => m.id));

const DEFAULT_MODEL: PollinationsModelId = "flux";
const DEFAULT_WIDTH = 1024;
const DEFAULT_HEIGHT = 1024;

export function normalizePollinationsModel(id: string | undefined): PollinationsModelId {
  return id && (KNOWN_MODELS as Set<string>).has(id) ? (id as PollinationsModelId) : DEFAULT_MODEL;
}

export interface BuildPollinationsUrlOptions {
  prompt: string;
  model?: string;
  width?: number;
  height?: number;
  noLogo?: boolean;
}

/**
 * Build the public Pollinations image URL: `https://image.pollinations.ai/prompt/<encoded>?...`.
 *
 * Always uses `encodeURIComponent` so quotes, Hebrew, emoji, slashes, and `&` cannot break the URL.
 * Throws on empty prompt so the caller never sends `https://image.pollinations.ai/prompt/?...`.
 */
export function buildPollinationsUrl(options: BuildPollinationsUrlOptions): string {
  const prompt = (options.prompt ?? "").trim();
  if (!prompt) throw new Error("buildPollinationsUrl: prompt is required");

  const model = normalizePollinationsModel(options.model);
  const width = Number.isFinite(options.width) && (options.width as number) > 0 ? Math.floor(options.width as number) : DEFAULT_WIDTH;
  const height = Number.isFinite(options.height) && (options.height as number) > 0 ? Math.floor(options.height as number) : DEFAULT_HEIGHT;
  const noLogo = options.noLogo === false ? false : true;

  const params = new URLSearchParams();
  params.set("width", String(width));
  params.set("height", String(height));
  params.set("model", model);
  if (noLogo) params.set("nologo", "true");

  return `https://image.pollinations.ai/prompt/${encodeURIComponent(prompt)}?${params.toString()}`;
}
