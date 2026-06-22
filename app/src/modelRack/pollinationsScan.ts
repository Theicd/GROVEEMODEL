import { buildPollinationsUrl } from "../cloudImage";
import { proxyAwareFetch } from "../webSearch/proxyFetch";
import { pollinationsDisplayName } from "./modelRackDisplay";
import type { RackModelEntry } from "./modelRack";
import { rackIdFromPollinations } from "./modelRack";

const PROBE_PROMPT = "red apple";
const PROBE_WIDTH = 256;
const PROBE_HEIGHT = 256;

/** All cloud image models use this host. */
export const POLLINATIONS_API_HOST = "image.pollinations.ai";

/** Stable trio — probed on every startup health check. */
export const CORE_CLOUD_IMAGE_MODELS = ["flux", "turbo", "sdxl"] as const;

export async function probePollinationsModel(model: string): Promise<boolean> {
  const url = buildPollinationsUrl({
    prompt: PROBE_PROMPT,
    model,
    width: PROBE_WIDTH,
    height: PROBE_HEIGHT,
    noLogo: true,
  });
  try {
    const response = await proxyAwareFetch(url, { method: "GET" });
    if (!response.ok) return false;
    const contentType = (response.headers.get("content-type") || "").toLowerCase();
    if (!contentType.includes("image")) return false;
    const blob = await response.blob();
    return blob.size > 80;
  } catch {
    return false;
  }
}

export function pollinationsEntry(model: string): RackModelEntry {
  return {
    id: rackIdFromPollinations(model),
    label: pollinationsDisplayName(model),
    modality: "image",
    adapter: "pollinations",
    status: "ready",
    source: "cloud-scan",
    pollinationsModel: model,
    pipelineTag: "text-to-image",
    addedAt: Date.now(),
  };
}

/** Live health check for the 3 core cloud image models. */
export async function scanCoreCloudImageModels(
  onProgress?: (model: string, ok: boolean, found: number) => void,
): Promise<RackModelEntry[]> {
  const found: RackModelEntry[] = [];
  for (const model of CORE_CLOUD_IMAGE_MODELS) {
    const ok = await probePollinationsModel(model);
    onProgress?.(model, ok, found.length);
    if (ok) found.push(pollinationsEntry(model));
  }
  return found;
}
