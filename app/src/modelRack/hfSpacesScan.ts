import { proxyAwareFetch } from "../webSearch/proxyFetch";
import { getHfToken } from "../webSearch/hf/hfModelSettings";
import type { ModelModality, RackModelEntry } from "./modelRack";
import { modalityFromPipeline, rackIdFromHfSpace } from "./modelRack";
import {
  buildGradioData,
  fetchGradioInfo,
  pickGradioEndpoint,
  resultLooksLikeImage,
  resultLooksLikeText,
  runGradioPredict,
  spaceIdToHost,
} from "./gradioSpaceClient";

export type HfSpaceSummary = {
  id?: string;
  likes?: number;
  private?: boolean;
  sdk?: string;
  tags?: string[];
  cardData?: { title?: string; short_description?: string };
};

/** Known public Gradio spaces (ZeroGPU / popular) — probed first. */
export const CURATED_HF_SPACES: { spaceId: string; modality: ModelModality; pipelineTag?: string }[] = [
  { spaceId: "black-forest-labs/FLUX.1-schnell", modality: "image", pipelineTag: "text-to-image" },
  { spaceId: "stabilityai/stable-diffusion", modality: "image", pipelineTag: "text-to-image" },
  { spaceId: "multimodalart/stable-diffusion", modality: "image", pipelineTag: "text-to-image" },
  { spaceId: "zerogpu-aoti/wan2-2-fp8da-aoti-faster", modality: "video", pipelineTag: "text-to-video" },
  { spaceId: "microsoft/HuggingGPT", modality: "text", pipelineTag: "text-generation" },
];

const SPACE_SEARCH_QUERIES: { q: string; modality: ModelModality; pipelineTag: string }[] = [
  { q: "flux text-to-image zerogpu", modality: "image", pipelineTag: "text-to-image" },
  { q: "stable diffusion gradio", modality: "image", pipelineTag: "text-to-image" },
  { q: "llm chat gradio", modality: "text", pipelineTag: "text-generation" },
  { q: "text-to-video zerogpu", modality: "video", pipelineTag: "text-to-video" },
];

const MAX_SPACE_PROBES = 12;
const PROBE_PROMPT = "red apple";

export function spaceEntryLabel(spaceId: string, title?: string): string {
  const short = spaceId.split("/").pop() || spaceId;
  if (title && title !== short) return `${title}`;
  return short;
}

export function hfSpaceRackEntry(input: {
  spaceId: string;
  modality: ModelModality;
  endpoint: string;
  probeData: unknown[];
  pipelineTag?: string;
  title?: string;
}): RackModelEntry {
  return {
    id: rackIdFromHfSpace(input.spaceId),
    label: `${spaceEntryLabel(input.spaceId, input.title)} (HF Space)`,
    modality: input.modality,
    adapter: "hf-gradio-space",
    status: "ready",
    source: "hf-space",
    hfSpaceId: input.spaceId,
    gradioEndpoint: input.endpoint,
    gradioProbeData: input.probeData,
    pipelineTag: input.pipelineTag,
    hfAccessMode: "FREE",
    addedAt: Date.now(),
  };
}

async function fetchHubSpaces(search: string, limit: number): Promise<HfSpaceSummary[]> {
  const url = `https://huggingface.co/api/spaces?search=${encodeURIComponent(search)}&limit=${limit}&sort=likes&direction=-1`;
  try {
    const response = await proxyAwareFetch(url, {
      headers: { Accept: "application/json", "User-Agent": "GROVEEMODEL/1.0" },
    });
    if (!response.ok) return [];
    const data = (await response.json()) as HfSpaceSummary[];
    return Array.isArray(data) ? data : [];
  } catch {
    return [];
  }
}

function spaceCandidatesFromHub(rows: HfSpaceSummary[]): string[] {
  return rows
    .filter((s) => s.id && !s.private && (s.sdk === "gradio" || (s.tags ?? []).includes("gradio")))
    .map((s) => s.id!)
    .filter((id, i, arr) => arr.indexOf(id) === i);
}

export async function probeHfGradioSpace(
  spaceId: string,
  modality: ModelModality,
  pipelineTag?: string,
  title?: string,
): Promise<RackModelEntry | null> {
  const host = spaceIdToHost(spaceId);
  const info = await fetchGradioInfo(host);
  if (!info) return null;

  const picked = pickGradioEndpoint(info, modality === "image" || modality === "video");
  if (!picked) return null;

  const probeData = buildGradioData(picked.parameters, PROBE_PROMPT);
  const token = getHfToken();
  const result = await runGradioPredict(host, picked.endpoint, probeData, token);
  if (!result?.length) return null;

  const okImage = modality === "image" || modality === "video";
  if (okImage && !resultLooksLikeImage(result)) return null;
  if (!okImage && !resultLooksLikeText(result)) return null;

  return hfSpaceRackEntry({
    spaceId,
    modality,
    endpoint: picked.endpoint,
    probeData,
    pipelineTag: pipelineTag ?? (okImage ? "text-to-image" : "text-generation"),
    title,
  });
}

export async function scanHfGradioSpaces(
  onProgress?: (spaceId: string, ok: boolean, found: number) => void,
): Promise<RackModelEntry[]> {
  const found: RackModelEntry[] = [];
  const seen = new Set<string>();
  let probes = 0;

  const trySpace = async (
    spaceId: string,
    modality: ModelModality,
    pipelineTag?: string,
    title?: string,
  ) => {
    if (probes >= MAX_SPACE_PROBES || seen.has(spaceId)) return;
    seen.add(spaceId);
    probes++;
    const entry = await probeHfGradioSpace(spaceId, modality, pipelineTag, title);
    onProgress?.(spaceId, !!entry, found.length);
    if (entry) found.push(entry);
  };

  for (const c of CURATED_HF_SPACES) {
    if (probes >= MAX_SPACE_PROBES) break;
    await trySpace(c.spaceId, c.modality, c.pipelineTag);
  }

  for (const sq of SPACE_SEARCH_QUERIES) {
    if (probes >= MAX_SPACE_PROBES) break;
    const hub = await fetchHubSpaces(sq.q, 8);
    for (const spaceId of spaceCandidatesFromHub(hub)) {
      if (probes >= MAX_SPACE_PROBES) break;
      const meta = hub.find((h) => h.id === spaceId);
      const modality =
        sq.modality === "image" && /video/i.test(spaceId + (meta?.cardData?.title ?? ""))
          ? "video"
          : sq.modality;
      await trySpace(spaceId, modality, sq.pipelineTag, meta?.cardData?.title);
    }
  }

  return found;
}

export function modalityForSpaceId(spaceId: string, pipelineTag?: string): ModelModality {
  if (pipelineTag) return modalityFromPipeline(pipelineTag, spaceId);
  const id = spaceId.toLowerCase();
  if (/video|wan|ltx/i.test(id)) return "video";
  if (/image|flux|sdxl|diffusion|txt2img/i.test(id)) return "image";
  if (/coder|code/i.test(id)) return "code";
  return "text";
}
