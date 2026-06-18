/** Hugging Face model row enriched with API probe / scanner metadata. */

export const HF_INFERENCE_CHAT_URL = "https://router.huggingface.co/v1/chat/completions";

export type HfAccessMode = "FREE" | "TOKEN" | "UNKNOWN";

export type HfProbeSource = "scanner" | "browser" | "none";

export type HfModelSerpHit = {
  id: string;
  modelId: string;
  url: string;
  title: string;
  snippet: string;
  pipelineTag?: string;
  category?: string;
  organization?: string;
  sizeParam?: string;
  downloads?: number;
  likes?: number;
  status: string;
  provider: string;
  accessMode: HfAccessMode;
  latency?: number;
  endpoint: string;
  curlSnippet: string;
  pythonSnippet: string;
  probed: boolean;
  probeSource: HfProbeSource;
  errorText?: string;
};

export type HfProbeResult = {
  modelId: string;
  status: string;
  provider: string;
  accessMode: HfAccessMode;
  latency?: number;
  endpoint: string;
  errorText?: string;
  testResponse?: string;
};

export type HfHubModelSummary = {
  id?: string;
  modelId?: string;
  downloads?: number;
  likes?: number;
  pipeline_tag?: string;
  tags?: string[];
  library_name?: string;
};

export function hubModelId(m: HfHubModelSummary): string {
  return (m.id ?? m.modelId ?? "").trim();
}
