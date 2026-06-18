import { fetchJson } from "../fetchJson";
import { proxyAwareFetch } from "../proxyFetch";
import { getHfToken } from "./hfModelSettings";
import { HF_INFERENCE_CHAT_URL, type HfAccessMode, type HfProbeResult } from "./hfModelTypes";

export const STATUS_WORKING = "WORKING";
export const STATUS_LOADING = "LOADING";
export const STATUS_PROVIDER_REQUIRED = "PROVIDER REQUIRED";
export const STATUS_RATE_LIMITED = "RATE LIMITED";
export const STATUS_MODEL_NOT_SUPPORTED = "MODEL NOT SUPPORTED";
export const STATUS_ERROR = "ERROR";

const PROVIDER_HF = "HF inference";
const PROVIDER_TOGETHER = "Together";
const PROVIDER_GROQ = "Groq";
const PROVIDER_FIREWORKS = "Fireworks";
const PROVIDER_UNKNOWN = "Unknown";

export function classifyModelStatus(statusCode: number, bodyText: string): string {
  const text = (bodyText || "").toLowerCase();
  if (statusCode === 410 || text.includes("no longer supported")) return "DEPRECATED ENDPOINT";
  if (text.includes("model_not_supported") || text.includes("is not a chat model")) {
    return STATUS_MODEL_NOT_SUPPORTED;
  }
  if (
    text.includes("pass a hf_token") ||
    text.includes("create a hf account") ||
    text.includes("login to your existing account")
  ) {
    return STATUS_PROVIDER_REQUIRED;
  }
  if (statusCode === 429 || text.includes("rate limit")) return STATUS_RATE_LIMITED;
  if (statusCode === 503 && text.includes("load")) return STATUS_LOADING;
  if (statusCode === 401 || statusCode === 403) return STATUS_PROVIDER_REQUIRED;
  if (statusCode === 400) return STATUS_MODEL_NOT_SUPPORTED;
  if (statusCode >= 200 && statusCode < 300 && (bodyText || "").trim()) return STATUS_WORKING;
  return STATUS_ERROR;
}

export function detectInferenceProvider(endpoint: string, headers: Record<string, string> = {}): string {
  const safeEndpoint = (endpoint || "").toLowerCase();
  const normalized = Object.fromEntries(
    Object.entries(headers).map(([k, v]) => [k.toLowerCase(), String(v).toLowerCase()]),
  );
  if ("x-compute-type" in normalized) return PROVIDER_HF;
  const haystack = [safeEndpoint, ...Object.keys(normalized), ...Object.values(normalized)].join(" ");
  if (haystack.includes("together")) return PROVIDER_TOGETHER;
  if (haystack.includes("groq")) return PROVIDER_GROQ;
  if (haystack.includes("fireworks")) return PROVIDER_FIREWORKS;
  if (
    haystack.includes("router.huggingface.co") ||
    haystack.includes("api-inference.huggingface.co") ||
    haystack.includes("huggingface")
  ) {
    return PROVIDER_HF;
  }
  return PROVIDER_UNKNOWN;
}

function accessModeFromToken(token?: string): HfAccessMode {
  return token?.trim() ? "TOKEN" : "FREE";
}

const PROBE_TIMEOUT_MS = 7_000;

/** Live probe via HF router (browser / dev proxy). */
export async function probeHfModelInBrowser(modelId: string, hfToken?: string): Promise<HfProbeResult> {
  const token = hfToken ?? getHfToken();
  const endpoint = HF_INFERENCE_CHAT_URL;
  const started = performance.now();
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token) headers.Authorization = `Bearer ${token}`;
  const controller = new AbortController();
  const timer = globalThis.setTimeout(() => controller.abort(), PROBE_TIMEOUT_MS);

  try {
    const response = await proxyAwareFetch(endpoint, {
      method: "POST",
      headers,
      signal: controller.signal,
      body: JSON.stringify({
        model: modelId,
        messages: [{ role: "user", content: "Reply with the word OK" }],
        max_tokens: 10,
      }),
    });
    const bodyText = await response.text();
    const latency = Math.round((performance.now() - started)) / 1000;
    const responseHeaders: Record<string, string> = {};
    response.headers.forEach((v, k) => {
      responseHeaders[k] = v;
    });
    return {
      modelId,
      status: classifyModelStatus(response.status, bodyText),
      provider: detectInferenceProvider(endpoint, responseHeaders),
      accessMode: accessModeFromToken(token),
      latency,
      endpoint,
      errorText: classifyModelStatus(response.status, bodyText) === STATUS_WORKING ? undefined : bodyText.slice(0, 260),
    };
  } catch (err) {
    return {
      modelId,
      status: STATUS_ERROR,
      provider: PROVIDER_UNKNOWN,
      accessMode: accessModeFromToken(token),
      latency: Math.round((performance.now() - started)) / 1000,
      endpoint,
      errorText: err instanceof Error ? err.message.slice(0, 260) : "Probe failed",
    };
  } finally {
    globalThis.clearTimeout(timer);
  }
}

/** Fetch hub metadata for a single model (no probe). */
export async function fetchHubModelMeta(modelId: string): Promise<{
  pipeline_tag?: string;
  downloads?: number;
  likes?: number;
  tags?: string[];
} | null> {
  try {
    return await fetchJson(`https://huggingface.co/api/models/${encodeURIComponent(modelId)}`, {
      headers: { "User-Agent": "GROVEEMODEL/1.0" },
    }, { timeoutMs: 12_000 });
  } catch {
    return null;
  }
}
