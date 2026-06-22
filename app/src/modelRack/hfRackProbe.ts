import {
  STATUS_PROVIDER_REQUIRED,
  STATUS_WORKING,
  probeHfModelInBrowser,
} from "../webSearch/hf/hfModelProbe";
import { testModelViaScanner } from "../webSearch/hf/hfApiScannerClient";
import { proxyAwareFetch } from "../webSearch/proxyFetch";

export type HfProbeKind = "chat" | "image" | "json";

export type HfRackProbeResult = {
  ok: boolean;
  accessMode: "FREE" | "TOKEN" | "UNKNOWN";
};

/** Rack picker: only models that respond without any HF token. */
export function isHfRackFreeEligible(status: string, accessMode?: string): boolean {
  const statusUpper = (status || "").toUpperCase();
  const access = (accessMode || "").toUpperCase();
  return statusUpper === STATUS_WORKING && access === "FREE";
}

async function probeImageNoToken(modelId: string): Promise<HfRackProbeResult> {
  const endpoints = [
    `https://router.huggingface.co/hf-inference/models/${encodeURIComponent(modelId)}`,
    `https://api-inference.huggingface.co/models/${encodeURIComponent(modelId)}`,
  ];
  const body = JSON.stringify({ inputs: "a photo of a red apple" });

  for (const endpoint of endpoints) {
    try {
      const response = await proxyAwareFetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
      });
      if (response.status === 401 || response.status === 403 || response.status === 503) continue;
      if (!response.ok) continue;
      const blob = await response.blob();
      if (blob.size > 80) return { ok: true, accessMode: "FREE" };
    } catch {
      /* try next endpoint */
    }
  }
  return { ok: false, accessMode: "UNKNOWN" };
}

async function probeJsonNoToken(modelId: string): Promise<HfRackProbeResult> {
  const endpoint = `https://api-inference.huggingface.co/models/${encodeURIComponent(modelId)}`;
  try {
    const response = await proxyAwareFetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ inputs: "Hello" }),
    });
    if (response.status === 401 || response.status === 403 || response.status === 503) {
      return { ok: false, accessMode: "UNKNOWN" };
    }
    if (!response.ok) return { ok: false, accessMode: "UNKNOWN" };
    const text = await response.text();
    if (!text.trim()) return { ok: false, accessMode: "UNKNOWN" };
    const lower = text.toLowerCase();
    if (lower.includes("hf_token") || lower.includes("authorization")) {
      return { ok: false, accessMode: "UNKNOWN" };
    }
    return { ok: true, accessMode: "FREE" };
  } catch {
    return { ok: false, accessMode: "UNKNOWN" };
  }
}

/**
 * Live probe for rack — never sends HF token.
 * Only models with WORKING + FREE (zero credentials) are eligible.
 */
export async function probeHfModelForRack(
  modelId: string,
  kind: HfProbeKind,
): Promise<HfRackProbeResult> {
  const scanner = await testModelViaScanner(modelId, undefined);
  if (scanner && isHfRackFreeEligible(scanner.status, scanner.accessMode)) {
    return { ok: true, accessMode: "FREE" };
  }
  if (scanner?.status === STATUS_PROVIDER_REQUIRED) {
    return { ok: false, accessMode: "UNKNOWN" };
  }

  if (kind === "chat") {
    const probe = await probeHfModelInBrowser(modelId, undefined);
    if (probe.status === STATUS_PROVIDER_REQUIRED) {
      return { ok: false, accessMode: "UNKNOWN" };
    }
    if (isHfRackFreeEligible(probe.status, probe.accessMode)) {
      return { ok: true, accessMode: "FREE" };
    }
    return { ok: false, accessMode: probe.accessMode === "TOKEN" ? "TOKEN" : "UNKNOWN" };
  }

  if (kind === "image") return probeImageNoToken(modelId);
  return probeJsonNoToken(modelId);
}

/** @deprecated use probeHfModelForRack */
export async function probeHfModelFree(modelId: string, kind: HfProbeKind): Promise<boolean> {
  const result = await probeHfModelForRack(modelId, kind);
  return result.ok && result.accessMode === "FREE";
}

export async function mapWithConcurrency<T, R>(
  items: T[],
  concurrency: number,
  fn: (item: T) => Promise<R>,
): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let next = 0;
  async function worker() {
    while (next < items.length) {
      const i = next++;
      results[i] = await fn(items[i]);
    }
  }
  const workers = Math.min(concurrency, Math.max(1, items.length));
  await Promise.all(Array.from({ length: workers }, () => worker()));
  return results;
}
