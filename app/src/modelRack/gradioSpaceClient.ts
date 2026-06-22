import { proxyAwareFetch } from "../webSearch/proxyFetch";

export type GradioParam = {
  parameter_name?: string;
  parameter_default?: unknown;
  example_input?: unknown;
  type?: { type?: string };
};

export type GradioNamedEndpoint = {
  parameters?: GradioParam[];
};

export type GradioInfo = {
  named_endpoints?: Record<string, GradioNamedEndpoint>;
};

export function spaceIdToHost(spaceId: string): string {
  return `${spaceId.trim().toLowerCase().replace(/\//g, "-")}.hf.space`;
}

export function gradioCallUrl(host: string, endpoint: string, eventId?: string): string {
  const ep = endpoint.startsWith("/") ? endpoint : `/${endpoint}`;
  const base = `https://${host}/gradio_api/call${ep}`;
  return eventId ? `${base}/${eventId}` : base;
}

export function defaultValueForParam(p: GradioParam): unknown {
  if (p.example_input !== undefined && p.example_input !== null) return p.example_input;
  if (p.parameter_default !== undefined && p.parameter_default !== null) return p.parameter_default;
  const t = p.type?.type;
  if (t === "boolean") return false;
  if (t === "number") return 0;
  if (t === "string") return "test";
  return null;
}

/** Build Gradio `data` array; first string slot gets the probe/generation prompt. */
export function buildGradioData(params: GradioParam[], prompt: string): unknown[] {
  let promptUsed = false;
  return params.map((p) => {
    const t = p.type?.type;
    if (!promptUsed && t === "string") {
      promptUsed = true;
      return prompt;
    }
    return defaultValueForParam(p);
  });
}

export function pickGradioEndpoint(
  info: GradioInfo,
  preferImage: boolean,
): { endpoint: string; parameters: GradioParam[] } | null {
  const named = info.named_endpoints ?? {};
  const preferred = ["/infer", "/predict", "/generate", "/text2img", "/chat"];
  const entries = Object.entries(named);
  if (!entries.length) return null;

  const score = (name: string, params: GradioParam[]): number => {
    const lower = name.toLowerCase();
    let s = 0;
    if (preferred.some((p) => lower.includes(p.replace("/", "")))) s += 10;
    if (params.some((p) => p.type?.type === "string")) s += 5;
    if (preferImage && params.length >= 2) s += 2;
    return s;
  };

  const sorted = entries
    .map(([endpoint, cfg]) => ({
      endpoint: endpoint.startsWith("/") ? endpoint : `/${endpoint}`,
      parameters: cfg.parameters ?? [],
      s: score(endpoint, cfg.parameters ?? []),
    }))
    .sort((a, b) => b.s - a.s);

  const best = sorted[0];
  if (!best?.parameters.length) return null;
  return { endpoint: best.endpoint, parameters: best.parameters };
}

export function parseGradioSseData(text: string): unknown[] | null {
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed.startsWith("data:")) continue;
    const payload = trimmed.slice(5).trim();
    if (!payload || payload === "null") continue;
    try {
      const parsed = JSON.parse(payload) as unknown;
      if (Array.isArray(parsed)) return parsed;
    } catch {
      /* next line */
    }
  }
  return null;
}

export function resultLooksLikeImage(data: unknown[]): boolean {
  for (const item of data) {
    if (typeof item === "string" && /^https?:\/\//i.test(item) && /\.(png|jpg|jpeg|webp|gif)/i.test(item)) {
      return true;
    }
    if (item && typeof item === "object") {
      const o = item as Record<string, unknown>;
      const url = String(o.url ?? "");
      if (url.includes("/gradio_api/file") || /\.(png|jpg|jpeg|webp|gif)/i.test(url)) return true;
    }
  }
  return false;
}

export function resultLooksLikeText(data: unknown[]): boolean {
  for (const item of data) {
    if (typeof item === "string" && item.trim().length > 2 && !item.startsWith("http")) return true;
  }
  return false;
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

export async function fetchGradioInfo(host: string): Promise<GradioInfo | null> {
  try {
    const response = await proxyAwareFetch(`https://${host}/gradio_api/info`, { method: "GET" });
    if (!response.ok) return null;
    return (await response.json()) as GradioInfo;
  } catch {
    return null;
  }
}

export async function submitGradioCall(
  host: string,
  endpoint: string,
  data: unknown[],
  hfToken?: string,
): Promise<string | null> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (hfToken) headers.Authorization = `Bearer ${hfToken}`;
  try {
    const response = await proxyAwareFetch(gradioCallUrl(host, endpoint), {
      method: "POST",
      headers,
      body: JSON.stringify({ data }),
    });
    if (!response.ok) return null;
    const json = (await response.json()) as { event_id?: string };
    return json.event_id ?? null;
  } catch {
    return null;
  }
}

export async function pollGradioCall(
  host: string,
  endpoint: string,
  eventId: string,
  hfToken?: string,
  maxWaitMs = 90_000,
): Promise<unknown[] | null> {
  const headers: Record<string, string> = {};
  if (hfToken) headers.Authorization = `Bearer ${hfToken}`;
  const started = Date.now();
  while (Date.now() - started < maxWaitMs) {
    try {
      const response = await proxyAwareFetch(gradioCallUrl(host, endpoint, eventId), {
        method: "GET",
        headers,
      });
      const text = await response.text();
      if (text.includes("event: error")) return null;
      if (text.includes("event: complete")) {
        return parseGradioSseData(text);
      }
    } catch {
      /* retry */
    }
    await sleep(1200);
  }
  return null;
}

export async function runGradioPredict(
  host: string,
  endpoint: string,
  data: unknown[],
  hfToken?: string,
): Promise<unknown[] | null> {
  const eventId = await submitGradioCall(host, endpoint, data, hfToken);
  if (!eventId) return null;
  return pollGradioCall(host, endpoint, eventId, hfToken);
}

export function extractImageUrlFromGradioResult(data: unknown[]): string | null {
  for (const item of data) {
    if (typeof item === "string" && item.startsWith("http")) return item;
    if (item && typeof item === "object") {
      const url = (item as { url?: string }).url;
      if (url?.startsWith("http")) return url;
    }
  }
  return null;
}

export function extractTextFromGradioResult(data: unknown[]): string | null {
  for (const item of data) {
    if (typeof item === "string" && item.trim() && !item.startsWith("http")) return item.trim();
  }
  return null;
}
