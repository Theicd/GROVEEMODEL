// @ts-nocheck
/// <reference lib="webworker" />

import { InterruptableStoppingCriteria, pipeline, env, TextStreamer } from "@huggingface/transformers";

env.allowLocalModels = false;
env.useBrowserCache = true;

type ChatTurn = { role: "user" | "assistant"; content: string };

type HfProgress = {
  status?: string;
  file?: string;
  progress?: number;
  loaded?: number;
  total?: number;
};

const post = (data: unknown) => self.postMessage(data);

const bootLog = (step: string, detail?: Record<string, unknown>) => {
  if (detail !== undefined) console.info("[GROVEE:boot]", step, detail);
  else console.info("[GROVEE:boot]", step);
};

const generators = new Map<string, Awaited<ReturnType<typeof pipeline>>>();
const loading = new Map<string, Promise<Awaited<ReturnType<typeof pipeline>>>>();
let activeModelId: string | null = null;
let chatBusy = false;
let abortRequested = false;
let activeInterrupt: InterruptableStoppingCriteria | null = null;

function progressFromHf(p: HfProgress): { pct: number; loaded: number; total: number; message: string } {
  const loaded = typeof p.loaded === "number" ? p.loaded : 0;
  const total = typeof p.total === "number" ? p.total : 0;
  let pct = 0;
  if (loaded > 0 && total > 0) {
    pct = Math.round((loaded / total) * 100);
  } else if (typeof p.progress === "number") {
    pct = p.progress <= 1 ? Math.round(p.progress * 100) : Math.round(p.progress);
  }
  const file = p.file ? (p.file.split("/").pop() ?? p.file) : "";
  const message = file ? `מוריד ${file}` : "טוען משקולות…";
  return { pct, loaded, total, message };
}

function makeProgressCallback(modelId: string) {
  return (p: HfProgress) => {
    const st = p.status ?? "";
    if (st === "progress" || st === "progress_total" || st === "download") {
      const { pct, loaded, total, message } = progressFromHf(p);
      post({ type: "progress", modelId, message, pct, loaded, total });
    }
    if (st === "initiate") {
      post({ type: "progress", modelId, message: "מכין קבצים…", pct: 1, loaded: 0, total: 0 });
    }
  };
}

let webGpuAdapterProbe: boolean | null = null;

const hasRunnableWebGpuAdapter = async (): Promise<boolean> => {
  if (webGpuAdapterProbe !== null) return webGpuAdapterProbe;
  try {
    const g = (
      self as unknown as {
        navigator?: { gpu?: { requestAdapter?: (opts?: object) => Promise<unknown | null> } };
      }
    ).navigator?.gpu;
    if (!g?.requestAdapter) {
      webGpuAdapterProbe = false;
      bootLog("SmolLM worker: navigator.gpu missing");
      return false;
    }
    const adapter = await g.requestAdapter({ powerPreference: "low-power" });
    webGpuAdapterProbe = adapter != null;
    bootLog("SmolLM worker: requestAdapter", {
      adapter: webGpuAdapterProbe ? "found" : "null",
      powerPreference: "low-power",
    });
    return webGpuAdapterProbe;
  } catch (err) {
    webGpuAdapterProbe = false;
    bootLog("SmolLM worker: requestAdapter threw", {
      error: err instanceof Error ? err.message : String(err),
    });
    return false;
  }
};

async function loadOnDevice(modelId: string, device: "webgpu" | "wasm") {
  post({ type: "status", modelId, text: `טוען ${modelId} (${device})…` });
  const pipe = await pipeline("text-generation", modelId, {
    device,
    dtype: "q4",
    progress_callback: makeProgressCallback(modelId),
  });
  generators.set(modelId, pipe);
  activeModelId = modelId;
  post({ type: "loaded", modelId, device });
  return pipe;
}

async function resolveDevice(
  modelId: string,
  backend: "auto" | "webgpu" | "wasm" = "auto",
): Promise<Awaited<ReturnType<typeof pipeline>>> {
  bootLog("SmolLM resolveDevice", { modelId, backend });
  const tryWasm = () => loadOnDevice(modelId, "wasm");
  const tryWebGpuWithFallback = async () => {
    try {
      return await loadOnDevice(modelId, "webgpu");
    } catch (err) {
      bootLog("SmolLM WebGPU load failed — WASM fallback", {
        error: err instanceof Error ? err.message : String(err),
      });
      post({ type: "status", modelId, text: "WebGPU נכשל — עובר ל-WASM…" });
      return tryWasm();
    }
  };

  if (backend === "wasm") {
    bootLog("SmolLM using WASM (settings)");
    return tryWasm();
  }
  if (backend === "webgpu") {
    if (await hasRunnableWebGpuAdapter()) {
      bootLog("SmolLM trying WebGPU (settings)");
      return tryWebGpuWithFallback();
    }
    bootLog("SmolLM no adapter — WASM (webgpu setting)");
    post({ type: "status", modelId, text: "אין WebGPU — טוען ב-WASM…" });
    return tryWasm();
  }
  if (await hasRunnableWebGpuAdapter()) {
    bootLog("SmolLM auto: trying WebGPU first");
    return tryWebGpuWithFallback();
  }
  bootLog("SmolLM auto: no adapter — WASM only");
  post({ type: "status", modelId, text: "אין WebGPU — טוען ב-WASM…" });
  return tryWasm();
}

async function ensureGenerator(modelId: string, backend: "auto" | "webgpu" | "wasm" = "auto") {
  const cached = generators.get(modelId);
  if (cached) {
    activeModelId = modelId;
    return cached;
  }
  const pending = loading.get(modelId);
  if (pending) return pending;

  const promise = resolveDevice(modelId, backend);

  loading.set(modelId, promise);
  try {
    return await promise;
  } finally {
    loading.delete(modelId);
  }
}

function extractReply(output: unknown): string {
  if (Array.isArray(output) && output[0]?.generated_text) {
    const gt = output[0].generated_text;
    if (Array.isArray(gt)) {
      const last = gt[gt.length - 1];
      if (last && typeof last === "object" && "content" in last) {
        return String((last as { content: string }).content).trim();
      }
    }
    if (typeof gt === "string") return gt.trim();
  }
  const text = (output as { generated_text?: string })?.generated_text;
  return typeof text === "string" ? text.trim() : "";
}

self.onmessage = async (ev: MessageEvent) => {
  const msg = ev.data as {
    type: string;
    modelId?: string;
    backend?: "auto" | "webgpu" | "wasm";
    systemPrompt?: string;
    history?: ChatTurn[];
    prompt?: string;
    maxNewTokens?: number;
    temperature?: number;
    topP?: number;
  };

  try {
    if (msg.type === "load" && msg.modelId) {
      await ensureGenerator(msg.modelId, msg.backend ?? "auto");
      return;
    }

    if (msg.type === "abort") {
      abortRequested = true;
      activeInterrupt?.interrupt();
      return;
    }

    if (msg.type === "generate" && msg.modelId && msg.prompt != null) {
      if (chatBusy) {
        post({ type: "error", error: "Generation already in progress.", scope: "chat" });
        return;
      }

      const pipe = await ensureGenerator(msg.modelId, msg.backend ?? "auto");
      chatBusy = true;
      abortRequested = false;
      const interrupt = new InterruptableStoppingCriteria();
      activeInterrupt = interrupt;

      const messages: ChatTurn[] = [
        { role: "system", content: msg.systemPrompt || "You are a helpful assistant." },
        ...(msg.history ?? []),
        { role: "user", content: msg.prompt },
      ];

      const streamer = pipe.tokenizer
        ? new TextStreamer(pipe.tokenizer, {
            skip_prompt: true,
            callback_function: (text: string) => {
              post({ type: "token", text });
            },
          })
        : undefined;

      try {
        const out = await pipe(messages, {
          max_new_tokens: msg.maxNewTokens ?? 256,
          temperature: msg.temperature ?? 0.7,
          top_p: msg.topP ?? 0.9,
          do_sample: true,
          streamer,
          stopping_criteria: interrupt,
        });
        if (abortRequested) {
          post({ type: "aborted", scope: "chat" });
          return;
        }
        const text = extractReply(out);
        post({ type: "complete", text });
      } finally {
        chatBusy = false;
        activeInterrupt = null;
      }
      return;
    }
  } catch (err) {
    chatBusy = false;
    activeInterrupt = null;
    post({
      type: "error",
      error: err instanceof Error ? err.message : String(err),
      scope: "chat",
    });
  }
};
