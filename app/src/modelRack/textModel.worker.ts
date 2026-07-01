// @ts-nocheck
/// <reference lib="webworker" />

import { InterruptableStoppingCriteria, pipeline, env, TextStreamer } from "@huggingface/transformers";

env.allowLocalModels = false;
env.useBrowserCache = true;

// Mobile / GitHub Pages hardening for ONNX Runtime Web (WASM CPU path).
// GitHub Pages cannot serve COOP/COEP headers, so `crossOriginIsolated` is false
// and SharedArrayBuffer is unavailable. Forcing single-threaded WASM (no pthreads,
// no proxy worker) prevents the silent hang that made SmolLM never finish loading
// on phones. When a host does provide cross-origin isolation, we keep multi-thread.
try {
  const onnxWasm = (env as unknown as {
    backends?: { onnx?: { wasm?: { numThreads?: number; proxy?: boolean; simd?: boolean } } };
  }).backends?.onnx?.wasm;
  if (onnxWasm) {
    const isolated =
      typeof (self as unknown as { crossOriginIsolated?: boolean }).crossOriginIsolated === "boolean"
        ? (self as unknown as { crossOriginIsolated?: boolean }).crossOriginIsolated
        : false;
    if (!isolated) onnxWasm.numThreads = 1;
    onnxWasm.proxy = false;
    onnxWasm.simd = true;
    console.info("[GROVEE:boot] SmolLM worker: wasm config", {
      numThreads: onnxWasm.numThreads,
      isolated,
    });
  }
} catch (err) {
  console.warn("[GROVEE:boot] SmolLM worker: failed to configure onnx wasm", err);
}

type ChatTurn = { role: "user" | "assistant"; content: string };

type HfProgress = {
  status?: string;
  file?: string;
  progress?: number;
  loaded?: number;
  total?: number;
};

const post = (data: unknown) => self.postMessage(data);

// Forward this worker's console to the main thread so the in-app console panel can
// show SmolLM load/inference logs on devices without DevTools (phones).
(() => {
  const levels = ["log", "info", "warn", "error"] as const;
  for (const level of levels) {
    const original = (console[level] as ((...a: unknown[]) => void) | undefined)?.bind(console);
    (console as unknown as Record<string, (...a: unknown[]) => void>)[level] = (...args: unknown[]) => {
      try {
        const text = args
          .map((a) =>
            typeof a === "string"
              ? a
              : a instanceof Error
                ? (a.stack ?? a.message)
                : (() => {
                    try {
                      return JSON.stringify(a);
                    } catch {
                      return String(a);
                    }
                  })(),
          )
          .join(" ");
        self.postMessage({ type: "__wlog", level, text });
      } catch {
        /* ignore */
      }
      original?.(...args);
    };
  }
})();

const bootLog = (step: string, detail?: Record<string, unknown>) => {
  if (detail !== undefined) console.info("[GROVEE:boot]", step, detail);
  else console.info("[GROVEE:boot]", step);
};

self.onunhandledrejection = (ev: PromiseRejectionEvent) => {
  console.error("[GROVEE:boot] SmolLM worker unhandledrejection", ev.reason);
  try {
    ev.preventDefault();
  } catch {
    /* ignore */
  }
  post({
    type: "error",
    error: ev.reason instanceof Error ? ev.reason.message : String(ev.reason),
    scope: "load",
  });
};

const generators = new Map<string, Awaited<ReturnType<typeof pipeline>>>();
const loading = new Map<string, Promise<Awaited<ReturnType<typeof pipeline>>>>();
let activeModelId: string | null = null;
let activeDevice: "webgpu" | "wasm" | null = null;
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
    const adapter = await Promise.race([
      g.requestAdapter({ powerPreference: "low-power" }),
      new Promise<null>((resolve) => setTimeout(() => resolve(null), 2500)),
    ]);
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

// Quantization fallback per device.
// - WebGPU (desktop): q4 is fast on GPU; q4f16 is a fine secondary.
// - WASM/CPU (mobile): prefer q8 (int8 dynamic → model_quantized.onnx). On a tiny
//   135M model, 4-bit weights degrade quality badly on the CPU kernel and cause the
//   model to loop / repeat a single word. int8 is both SMALLER here (~130MB vs ~174MB)
//   and noticeably more coherent, so it is the primary choice on phones.
function dtypeChainFor(device: "webgpu" | "wasm"): string[] {
  return device === "webgpu" ? ["q4", "q4f16"] : ["q8", "q4"];
}

async function loadOnDevice(modelId: string, device: "webgpu" | "wasm") {
  const chain = dtypeChainFor(device);
  let lastErr: unknown = null;
  for (let i = 0; i < chain.length; i++) {
    const dtype = chain[i];
    post({ type: "status", modelId, text: `טוען ${modelId} (${device} · ${dtype})…` });
    try {
      const pipe = await pipeline("text-generation", modelId, {
        device,
        dtype,
        progress_callback: makeProgressCallback(modelId),
      });
      generators.set(modelId, pipe);
      activeModelId = modelId;
      activeDevice = device;
      bootLog("SmolLM loaded", { modelId, device, dtype });
      post({ type: "loaded", modelId, device });
      return pipe;
    } catch (err) {
      lastErr = err;
      bootLog("SmolLM dtype failed", {
        modelId,
        device,
        dtype,
        next: chain[i + 1] ?? null,
        error: err instanceof Error ? err.message : String(err),
      });
      if (i < chain.length - 1) {
        post({ type: "status", modelId, text: `דחיסה ${dtype} נכשלה — מנסה ${chain[i + 1]}…` });
      }
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error(String(lastErr));
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
      for (let i = gt.length - 1; i >= 0; i--) {
        const entry = gt[i];
        if (entry && typeof entry === "object" && "role" in entry && "content" in entry) {
          if (String((entry as { role: string }).role) === "assistant") {
            return String((entry as { content: string }).content).trim();
          }
        }
      }
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
      post({
        type: "progress",
        modelId: msg.modelId,
        message: "מתחיל טעינה…",
        pct: 0,
        loaded: 0,
        total: 0,
      });
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

      const historyLen = msg.history?.length ?? 0;
      const promptLen = msg.prompt?.length ?? 0;
      const simpleTurn = historyLen === 0 && promptLen < 64;
      // On mobile/CPU the tiny 135M model collapses into single-word loops with the
      // near-greedy (very low temperature) decoding we use on desktop. Keep enough
      // sampling entropy there and lean harder on repetition controls.
      const onCpu = activeDevice === "wasm";
      // Tiny 135M model rambles and rarely stops on its own. Keep replies short and
      // focused: a lower temperature (safe now that repetition_penalty +
      // no_repeat_ngram guard against loops) gives more accurate, less wandering
      // answers, and we cap the length on the CPU/mobile path.
      const temperature = onCpu
        ? Math.min(msg.temperature ?? 0.4, 0.45)
        : simpleTurn
          ? Math.min(msg.temperature ?? 0.35, 0.25)
          : (msg.temperature ?? 0.35);
      const maxNewTokens = onCpu
        ? Math.min(msg.maxNewTokens ?? 160, 160)
        : (msg.maxNewTokens ?? 192);

      try {
        const out = await pipe(messages, {
          max_new_tokens: maxNewTokens,
          temperature,
          top_p: onCpu ? 0.85 : (msg.topP ?? 0.85),
          top_k: onCpu ? 40 : 50,
          // Anti-repetition: penalize already-seen tokens and forbid repeating any
          // 3-gram. This is what stops the "same word over and over" failure mode
          // on small quantized models.
          repetition_penalty: onCpu ? 1.3 : 1.15,
          no_repeat_ngram_size: 3,
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
    const scope = msg.type === "load" ? "load" : "chat";
    post({
      type: "error",
      error: err instanceof Error ? err.message : String(err),
      scope,
    });
  }
};
