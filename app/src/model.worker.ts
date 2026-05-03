/// <reference lib="webworker" />

import { TextStreamer, env, pipeline } from "@huggingface/transformers";

type InferenceBackend = "auto" | "webgpu" | "wasm";

type LoadMessage = {
  type: "load";
  modelId: string;
  dtype: "q4" | "q8" | "fp16" | "fp32";
};

type GenerateMessage = {
  type: "generate";
  modelId: string;
  prompt: string;
  systemPrompt: string;
  maxNewTokens: number;
  temperature: number;
  repetitionPenalty: number;
  topP: number;
  thinkingMode: boolean;
  webContext: string;
};

type CaptionMessage = {
  type: "caption";
  imageDataUrl: string;
  prompt?: string;
  modelId: string;
  maxNewTokens?: number;
};

type PreloadCaptionMessage = {
  type: "preload_caption";
  modelId: string;
};

type PreloadAllMessage = {
  type: "preload_all";
  textModelIds: string[];
  captionModelIds: string[];
  dtype: "q4" | "q8" | "fp16" | "fp32";
};

type WarmupAllMessage = {
  type: "warmup_all";
  textModelIds: string[];
  captionModelIds: string[];
  dtype: "q4" | "q8" | "fp16" | "fp32";
};

type ClearRuntimeCacheMessage = {
  type: "clear_runtime_cache";
};

/** Point Transformers.js at a Hugging Face mirror when huggingface.co is unreachable (e.g. regional block). */
type ConfigureHubMessage = {
  type: "configure_hub";
  /** Empty string restores default https://huggingface.co/ */
  remoteHost: string;
};

/** How to run ONNX: auto tries WebGPU then WASM; wasm works on all machines; webgpu may fail on some drivers. */
type ConfigureInferenceMessage = {
  type: "configure_inference";
  backend: InferenceBackend;
};

type WorkerInput =
  | ConfigureHubMessage
  | ConfigureInferenceMessage
  | LoadMessage
  | GenerateMessage
  | CaptionMessage
  | PreloadCaptionMessage
  | PreloadAllMessage
  | WarmupAllMessage
  | ClearRuntimeCacheMessage;

type TextGenerator = ((
  input: string,
  options: Record<string, unknown>,
) => Promise<unknown>) & { tokenizer: unknown };

/** Must match App.tsx `CODE_MODEL` — used to pick a separate in-memory slot so code LLM never evicts chat Gemma. */
const CODE_MODEL_ID = "onnx-community/Qwen2.5-Coder-0.5B-Instruct";

type TextGenSlot = {
  generator: TextGenerator | null;
  modelId: string;
  device: string;
};

const chatSlot: TextGenSlot = { generator: null, modelId: "", device: "unknown" };
const codeSlot: TextGenSlot = { generator: null, modelId: "", device: "unknown" };

const textGenSlotForModelId = (modelId: string): TextGenSlot =>
  modelId === CODE_MODEL_ID ? codeSlot : chatSlot;

const clearTextGenSlots = () => {
  chatSlot.generator = null;
  chatSlot.modelId = "";
  chatSlot.device = "unknown";
  codeSlot.generator = null;
  codeSlot.modelId = "";
  codeSlot.device = "unknown";
};

const textGeneratorCache = new Map<string, { generator: TextGenerator; device: "webgpu" | "wasm" }>();
const captionerCache = new Map<
  string,
  (image: string, options?: Record<string, unknown>) => Promise<Array<{ generated_text: string }>>
>();
let busy = false;

/** Last preference from the UI — affects text + vision pipelines across GPUs. */
let inferenceBackend: InferenceBackend = "auto";

/**
 * `navigator.gpu` can exist while `requestAdapter()` returns null (Linux Mint without Vulkan, bad drivers, etc.).
 * Skipping WebGPU avoids ORT "No available adapters" and hard failures so chat can load on WASM/CPU.
 */
let webGpuAdapterProbe: boolean | null = null;

const resetWebGpuProbe = () => {
  webGpuAdapterProbe = null;
};

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
      return false;
    }
    const adapter = await g.requestAdapter();
    webGpuAdapterProbe = adapter != null;
    return webGpuAdapterProbe;
  } catch {
    webGpuAdapterProbe = false;
    return false;
  }
};

env.allowLocalModels = false;
env.useBrowserCache = true;

const DEFAULT_REMOTE_HOST = "https://huggingface.co/";
/** When the official hub is blocked, retry once with a public mirror (session-only until user saves Settings). */
const PUBLIC_FALLBACK_MIRROR = "https://hf-mirror.com";

const normalizedRemoteHost = (): string => {
  const t = (env.remoteHost ?? "").trim();
  if (t === "") return DEFAULT_REMOTE_HOST;
  return t.endsWith("/") ? t : `${t}/`;
};

const isUsingOfficialHubOnly = () => normalizedRemoteHost() === DEFAULT_REMOTE_HOST;

const isLikelyHubNetworkFailure = (err: unknown): boolean => {
  const raw = err instanceof Error ? err.message : String(err);
  const lower = raw.toLowerCase();
  return (
    lower.includes("failed to fetch") ||
    lower.includes("networkerror") ||
    lower.includes("load failed") ||
    lower.includes("network request failed")
  );
};

const formatHubLoadError = (err: unknown): string => {
  const raw = err instanceof Error ? err.message : String(err);
  const lower = raw.toLowerCase();
  if (
    lower.includes("failed to fetch") ||
    lower.includes("networkerror") ||
    lower.includes("load failed") ||
    lower.includes("network request failed")
  ) {
    const mirrorHint = isUsingOfficialHubOnly()
      ? " Try Settings → HF mirror (e.g. https://hf-mirror.com), Clear cache, Start again."
      : " Official hub and/or your mirror may be blocked — try VPN, another network, or a different mirror.";
    return `${raw} — Cannot reach the model host (firewall, ISP, block, or offline).${mirrorHint}`;
  }
  return raw;
};

/** Apply mirror host; clears in-memory model handles when the host changes. */
const applyHubRemoteHost = (remoteHost: string) => {
  const trimmed = remoteHost.trim();
  const next = trimmed === "" ? DEFAULT_REMOTE_HOST : trimmed.endsWith("/") ? trimmed : `${trimmed}/`;
  if (env.remoteHost === next) return;
  env.remoteHost = next;
  textGeneratorCache.clear();
  captionerCache.clear();
  clearTextGenSlots();
};

const post = (msg: unknown) => {
  self.postMessage(msg);
};

/** Same failure sometimes surfaces both as await throw and as an extra unhandled rejection inside Transformers.js. */
let lastWorkerErrorPostedAt = 0;
const postLoadFailureOnce = (err: unknown) => {
  const now = Date.now();
  if (now - lastWorkerErrorPostedAt < 900) return;
  lastWorkerErrorPostedAt = now;
  post({ type: "error", error: formatHubLoadError(err) });
};

/** Transformers.js sometimes rejects parallel fetches without tying them to pipeline()'s await — surface that to the UI. */
self.onunhandledrejection = (ev: PromiseRejectionEvent) => {
  console.error("[GROVEE worker] unhandledrejection:", ev.reason);
  try {
    ev.preventDefault();
  } catch {
    /* ignore */
  }
  busy = false;
  postLoadFailureOnce(ev.reason);
};

const fmtBytes = (bytes: number) => {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let idx = 0;
  while (value >= 1024 && idx < units.length - 1) {
    value /= 1024;
    idx += 1;
  }
  return `${value.toFixed(idx === 0 ? 0 : 1)} ${units[idx]}`;
};

const clampPercent = (raw: unknown) => {
  if (typeof raw !== "number" || !Number.isFinite(raw)) return 0;
  const pct = raw <= 1 ? raw * 100 : raw;
  return Math.max(0, Math.min(100, Math.round(pct)));
};

const clampProgress = (value: number) => {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, Math.round(value)));
};

/** Room left on the bar until pipeline() resolves (ONNX/WebGPU init after bytes finish). */
const PROGRESS_CAP_UNTIL_PIPELINE_DONE = 96;

/** One model (or one segment of a multi-model batch) maps onto this slice of the 0–100% bar. */
type ProgressSlice = { basePct: number; spanPct: number };

const FULL_PROGRESS_SLICE: ProgressSlice = { basePct: 0, spanPct: 100 };

/** Likely Cache API / disk read — avoid showing “download” and huge speed noise. */
const MEMORY_SPEED_THRESHOLD = 9 * 1024 * 1024;

const fileBaseName = (path: string) => {
  const s = path.trim();
  if (!s) return "";
  const parts = s.split(/[/\\]/);
  return parts[parts.length - 1] ?? s;
};

const capDetailLen = (s: string, n: number) => (s.length <= n ? s : `${s.slice(0, n - 1)}…`);

const normalizeProgressStatus = (status?: string) => {
  const raw = (status ?? "").trim().toLowerCase();
  if (raw === "done" || raw === "complete" || raw === "completed") {
    return "Loading…";
  }
  if (!raw || raw === "progress") {
    return "Loading model (weights + runtime)…";
  }
  return status ?? "Loading…";
};

type HfProgress = {
  status?: string;
  progress?: number;
  loaded?: number;
  total?: number;
  file?: string;
  name?: string;
};

/**
 * Transformers.js reports progress per shard/file; each new file resets % and byte counters.
 * Keep a monotonic bar (never jumps backward). Map inner 0…cap into `slice` so multi-model loads
 * don’t sit at ~96% while the next model is still fetching.
 */
const createMonotonicProgressBridge = (startedAt: number, slice: ProgressSlice) => {
  let highWaterPct = 0;
  /** If % drops by more than this vs the peak, treat as a new file starting. */
  const NEW_FILE_DROP = 15;
  let lastLoaded = 0;
  let lastTs = startedAt;
  let emaSpeed = 0;

  return (progressData: HfProgress) => {
    const rawPct = clampPercent(progressData.progress);
    if (rawPct + NEW_FILE_DROP < highWaterPct) {
      // New ONNX shard — hold the bar steady (library resets % per file)
    } else {
      highWaterPct = Math.max(highWaterPct, rawPct);
    }
    /**
     * Cap during fetch phase so the bar never sits at 100% while ONNX/WebGPU still initializes.
     * Segment end is posted explicitly when pipeline() resolves.
     */
    const innerCapped = Math.min(PROGRESS_CAP_UNTIL_PIPELINE_DONE, highWaterPct);
    const globalPct = slice.basePct + (innerCapped / 100) * slice.spanPct;

    const loaded = typeof progressData.loaded === "number" ? progressData.loaded : 0;
    const total = typeof progressData.total === "number" ? progressData.total : 0;
    const now = Date.now();
    const dt = Math.max(1, (now - lastTs) / 1000);
    const speed = loaded > 0 && loaded >= lastLoaded ? (loaded - lastLoaded) / dt : 0;
    lastLoaded = loaded;
    lastTs = now;
    const elapsed = Math.max(1, (now - startedAt) / 1000);
    if (speed > 0) {
      emaSpeed = emaSpeed <= 0 ? speed : emaSpeed * 0.82 + speed * 0.18;
    }
    const burstFromCache = total > 0 && loaded > 0 && loaded >= total * 0.98 && elapsed < 1.2;
    const loadMode: "network" | "memory" =
      burstFromCache || emaSpeed >= MEMORY_SPEED_THRESHOLD || speed >= MEMORY_SPEED_THRESHOLD
        ? "memory"
        : "network";

    const fname = (progressData.file ?? progressData.name ?? "").trim();
    const bn = fileBaseName(fname);
    let detailText: string;
    if (loadMode === "memory") {
      detailText = bn ? `טוען ממטמון · ${bn}` : "טוען ממטמון הדפדפן";
    } else if (loaded > 0 && total > 0) {
      detailText = `הורדה · ${fmtBytes(loaded)}/${fmtBytes(total)}${bn ? ` · ${bn}` : ""}`;
    } else {
      detailText = bn ? `מכין · ${bn}` : "מכין משקלים והרצה…";
    }
    detailText = capDetailLen(detailText, 90);

    const statusText = normalizeProgressStatus(progressData.status);

    post({
      type: "progress",
      text: statusText,
      progress: clampProgress(globalPct),
      detail: detailText,
      file: fname,
      loadMode,
    });
  };
};

const loadWithDevice = async (
  modelId: string,
  dtype: LoadMessage["dtype"],
  device: "webgpu" | "wasm",
  slice: ProgressSlice = FULL_PROGRESS_SLICE,
) => {
  const runPipeline = async () => {
    post({ type: "status", text: `Loading ${modelId} on ${device}...` });
    const startedAt = Date.now();
    const onProgress = createMonotonicProgressBridge(startedAt, slice);
    const pipe = (await pipeline("text-generation", modelId, {
      device,
      dtype,
      progress_callback: onProgress,
    })) as TextGenerator;

    post({
      type: "progress",
      text: "Loading…",
      progress: clampProgress(slice.basePct + slice.spanPct),
      detail: "",
      file: "",
    });

    return pipe;
  };

  try {
    return await runPipeline();
  } catch (e) {
    if (isLikelyHubNetworkFailure(e) && isUsingOfficialHubOnly()) {
      post({
        type: "status",
        text: `Cannot reach huggingface.co — retrying once via ${PUBLIC_FALLBACK_MIRROR} …`,
      });
      applyHubRemoteHost(PUBLIC_FALLBACK_MIRROR);
      return await runPipeline();
    }
    throw e;
  }
};

const loadCaptioner = async (
  modelId: string,
  device: "webgpu" | "wasm",
  slice: ProgressSlice = FULL_PROGRESS_SLICE,
) => {
  const visionModel = modelId;
  const runPipeline = async () => {
    post({ type: "status", text: `Loading ${visionModel} on ${device}...` });
    const startedAt = Date.now();
    const onProgress = createMonotonicProgressBridge(startedAt, slice);
    const pipe = (await pipeline("image-to-text", visionModel, {
      device,
      dtype: "q8",
      progress_callback: onProgress,
    })) as (image: string, options?: Record<string, unknown>) => Promise<Array<{ generated_text: string }>>;
    post({
      type: "progress",
      text: "Loading…",
      progress: clampProgress(slice.basePct + slice.spanPct),
      detail: "",
      file: "",
    });
    return pipe;
  };

  try {
    return await runPipeline();
  } catch (e) {
    if (isLikelyHubNetworkFailure(e) && isUsingOfficialHubOnly()) {
      post({
        type: "status",
        text: `Cannot reach huggingface.co — retrying vision download via ${PUBLIC_FALLBACK_MIRROR} …`,
      });
      applyHubRemoteHost(PUBLIC_FALLBACK_MIRROR);
      return await runPipeline();
    }
    throw e;
  }
};

const loadTextGenerator = async (
  modelId: string,
  dtype: LoadMessage["dtype"],
  progressSlice: ProgressSlice = FULL_PROGRESS_SLICE,
) => {
  const pref = inferenceBackend;

  const tryWasm = async () => {
    const loaded = await loadWithDevice(modelId, dtype, "wasm", progressSlice);
    const entry = { generator: loaded, device: "wasm" as const };
    textGeneratorCache.set(modelId, entry);
    return entry;
  };

  const tryWebGpu = async () => {
    const loaded = await loadWithDevice(modelId, dtype, "webgpu", progressSlice);
    const entry = { generator: loaded, device: "webgpu" as const };
    textGeneratorCache.set(modelId, entry);
    return entry;
  };

  const cached = textGeneratorCache.get(modelId);
  if (cached) {
    if (pref === "auto") return cached;
    if (pref === "webgpu" && cached.device === "webgpu") return cached;
    if (pref === "wasm" && cached.device === "wasm") return cached;
    textGeneratorCache.delete(modelId);
  }

  if (pref === "wasm") {
    post({
      type: "status",
      text: `Loading ${modelId} on WASM (CPU — works on all systems; slower than WebGPU)…`,
    });
    return await tryWasm();
  }

  if (pref === "webgpu") {
    if (await hasRunnableWebGpuAdapter()) {
      post({ type: "status", text: `Loading ${modelId} on WebGPU…` });
      try {
        return await tryWebGpu();
      } catch {
        post({ type: "status", text: `WebGPU error — using WASM (CPU) for ${modelId}.` });
        return await tryWasm();
      }
    }
    post({
      type: "status",
      text: `No WebGPU adapter — loading ${modelId} on WASM (CPU).`,
    });
    return await tryWasm();
  }

  // auto: WebGPU only when adapter exists; else WASM (Linux-friendly)
  if (await hasRunnableWebGpuAdapter()) {
    try {
      return await tryWebGpu();
    } catch {
      post({
        type: "status",
        text: `WebGPU failed for ${modelId}. Using WASM (CPU) — slower but compatible.`,
      });
      return await tryWasm();
    }
  }
  post({
    type: "status",
    text: `No WebGPU adapter — using WASM (CPU) for ${modelId} (try Chromium + GPU drivers on Linux).`,
  });
  return await tryWasm();
};

type CaptionFn = (image: string, options?: Record<string, unknown>) => Promise<Array<{ generated_text: string }>>;

/** Vision pipeline: same backend rules as text (consistent across Intel / AMD / NVIDIA / no GPU). */
const loadCaptionerPreferred = async (
  modelId: string,
  progressSlice: ProgressSlice = FULL_PROGRESS_SLICE,
): Promise<{ captioner: CaptionFn; device: "webgpu" | "wasm" }> => {
  const pref = inferenceBackend;

  if (pref === "wasm") {
    const captioner = await loadCaptioner(modelId, "wasm", progressSlice);
    return { captioner, device: "wasm" };
  }
  if (pref === "webgpu") {
    if (await hasRunnableWebGpuAdapter()) {
      try {
        const captioner = await loadCaptioner(modelId, "webgpu", progressSlice);
        return { captioner, device: "webgpu" };
      } catch {
        const captioner = await loadCaptioner(modelId, "wasm", progressSlice);
        return { captioner, device: "wasm" };
      }
    }
    const captioner = await loadCaptioner(modelId, "wasm", progressSlice);
    return { captioner, device: "wasm" };
  }
  if (await hasRunnableWebGpuAdapter()) {
    try {
      const captioner = await loadCaptioner(modelId, "webgpu", progressSlice);
      return { captioner, device: "webgpu" };
    } catch {
      const captioner = await loadCaptioner(modelId, "wasm", progressSlice);
      return { captioner, device: "wasm" };
    }
  }
  const captioner = await loadCaptioner(modelId, "wasm", progressSlice);
  return { captioner, device: "wasm" };
};

const normalizePrompt = (message: GenerateMessage) => {
  const webBlock = message.webContext?.trim()
    ? `\n\nWeb context:\n${message.webContext.trim()}\nUse it only if relevant.\n`
    : "";
  const thinkingBlock = message.thinkingMode
    ? "\nThink carefully before answering, but output only the final answer."
    : "";
  return `${message.systemPrompt}${thinkingBlock}${webBlock}\n\nQuestion:\n${message.prompt}\n\nAnswer:`;
};

const buildPrompt = (model: TextGenerator, message: GenerateMessage) => {
  const tokenizer = (model as { tokenizer?: unknown }).tokenizer as
    | { apply_chat_template?: (messages: unknown, opts: unknown) => string }
    | undefined;

  if (tokenizer?.apply_chat_template) {
    const messages = [
      { role: "system", content: message.systemPrompt },
      ...(message.webContext?.trim()
        ? [{ role: "system", content: `Web context:\n${message.webContext.trim()}` }]
        : []),
      { role: "user", content: message.prompt },
    ];

    try {
      return tokenizer.apply_chat_template(messages, {
        tokenize: false,
        add_generation_prompt: true,
      });
    } catch {
      return normalizePrompt(message);
    }
  }

  return normalizePrompt(message);
};

self.onmessage = async (event: MessageEvent<WorkerInput>) => {
  const message = event.data;

  try {
    if (message.type === "configure_hub") {
      applyHubRemoteHost(message.remoteHost);
      return;
    }

    if (message.type === "configure_inference") {
      const next = message.backend;
      if (next !== inferenceBackend) {
        inferenceBackend = next;
        resetWebGpuProbe();
        textGeneratorCache.clear();
        captionerCache.clear();
        clearTextGenSlots();
      }
      return;
    }

    if (message.type === "load") {
      if (busy) {
        post({ type: "error", error: "Generation in progress. Please wait." });
        return;
      }

      if (chatSlot.generator && message.modelId === chatSlot.modelId) {
        post({ type: "loaded", modelId: chatSlot.modelId, device: chatSlot.device });
        return;
      }
      const loaded = await loadTextGenerator(message.modelId, message.dtype);
      chatSlot.generator = loaded.generator;
      chatSlot.device = loaded.device;
      chatSlot.modelId = message.modelId;
      post({ type: "loaded", modelId: chatSlot.modelId, device: chatSlot.device });
      return;
    }

    if (message.type === "preload_all") {
      if (busy) {
        post({ type: "error", error: "Another task is in progress. Please wait." });
        return;
      }
      const total = message.textModelIds.length + message.captionModelIds.length;
      let completed = 0;
      post({
        type: "progress",
        text: "טוען מודלים מקומית…",
        progress: 0,
        detail: capDetailLen(`0 מתוך ${total} מודלים`, 90),
        file: "",
      });
      const failedTextModelIds: string[] = [];
      const failedCaptionModelIds: string[] = [];
      for (const textModelId of message.textModelIds) {
        const basePct = (completed / total) * 100;
        const spanPct = 100 / total;
        post({ type: "status", text: `Preparing text model ${completed + 1}/${total}: ${textModelId}` });
        try {
          if (!textGeneratorCache.has(textModelId)) {
            await loadTextGenerator(textModelId, message.dtype, { basePct, spanPct });
          }
        } catch (e) {
          const err = e instanceof Error ? e.message : String(e);
          failedTextModelIds.push(textModelId);
          post({ type: "status", text: `Could not load ${textModelId}: ${err}` });
        }
        completed += 1;
        post({
          type: "progress",
          text: "טוען מודלים…",
          progress: clampProgress((completed / total) * 100),
          detail: capDetailLen(`${completed}/${total} מודלים · ${fileBaseName(textModelId)}`, 90),
          file: textModelId,
        });
      }
      for (const captionModelId of message.captionModelIds) {
        const basePct = (completed / total) * 100;
        const spanPct = 100 / total;
        post({ type: "status", text: `Preparing vision model ${completed + 1}/${total}: ${captionModelId}` });
        if (!captionerCache.has(captionModelId)) {
          try {
            const { captioner } = await loadCaptionerPreferred(captionModelId, { basePct, spanPct });
            captionerCache.set(captionModelId, captioner);
          } catch (e) {
            const err = e instanceof Error ? e.message : String(e);
            failedCaptionModelIds.push(captionModelId);
            post({ type: "status", text: `Could not load vision ${captionModelId}: ${err}` });
          }
        }
        completed += 1;
        post({
          type: "progress",
          text: "טוען מודלים…",
          progress: clampProgress((completed / total) * 100),
          detail: capDetailLen(`${completed}/${total} מודלים · ${fileBaseName(captionModelId)}`, 90),
          file: captionModelId,
        });
      }
      post({
        type: "preload_all_done",
        textModels: message.textModelIds.length,
        captionModels: message.captionModelIds.length,
        ...(failedTextModelIds.length ? { failedTextModelIds } : {}),
        ...(failedCaptionModelIds.length ? { failedCaptionModelIds } : {}),
      });
      return;
    }

    if (message.type === "warmup_all") {
      if (busy) {
        post({ type: "error", error: "Another task is in progress. Please wait." });
        return;
      }
      const total = message.textModelIds.length + message.captionModelIds.length;
      let completed = 0;
      post({
        type: "status",
        text: "Gemma is ready. Downloading additional models in background...",
      });
      const failedTextModelIds: string[] = [];
      const failedCaptionModelIds: string[] = [];
      for (const textModelId of message.textModelIds) {
        const basePct = (completed / total) * 100;
        const spanPct = 100 / total;
        try {
          if (!textGeneratorCache.has(textModelId)) {
            await loadTextGenerator(textModelId, message.dtype, { basePct, spanPct });
          }
        } catch (e) {
          const err = e instanceof Error ? e.message : String(e);
          failedTextModelIds.push(textModelId);
          post({ type: "status", text: `Warmup failed for ${textModelId}: ${err}` });
        }
        completed += 1;
        post({
          type: "progress",
          text: "טעינת רקע…",
          progress: clampProgress((completed / total) * 100),
          detail: capDetailLen(`${completed}/${total} · ${fileBaseName(textModelId)}`, 90),
          file: textModelId,
        });
      }
      for (const captionModelId of message.captionModelIds) {
        const basePct = (completed / total) * 100;
        const spanPct = 100 / total;
        if (!captionerCache.has(captionModelId)) {
          try {
            const { captioner } = await loadCaptionerPreferred(captionModelId, { basePct, spanPct });
            captionerCache.set(captionModelId, captioner);
          } catch (e) {
            const err = e instanceof Error ? e.message : String(e);
            failedCaptionModelIds.push(captionModelId);
            post({ type: "status", text: `Vision warmup failed for ${captionModelId}: ${err}` });
          }
        }
        completed += 1;
        post({
          type: "progress",
          text: "טעינת רקע…",
          progress: clampProgress((completed / total) * 100),
          detail: capDetailLen(`${completed}/${total} · ${fileBaseName(captionModelId)}`, 90),
          file: captionModelId,
        });
      }
      post({
        type: "preload_all_done",
        textModels: message.textModelIds.length,
        captionModels: message.captionModelIds.length,
        ...(failedTextModelIds.length ? { failedTextModelIds } : {}),
        ...(failedCaptionModelIds.length ? { failedCaptionModelIds } : {}),
      });
      return;
    }

    if (message.type === "clear_runtime_cache") {
      clearTextGenSlots();
      textGeneratorCache.clear();
      captionerCache.clear();
      busy = false;
      post({ type: "status", text: "Runtime model cache cleared." });
      return;
    }

    if (message.type === "generate") {
      if (busy) {
        post({ type: "error", error: "Generation already in progress." });
        return;
      }

      const slot = textGenSlotForModelId(message.modelId);
      if (!slot.generator || slot.modelId !== message.modelId) {
        const switched = await loadTextGenerator(message.modelId, "q4");
        slot.generator = switched.generator;
        slot.modelId = message.modelId;
        slot.device = switched.device;
      }

      const gen = slot.generator;
      if (!gen) {
        post({ type: "error", error: "Model is not loaded yet." });
        return;
      }

      busy = true;
      const finalPrompt = buildPrompt(gen, message);
      const streamer = new TextStreamer(gen.tokenizer as never, {
        skip_prompt: true,
        callback_function: (text: string) => {
          post({ type: "token", text });
        },
      });

      const temperature = message.temperature;
      await gen(finalPrompt, {
        max_new_tokens: message.maxNewTokens,
        temperature,
        do_sample: temperature > 0.01,
        repetition_penalty: message.repetitionPenalty,
        top_p: message.topP,
        no_repeat_ngram_size: 3,
        return_full_text: false,
        streamer,
      });

      post({ type: "done" });
      busy = false;
      return;
    }

    if (message.type === "caption") {
      if (busy) {
        post({ type: "error", error: "Another task is in progress. Please wait." });
        return;
      }

      busy = true;
      let captioner = captionerCache.get(message.modelId) ?? null;
      if (!captioner) {
        const loadedCap = await loadCaptionerPreferred(message.modelId);
        captioner = loadedCap.captioner;
        captionerCache.set(message.modelId, captioner);
      }

      const result = await captioner(message.imageDataUrl, {
        max_new_tokens: message.maxNewTokens ?? 80,
      });
      const captionText = result?.[0]?.generated_text?.trim() || "Could not generate image description.";
      const prefix = message.prompt?.trim() ? `${message.prompt.trim()}\n` : "";
      post({ type: "caption_done", text: `${prefix}${captionText}` });
      busy = false;
      return;
    }

    if (message.type === "preload_caption") {
      if (busy) {
        post({ type: "error", error: "Another task is in progress. Please wait." });
        return;
      }

      const existing = captionerCache.get(message.modelId);
      if (existing) {
        post({ type: "caption_model_loaded", modelId: message.modelId, device: "cache" });
        return;
      }

      const { captioner, device } = await loadCaptionerPreferred(message.modelId);
      captionerCache.set(message.modelId, captioner);
      post({ type: "caption_model_loaded", modelId: message.modelId, device });
    }
  } catch (error) {
    busy = false;
    postLoadFailureOnce(error);
  }
};
