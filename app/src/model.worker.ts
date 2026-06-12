/// <reference lib="webworker" />

import {
  AutoProcessor,
  Gemma4ForConditionalGeneration,
  InterruptableStoppingCriteria,
  RawImage,
  TextStreamer,
  env,
} from "@huggingface/transformers";
import {
  SCENE_ANALYSIS_SYSTEM_PROMPT,
  buildSceneAnalysisUserPrompt,
} from "./cameraPrompts";
import { parseSceneAnalysisJson } from "./worldMemory";

type Gemma4Processor = Awaited<ReturnType<typeof AutoProcessor.from_pretrained>>;
type Gemma4Model = InstanceType<typeof Gemma4ForConditionalGeneration>;

export const GEMMA_MODEL_ID = "onnx-community/gemma-4-E2B-it-ONNX";

type InferenceBackend = "auto" | "webgpu" | "wasm";

type LoadMessage = {
  type: "load";
  modelId: string;
  dtype: "q4" | "q8" | "fp16" | "fp32";
};

export type WorkerImagePayload = {
  bytes: ArrayBuffer;
  mime: string;
};

export type ChatTurn = {
  role: "user" | "assistant";
  content: string;
  images?: WorkerImagePayload[];
};

type GenerateMessage = {
  type: "generate";
  modelId: string;
  prompt: string;
  systemPrompt: string;
  history: ChatTurn[];
  images: WorkerImagePayload[];
  maxNewTokens: number;
  temperature: number;
  repetitionPenalty: number;
  topP: number;
  thinkingMode: boolean;
  webContext: string;
};

type ClearRuntimeCacheMessage = {
  type: "clear_runtime_cache";
};

type AbortMessage = {
  type: "abort";
};

type ConfigureHubMessage = {
  type: "configure_hub";
  remoteHost: string;
};

type ConfigureInferenceMessage = {
  type: "configure_inference";
  backend: InferenceBackend;
};

type AnalyzeSceneMessage = {
  type: "analyze_scene";
  requestId: string;
  modelId: string;
  images: WorkerImagePayload[];
  previousSummary?: string;
  sensorBlock?: string;
};

type CharacterUtteranceMessage = {
  type: "character_utterance";
  requestId: string;
  modelId: string;
  systemPrompt: string;
  userPrompt: string;
  maxNewTokens?: number;
};

type WorkerInput =
  | ConfigureHubMessage
  | ConfigureInferenceMessage
  | LoadMessage
  | GenerateMessage
  | AnalyzeSceneMessage
  | CharacterUtteranceMessage
  | ClearRuntimeCacheMessage
  | AbortMessage;

type ModelSlot = {
  model: Gemma4Model | null;
  processor: Gemma4Processor | null;
  modelId: string;
  device: string;
};

const chatSlot: ModelSlot = { model: null, processor: null, modelId: "", device: "unknown" };

const clearModelSlots = () => {
  chatSlot.model = null;
  chatSlot.processor = null;
  chatSlot.modelId = "";
  chatSlot.device = "unknown";
};

type CachedModel = {
  model: Gemma4Model;
  processor: Gemma4Processor;
  device: "webgpu" | "wasm";
};

const modelCache = new Map<string, CachedModel>();
let chatBusy = false;
let sceneBusy = false;
let abortRequested = false;
let activeInterrupt: InterruptableStoppingCriteria | null = null;
let inferenceBackend: InferenceBackend = "auto";

const isWorkerBusy = () => chatBusy || sceneBusy;

const isWebGpuRuntimeError = (err: unknown): boolean => {
  const msg = err instanceof Error ? err.message : String(err);
  return (
    /webgpu|OrtRun|GPUBuffer|mapAsync|Device.*is lost|device is lost|external Instance/i.test(msg) ||
    /GatherBlockQuantized|Can't create a session|ERROR_CODE:\s*9|Could not find an implementation/i.test(
      msg,
    )
  );
};

/** Remember GPUs that advertise WebGPU but lack quantized ONNX ops (e.g. GatherBlockQuantized). */
let webGpuOnnxBlocked = false;

const ensureChatSlot = async (modelId: string): Promise<{ model: Gemma4Model; processor: Gemma4Processor }> => {
  if (!chatSlot.model || !chatSlot.processor || chatSlot.modelId !== modelId) {
    const switched = await loadMultimodalModel(modelId, "q4");
    chatSlot.model = switched.model;
    chatSlot.processor = switched.processor;
    chatSlot.modelId = modelId;
    chatSlot.device = switched.device;
  }
  const model = chatSlot.model;
  const processor = chatSlot.processor;
  if (!model || !processor) {
    throw new Error("Model not ready.");
  }
  return { model, processor };
};

const runSceneGenerateWithFallback = async (
  modelId: string,
  run: (model: Gemma4Model) => Promise<void>,
): Promise<void> => {
  const { model } = await ensureChatSlot(modelId);
  try {
    await run(model);
  } catch (err) {
    if (chatSlot.device !== "webgpu" || !isWebGpuRuntimeError(err)) {
      throw err;
    }
    post({
      type: "status",
      text: "WebGPU נכשל (עומס GPU) — עובר ל-WASM ומנסה שוב…",
    });
    const switched = await forceReloadWasm(modelId);
    chatSlot.model = switched.model;
    chatSlot.processor = switched.processor;
    chatSlot.modelId = modelId;
    chatSlot.device = switched.device;
    await run(switched.model);
  }
};

const forceReloadWasm = async (modelId: string): Promise<CachedModel> => {
  webGpuOnnxBlocked = true;
  inferenceBackend = "wasm";
  const cachedWasm = modelCache.get(modelId);
  if (cachedWasm?.device === "wasm") {
    return cachedWasm;
  }
  modelCache.delete(modelId);
  clearModelSlots();
  return loadMultimodalModel(modelId, "q4");
};

const waitForSceneIdle = async (maxMs = 180_000): Promise<boolean> => {
  const start = Date.now();
  while (sceneBusy && Date.now() - start < maxMs) {
    await new Promise<void>((resolve) => setTimeout(resolve, 150));
  }
  return !sceneBusy;
};

let webGpuAdapterProbe: boolean | null = null;

const resetInferenceRuntime = () => {
  webGpuAdapterProbe = null;
  webGpuOnnxBlocked = false;
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
    const adapter = await g.requestAdapter({ powerPreference: "low-power" });
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
  if (
    lower.includes("gatherblockquantized") ||
    lower.includes("could not find an implementation") ||
    lower.includes("can't create a session")
  ) {
    return `${raw} — WebGPU on this GPU cannot run the quantized Gemma model. Settings → Inference → WASM, then Clear cache → Start again.`;
  }
  return raw;
};

const applyHubRemoteHost = (remoteHost: string) => {
  const trimmed = remoteHost.trim();
  const next = trimmed === "" ? DEFAULT_REMOTE_HOST : trimmed.endsWith("/") ? trimmed : `${trimmed}/`;
  if (env.remoteHost === next) return;
  env.remoteHost = next;
  modelCache.clear();
  clearModelSlots();
};

const post = (msg: unknown) => {
  self.postMessage(msg);
};

let lastWorkerErrorPostedAt = 0;
const postLoadFailureOnce = (err: unknown) => {
  const now = Date.now();
  if (now - lastWorkerErrorPostedAt < 900) return;
  lastWorkerErrorPostedAt = now;
  post({ type: "error", error: formatHubLoadError(err) });
};

self.onunhandledrejection = (ev: PromiseRejectionEvent) => {
  console.error("[GROVEE worker] unhandledrejection:", ev.reason);
  try {
    ev.preventDefault();
  } catch {
    /* ignore */
  }
  chatBusy = false;
  sceneBusy = false;
  postLoadFailureOnce(ev.reason);
};

const clampProgress = (value: number) => {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, Math.round(value)));
};

const shortFileName = (path: string) => {
  const t = path.trim();
  if (!t) return "";
  const parts = t.split(/[/\\]/);
  return parts[parts.length - 1] ?? t;
};

type HfProgress = {
  status?: string;
  progress?: number;
  loaded?: number;
  total?: number;
  file?: string;
  name?: string;
  files?: Record<string, { loaded: number; total: number }>;
};

/** Gemma 4 E2B q4 multimodal pack (text + vision + audio encoders) — ~3.9 GB. */
const GEMMA_Q4_MULTIMODAL_PACK_BYTES = 3_900_000_000;

const pickActiveDownloadFile = (
  files: Record<string, { loaded: number; total: number }> | undefined,
): string => {
  if (!files) return "";
  for (const [path, st] of Object.entries(files)) {
    if (st.total > 0 && st.loaded < st.total * 0.995) return path;
  }
  return "";
};

const createDownloadProgressBridge = (startedAt: number) => {
  let overallLoaded = 0;
  let overallTotal = 0;
  let overallPct = 0;
  let activeFile = "";
  let speedEma = 0;
  let speedLastLoaded = 0;
  let speedLastTs = startedAt;

  const emitOverall = () => {
    const label = shortFileName(activeFile);
    post({
      type: "progress",
      text: label ? `מוריד: ${label}` : "מוריד מודל…",
      progress: overallPct,
      phase: "download",
      loaded: overallLoaded,
      total: overallTotal,
      speedBps: Math.round(speedEma),
      detail: "",
      file: activeFile,
    });
  };

  const updateSpeed = (loaded: number) => {
    const now = Date.now();
    const dt = Math.max(0.25, (now - speedLastTs) / 1000);
    if (loaded > speedLastLoaded) {
      const instant = (loaded - speedLastLoaded) / dt;
      speedEma = speedEma > 0 ? speedEma * 0.75 + instant * 0.25 : instant;
      speedLastLoaded = loaded;
      speedLastTs = now;
    }
  };

  return (progressData: HfProgress) => {
    const status = progressData.status ?? "";

    if (status === "progress_total") {
      const loaded = typeof progressData.loaded === "number" ? progressData.loaded : 0;
      const total = typeof progressData.total === "number" ? progressData.total : 0;
      if (loaded <= 0 && total <= 0) return;

      overallLoaded = Math.max(overallLoaded, loaded);
      overallTotal = Math.max(overallTotal, total || GEMMA_Q4_MULTIMODAL_PACK_BYTES);
      const pct =
        typeof progressData.progress === "number" && progressData.progress > 0
          ? progressData.progress
          : overallTotal > 0
            ? (overallLoaded / overallTotal) * 100
            : 0;
      overallPct = Math.max(overallPct, clampProgress(pct));

      const fromMap = pickActiveDownloadFile(progressData.files);
      if (fromMap) activeFile = fromMap;

      updateSpeed(overallLoaded);
      emitOverall();
      return;
    }

    if (status === "progress" && progressData.file) {
      activeFile = progressData.file;
      const loaded = typeof progressData.loaded === "number" ? progressData.loaded : 0;
      const total = typeof progressData.total === "number" ? progressData.total : 0;
      if (overallTotal <= 0 && (loaded > 0 || total > 0)) {
        overallLoaded = Math.max(overallLoaded, loaded);
        overallTotal = Math.max(overallTotal, total);
        if (overallTotal > 0) {
          overallPct = Math.max(overallPct, clampProgress((overallLoaded / overallTotal) * 100));
        }
        updateSpeed(overallLoaded);
        emitOverall();
      }
    }
  };
};

const loadWithDevice = async (
  modelId: string,
  dtype: LoadMessage["dtype"],
  device: "webgpu" | "wasm",
) => {
  const runLoad = async () => {
    post({ type: "status", text: `Loading ${modelId} (vision + text) on ${device}...` });
    const startedAt = Date.now();
    const onProgress = createDownloadProgressBridge(startedAt);

    const loadOpts = { device, dtype, progress_callback: onProgress };

    const processor = (await AutoProcessor.from_pretrained(modelId, loadOpts)) as Gemma4Processor;
    const model = (await Gemma4ForConditionalGeneration.from_pretrained(modelId, loadOpts)) as Gemma4Model;

    post({
      type: "progress",
      text: "מאתחל מודל (ONNX / WebGPU)…",
      progress: 100,
      phase: "init",
      loaded: 0,
      total: 0,
      detail: "הקבצים הורדו — טוען לזיכרון",
      file: "",
    });

    return { model, processor };
  };

  try {
    return await runLoad();
  } catch (e) {
    if (isLikelyHubNetworkFailure(e) && isUsingOfficialHubOnly()) {
      post({
        type: "status",
        text: `Cannot reach huggingface.co — retrying once via ${PUBLIC_FALLBACK_MIRROR} …`,
      });
      applyHubRemoteHost(PUBLIC_FALLBACK_MIRROR);
      return await runLoad();
    }
    throw e;
  }
};

const loadMultimodalModel = async (modelId: string, dtype: LoadMessage["dtype"]) => {
  if (modelId !== GEMMA_MODEL_ID) {
    throw new Error(`Only ${GEMMA_MODEL_ID} is supported.`);
  }

  const pref = inferenceBackend;

  const tryWasm = async () => {
    const loaded = await loadWithDevice(modelId, dtype, "wasm");
    const entry: CachedModel = { ...loaded, device: "wasm" };
    modelCache.set(modelId, entry);
    return entry;
  };

  const tryWebGpu = async () => {
    const loaded = await loadWithDevice(modelId, dtype, "webgpu");
    const entry: CachedModel = { ...loaded, device: "webgpu" };
    modelCache.set(modelId, entry);
    return entry;
  };

  const cached = modelCache.get(modelId);
  if (cached) {
    if (cached.device === "webgpu" && webGpuOnnxBlocked) {
      modelCache.delete(modelId);
    } else if (pref === "auto") {
      return cached;
    } else if (pref === "webgpu" && cached.device === "webgpu") {
      return cached;
    } else if (pref === "wasm" && cached.device === "wasm") {
      return cached;
    } else {
      modelCache.delete(modelId);
    }
  }

  const tryWebGpuWithFallback = async (): Promise<CachedModel> => {
    if (webGpuOnnxBlocked) {
      post({
        type: "status",
        text: `WebGPU lacks required ONNX ops on this GPU — loading ${modelId} on WASM (CPU).`,
      });
      return await tryWasm();
    }
    try {
      return await tryWebGpu();
    } catch (err) {
      if (!isWebGpuRuntimeError(err)) throw err;
      webGpuOnnxBlocked = true;
      modelCache.delete(modelId);
      post({
        type: "status",
        text: `WebGPU error — using WASM (CPU) for ${modelId}.`,
      });
      return await tryWasm();
    }
  };

  if (pref === "wasm") {
    post({
      type: "status",
      text: `Loading ${modelId} on WASM (CPU — slower; vision works best with WebGPU)…`,
    });
    return await tryWasm();
  }

  if (pref === "webgpu") {
    if (await hasRunnableWebGpuAdapter()) {
      post({ type: "status", text: `Loading ${modelId} on WebGPU…` });
      return await tryWebGpuWithFallback();
    }
    post({
      type: "status",
      text: `No WebGPU adapter — loading ${modelId} on WASM (CPU).`,
    });
    return await tryWasm();
  }

  if (await hasRunnableWebGpuAdapter()) {
    return await tryWebGpuWithFallback();
  }

  post({ type: "status", text: `No WebGPU adapter — loading ${modelId} on WASM (CPU).` });
  return await tryWasm();
};

const DEFAULT_VISION_PROMPT =
  "Describe this image in detail. Answer in the same language as the user's message.";

const toRawImages = async (payloads: WorkerImagePayload[]): Promise<RawImage[]> => {
  const out: RawImage[] = [];
  for (const p of payloads) {
    const blob = new Blob([p.bytes], { type: p.mime || "image/jpeg" });
    out.push(await RawImage.fromBlob(blob));
  }
  return out;
};

const collectImagesInOrder = (message: GenerateMessage): WorkerImagePayload[] => {
  const ordered: WorkerImagePayload[] = [];
  const historyLen = message.history.length;
  message.history.forEach((turn, i) => {
    if (turn.role === "user" && turn.images?.length && i >= historyLen - 2) {
      ordered.push(...turn.images);
    }
  });
  if (message.images.length) ordered.push(...message.images);
  return ordered;
};

const buildTurnContent = (role: "user" | "assistant", text: string, imageCount: number) => {
  if (role !== "user" || imageCount <= 0) return text;
  const parts: Array<{ type: string; text?: string }> = [];
  for (let i = 0; i < imageCount; i++) parts.push({ type: "image" });
  parts.push({ type: "text", text: text.trim() || DEFAULT_VISION_PROMPT });
  return parts;
};

const buildInputs = async (processor: Gemma4Processor, message: GenerateMessage) => {
  type ChatContent = string | Array<{ type: string; text?: string }>;
  type ChatMsg = { role: string; content: ChatContent };

  const chatMessages: ChatMsg[] = [{ role: "system", content: message.systemPrompt }];

  if (message.webContext?.trim()) {
    chatMessages.push({ role: "system", content: `Web context:\n${message.webContext.trim()}` });
  }

  const historyLen = message.history.length;
  for (let i = 0; i < historyLen; i++) {
    const turn = message.history[i];
    const keepImages = turn.role === "user" && i >= historyLen - 2;
    const imgCount = keepImages ? (turn.images?.length ?? 0) : 0;
    chatMessages.push({
      role: turn.role,
      content: buildTurnContent(turn.role, turn.content, imgCount),
    });
  }

  const currentImages = message.images.length;
  chatMessages.push({
    role: "user",
    content: buildTurnContent("user", message.prompt, currentImages),
  });

  const promptText = processor.apply_chat_template(chatMessages, {
    tokenize: false,
    add_generation_prompt: true,
    ...(message.thinkingMode ? { enable_thinking: true } : {}),
  } as Parameters<Gemma4Processor["apply_chat_template"]>[1]) as string;

  const allPayloads = collectImagesInOrder(message);
  let imagesArg: RawImage | RawImage[] | null = null;
  if (allPayloads.length > 0) {
    const raw = await toRawImages(allPayloads);
    imagesArg = raw.length === 1 ? raw[0] : raw;
  }

  return processor(promptText, imagesArg, null, { add_special_tokens: false });
};

const buildSceneAnalysisInputs = async (
  processor: Gemma4Processor,
  message: AnalyzeSceneMessage,
) => {
  type ChatMsg = {
    role: string;
    content: string | Array<{ type: string; text?: string }>;
  };

  const userText = buildSceneAnalysisUserPrompt(message.previousSummary, message.sensorBlock);
  const chatMessages: ChatMsg[] = [
    { role: "system", content: SCENE_ANALYSIS_SYSTEM_PROMPT },
    {
      role: "user",
      content: [
        { type: "image" },
        { type: "text", text: userText },
      ],
    },
  ];

  const promptText = processor.apply_chat_template(chatMessages, {
    tokenize: false,
    add_generation_prompt: true,
  } as Parameters<Gemma4Processor["apply_chat_template"]>[1]) as string;

  const raw = await toRawImages(message.images);
  const imagesArg = raw.length === 1 ? raw[0] : raw;
  return processor(promptText, imagesArg, null, { add_special_tokens: false });
};

const buildCharacterUtteranceInputs = async (
  processor: Gemma4Processor,
  message: CharacterUtteranceMessage,
) => {
  type ChatMsg = { role: string; content: string };
  const chatMessages: ChatMsg[] = [
    { role: "system", content: message.systemPrompt },
    { role: "user", content: message.userPrompt },
  ];
  const promptText = processor.apply_chat_template(chatMessages, {
    tokenize: false,
    add_generation_prompt: true,
  } as Parameters<Gemma4Processor["apply_chat_template"]>[1]) as string;
  return processor(promptText, null, null, { add_special_tokens: false });
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
        resetInferenceRuntime();
        modelCache.clear();
        clearModelSlots();
      }
      return;
    }

    if (message.type === "load") {
      if (isWorkerBusy()) {
        post({ type: "error", error: "Generation in progress. Please wait." });
        return;
      }

      if (chatSlot.model && message.modelId === chatSlot.modelId) {
        post({ type: "loaded", modelId: chatSlot.modelId, device: chatSlot.device });
        return;
      }
      const loaded = await loadMultimodalModel(message.modelId, message.dtype);
      chatSlot.model = loaded.model;
      chatSlot.processor = loaded.processor;
      chatSlot.device = loaded.device;
      chatSlot.modelId = message.modelId;
      post({ type: "loaded", modelId: chatSlot.modelId, device: chatSlot.device });
      return;
    }

    if (message.type === "clear_runtime_cache") {
      clearModelSlots();
      modelCache.clear();
      chatBusy = false;
      sceneBusy = false;
      abortRequested = false;
      activeInterrupt = null;
      post({ type: "status", text: "Runtime model cache cleared." });
      return;
    }

    if (message.type === "abort") {
      abortRequested = true;
      activeInterrupt?.interrupt();
      return;
    }

    if (message.type === "generate") {
      if (chatBusy) {
        post({ type: "error", error: "Generation already in progress.", scope: "chat" });
        return;
      }

      if (!(await waitForSceneIdle())) {
        post({
          type: "error",
          error: "Camera analysis is still running. Please wait a moment and try again.",
          scope: "chat",
        });
        return;
      }

      if (message.modelId !== GEMMA_MODEL_ID) {
        post({ type: "error", error: `Only ${GEMMA_MODEL_ID} is supported.`, scope: "chat" });
        return;
      }

      if (!chatSlot.model || !chatSlot.processor || chatSlot.modelId !== message.modelId) {
        const switched = await loadMultimodalModel(message.modelId, "q4");
        chatSlot.model = switched.model;
        chatSlot.processor = switched.processor;
        chatSlot.modelId = message.modelId;
        chatSlot.device = switched.device;
      }

      const model = chatSlot.model;
      const processor = chatSlot.processor;
      if (!model || !processor) {
        post({ type: "error", error: "Model is not loaded yet.", scope: "chat" });
        return;
      }

      chatBusy = true;
      abortRequested = false;
      const interrupt = new InterruptableStoppingCriteria();
      activeInterrupt = interrupt;

      post({ type: "status", text: message.images.length ? "מעבד תמונה…" : "Generating…" });

      const inputs = await buildInputs(processor, message);

      if (abortRequested || interrupt.interrupted) {
        post({ type: "aborted" });
        chatBusy = false;
        activeInterrupt = null;
        return;
      }

      const streamer = new TextStreamer(processor.tokenizer as never, {
        skip_prompt: true,
        callback_function: (text: string) => {
          post({ type: "token", text });
        },
      });

      const temperature = message.temperature;
      const runGenerate = async (activeModel: Gemma4Model) =>
        activeModel.generate({
          ...inputs,
          max_new_tokens: message.maxNewTokens,
          temperature,
          do_sample: temperature > 0.01,
          repetition_penalty: message.repetitionPenalty,
          top_p: message.topP,
          streamer,
          stopping_criteria: interrupt,
        });

      try {
        await runGenerate(model);
        post({ type: interrupt.interrupted || abortRequested ? "aborted" : "done" });
      } catch (err) {
        if (chatSlot.device === "webgpu" && isWebGpuRuntimeError(err)) {
          post({
            type: "status",
            text: "WebGPU נכשל (זיכרון GPU / שיחה ארוכה) — עובר ל-WASM ומנסה שוב…",
          });
          try {
            const switched = await forceReloadWasm(message.modelId);
            chatSlot.model = switched.model;
            chatSlot.processor = switched.processor;
            chatSlot.modelId = message.modelId;
            chatSlot.device = switched.device;
            post({
              type: "loaded",
              modelId: chatSlot.modelId,
              device: chatSlot.device,
            });
            await runGenerate(switched.model);
            post({ type: interrupt.interrupted || abortRequested ? "aborted" : "done" });
          } catch (retryErr) {
            post({
              type: "error",
              error: formatHubLoadError(retryErr),
              scope: "chat",
            });
          }
        } else {
          post({
            type: "error",
            error: err instanceof Error ? err.message : String(err),
            scope: "chat",
          });
        }
      } finally {
        chatBusy = false;
        activeInterrupt = null;
      }
      return;
    }

    if (message.type === "analyze_scene") {
      if (chatBusy) {
        post({
          type: "scene_analysis",
          requestId: message.requestId,
          ok: false,
          error: "chat_active",
        });
        return;
      }

      if (sceneBusy) {
        post({
          type: "scene_analysis",
          requestId: message.requestId,
          ok: false,
          error: "scene_busy",
        });
        return;
      }

      if (message.modelId !== GEMMA_MODEL_ID) {
        post({
          type: "scene_analysis",
          requestId: message.requestId,
          ok: false,
          error: `Only ${GEMMA_MODEL_ID} is supported.`,
        });
        return;
      }

      if (!message.images.length) {
        post({
          type: "scene_analysis",
          requestId: message.requestId,
          ok: false,
          error: "Model or image not ready.",
        });
        return;
      }

      sceneBusy = true;
      let fullText = "";
      try {
        const { processor } = await ensureChatSlot(message.modelId);
        await runSceneGenerateWithFallback(message.modelId, async (model) => {
          const inputs = await buildSceneAnalysisInputs(processor, message);
          const streamer = new TextStreamer(processor.tokenizer as never, {
            skip_prompt: true,
            callback_function: (text: string) => {
              fullText += text;
            },
          });

          await model.generate({
            ...inputs,
            max_new_tokens: 280,
            temperature: 0.1,
            do_sample: false,
            repetition_penalty: 1.05,
            top_p: 0.9,
            streamer,
          });
        });

        const parsed = parseSceneAnalysisJson(fullText);
        if (parsed) {
          post({
            type: "scene_analysis",
            requestId: message.requestId,
            ok: true,
            objects: parsed.objects ?? parsed.current ?? [],
            people: parsed.people ?? [],
            current: parsed.current ?? parsed.objects ?? [],
            events: parsed.events ?? [],
            interesting: parsed.interesting ?? false,
            summary: parsed.summary ?? "",
            raw: fullText.trim(),
          });
        } else {
          post({
            type: "scene_analysis",
            requestId: message.requestId,
            ok: true,
            current: [],
            events: [],
            interesting: false,
            summary: fullText.trim().slice(0, 400),
            raw: fullText.trim(),
          });
        }
      } catch (error) {
        post({
          type: "scene_analysis",
          requestId: message.requestId,
          ok: false,
          error: error instanceof Error ? error.message : String(error),
        });
      } finally {
        sceneBusy = false;
      }
      return;
    }

    if (message.type === "character_utterance") {
      if (chatBusy) {
        post({
          type: "character_utterance",
          requestId: message.requestId,
          ok: false,
          error: "chat_active",
        });
        return;
      }
      if (sceneBusy) {
        post({
          type: "character_utterance",
          requestId: message.requestId,
          ok: false,
          error: "scene_busy",
        });
        return;
      }
      if (message.modelId !== GEMMA_MODEL_ID) {
        post({
          type: "character_utterance",
          requestId: message.requestId,
          ok: false,
          error: `Only ${GEMMA_MODEL_ID} is supported.`,
        });
        return;
      }
      sceneBusy = true;
      let fullText = "";
      try {
        const { processor } = await ensureChatSlot(message.modelId);
        await runSceneGenerateWithFallback(message.modelId, async (model) => {
          const inputs = await buildCharacterUtteranceInputs(processor, message);
          const streamer = new TextStreamer(processor.tokenizer as never, {
            skip_prompt: true,
            callback_function: (text: string) => {
              fullText += text;
            },
          });
          await model.generate({
            ...inputs,
            max_new_tokens: message.maxNewTokens ?? 80,
            temperature: 0.45,
            do_sample: true,
            repetition_penalty: 1.08,
            top_p: 0.9,
            streamer,
          });
        });
        post({
          type: "character_utterance",
          requestId: message.requestId,
          ok: true,
          text: fullText.trim(),
        });
      } catch (error) {
        post({
          type: "character_utterance",
          requestId: message.requestId,
          ok: false,
          error: error instanceof Error ? error.message : String(error),
        });
      } finally {
        sceneBusy = false;
      }
      return;
    }
  } catch (error) {
    chatBusy = false;
    sceneBusy = false;
    activeInterrupt = null;
    postLoadFailureOnce(error);
  }
};
