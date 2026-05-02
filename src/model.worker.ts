/// <reference lib="webworker" />

import { TextStreamer, env, pipeline } from "@huggingface/transformers";

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

type WorkerInput =
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

let generator: TextGenerator | null = null;
const textGeneratorCache = new Map<string, { generator: TextGenerator; device: "webgpu" | "wasm" }>();
const captionerCache = new Map<
  string,
  (image: string, options?: Record<string, unknown>) => Promise<Array<{ generated_text: string }>>
>();
let activeModel = "";
let activeDevice = "unknown";
let busy = false;

env.allowLocalModels = false;
env.useBrowserCache = true;

const post = (msg: unknown) => {
  self.postMessage(msg);
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

const normalizeProgressStatus = (status?: string, percent?: number) => {
  const raw = (status ?? "").trim().toLowerCase();
  if (!raw || raw === "progress") {
    return (percent ?? 0) >= 100 ? "Finalizing model runtime..." : "Downloading model files...";
  }
  return status ?? "Downloading model files...";
};

const loadWithDevice = async (
  modelId: string,
  dtype: LoadMessage["dtype"],
  device: "webgpu" | "wasm",
) => {
  post({ type: "status", text: `Loading ${modelId} on ${device}...` });
  const startedAt = Date.now();
  let lastLoaded = 0;
  let lastTs = startedAt;
  const pipe = (await pipeline("text-generation", modelId, {
    device,
    dtype,
    progress_callback: (progressData: {
      status?: string;
      progress?: number;
      loaded?: number;
      total?: number;
      file?: string;
      name?: string;
    }) => {
      const percent = clampPercent(progressData.progress);
      const loaded = typeof progressData.loaded === "number" ? progressData.loaded : 0;
      const total = typeof progressData.total === "number" ? progressData.total : 0;
      const now = Date.now();
      const dt = Math.max(1, (now - lastTs) / 1000);
      const speed = loaded > 0 && loaded >= lastLoaded ? (loaded - lastLoaded) / dt : 0;
      lastLoaded = loaded;
      lastTs = now;
      const elapsed = Math.max(1, (now - startedAt) / 1000);
      const speedText = speed > 0 ? `${fmtBytes(speed)}/s` : "calculating...";
      const detailText =
        loaded > 0 && total > 0
          ? `${fmtBytes(loaded)} / ${fmtBytes(total)} (${speedText})`
          : `elapsed ${elapsed.toFixed(1)}s`;
      const statusText = normalizeProgressStatus(progressData.status, percent);

      post({
        type: "progress",
        text: statusText,
        progress: clampProgress(percent),
        detail: detailText,
        file: progressData.file ?? progressData.name ?? "",
      });
    },
  })) as TextGenerator;

  post({
    type: "progress",
    text: "Finalizing model runtime...",
    progress: 100,
    detail: "Preparing tokenizer and graph in your browser...",
    file: "",
  });

  return pipe;
};

const loadCaptioner = async (modelId: string, device: "webgpu" | "wasm") => {
  const visionModel = modelId;
  post({ type: "status", text: `Loading ${visionModel} on ${device}...` });
  return (await pipeline("image-to-text", visionModel, {
    device,
    dtype: "q8",
  })) as (image: string, options?: Record<string, unknown>) => Promise<Array<{ generated_text: string }>>;
};

const loadTextGenerator = async (modelId: string, dtype: LoadMessage["dtype"]) => {
  const cached = textGeneratorCache.get(modelId);
  if (cached) return cached;
  try {
    const loaded = await loadWithDevice(modelId, dtype, "webgpu");
    const entry = { generator: loaded, device: "webgpu" as const };
    textGeneratorCache.set(modelId, entry);
    return entry;
  } catch {
    post({ type: "status", text: `WebGPU unavailable for ${modelId}. Falling back to WASM...` });
    const loaded = await loadWithDevice(modelId, dtype, "wasm");
    const entry = { generator: loaded, device: "wasm" as const };
    textGeneratorCache.set(modelId, entry);
    return entry;
  }
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
    if (message.type === "load") {
      if (busy) {
        post({ type: "error", error: "Generation in progress. Please wait." });
        return;
      }

      if (generator && message.modelId === activeModel) {
        post({ type: "loaded", modelId: activeModel, device: activeDevice });
        return;
      }
      const loaded = await loadTextGenerator(message.modelId, message.dtype);
      generator = loaded.generator;
      activeDevice = loaded.device;
      activeModel = message.modelId;
      post({ type: "loaded", modelId: activeModel, device: activeDevice });
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
        text: "Starting full local model download...",
        progress: 0,
        detail: `0 / ${total} models ready`,
        file: "",
      });
      const failedTextModelIds: string[] = [];
      const failedCaptionModelIds: string[] = [];
      for (const textModelId of message.textModelIds) {
        post({ type: "status", text: `Preparing text model ${completed + 1}/${total}: ${textModelId}` });
        try {
          if (!textGeneratorCache.has(textModelId)) {
            await loadTextGenerator(textModelId, message.dtype);
          }
        } catch (e) {
          const err = e instanceof Error ? e.message : String(e);
          failedTextModelIds.push(textModelId);
          post({ type: "status", text: `Could not load ${textModelId}: ${err}` });
        }
        completed += 1;
        post({
          type: "progress",
          text: "Downloaded local models",
          progress: clampProgress((completed / total) * 100),
          detail: `${completed} / ${total} models ready`,
          file: textModelId,
        });
      }
      for (const captionModelId of message.captionModelIds) {
        post({ type: "status", text: `Preparing vision model ${completed + 1}/${total}: ${captionModelId}` });
        if (!captionerCache.has(captionModelId)) {
          try {
            let captioner: (image: string, options?: Record<string, unknown>) => Promise<Array<{ generated_text: string }>>;
            try {
              captioner = await loadCaptioner(captionModelId, "webgpu");
            } catch {
              captioner = await loadCaptioner(captionModelId, "wasm");
            }
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
          text: "Downloaded local models",
          progress: clampProgress((completed / total) * 100),
          detail: `${completed} / ${total} models ready`,
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
        try {
          if (!textGeneratorCache.has(textModelId)) {
            await loadTextGenerator(textModelId, message.dtype);
          }
        } catch (e) {
          const err = e instanceof Error ? e.message : String(e);
          failedTextModelIds.push(textModelId);
          post({ type: "status", text: `Warmup failed for ${textModelId}: ${err}` });
        }
        completed += 1;
        post({
          type: "progress",
          text: "Background model warmup",
          progress: clampProgress((completed / total) * 100),
          detail: `${completed} / ${total} background models ready`,
          file: textModelId,
        });
      }
      for (const captionModelId of message.captionModelIds) {
        if (!captionerCache.has(captionModelId)) {
          try {
            let captioner: (image: string, options?: Record<string, unknown>) => Promise<Array<{ generated_text: string }>>;
            try {
              captioner = await loadCaptioner(captionModelId, "webgpu");
            } catch {
              captioner = await loadCaptioner(captionModelId, "wasm");
            }
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
          text: "Background model warmup",
          progress: clampProgress((completed / total) * 100),
          detail: `${completed} / ${total} background models ready`,
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
      generator = null;
      textGeneratorCache.clear();
      captionerCache.clear();
      activeModel = "";
      activeDevice = "unknown";
      busy = false;
      post({ type: "status", text: "Runtime model cache cleared." });
      return;
    }

    if (message.type === "generate") {
      if (!generator) {
        post({ type: "error", error: "Model is not loaded yet." });
        return;
      }
      if (busy) {
        post({ type: "error", error: "Generation already in progress." });
        return;
      }

      if (message.modelId !== activeModel) {
        const switched = await loadTextGenerator(message.modelId, "q4");
        generator = switched.generator;
        activeModel = message.modelId;
        activeDevice = switched.device;
      }

      busy = true;
      const finalPrompt = buildPrompt(generator, message);
      const streamer = new TextStreamer(generator.tokenizer as never, {
        skip_prompt: true,
        callback_function: (text: string) => {
          post({ type: "token", text });
        },
      });

      const temperature = message.temperature;
      await generator(finalPrompt, {
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
        try {
          captioner = await loadCaptioner(message.modelId, "webgpu");
        } catch {
          captioner = await loadCaptioner(message.modelId, "wasm");
        }
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

      let device: "webgpu" | "wasm" = "webgpu";
      let captioner: (image: string, options?: Record<string, unknown>) => Promise<Array<{ generated_text: string }>>;
      try {
        captioner = await loadCaptioner(message.modelId, "webgpu");
        device = "webgpu";
      } catch {
        captioner = await loadCaptioner(message.modelId, "wasm");
        device = "wasm";
      }
      captionerCache.set(message.modelId, captioner);
      post({ type: "caption_model_loaded", modelId: message.modelId, device });
    }
  } catch (error) {
    busy = false;
    const text = error instanceof Error ? error.message : "Unknown error";
    post({ type: "error", error: text });
  }
};
