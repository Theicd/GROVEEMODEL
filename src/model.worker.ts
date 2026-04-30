/// <reference lib="webworker" />

import { TextStreamer, env, pipeline } from "@huggingface/transformers";

type LoadMessage = {
  type: "load";
  modelId: string;
  dtype: "q4" | "q8" | "fp16" | "fp32";
};

type GenerateMessage = {
  type: "generate";
  prompt: string;
  systemPrompt: string;
  maxNewTokens: number;
  temperature: number;
};

type WorkerInput = LoadMessage | GenerateMessage;

let generator: any = null;
let activeModel = "";
let activeDevice = "unknown";
let busy = false;

env.allowLocalModels = false;

const post = (msg: unknown) => {
  self.postMessage(msg);
};

const clampProgress = (value: number) => {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, Math.round(value)));
};

const loadWithDevice = async (modelId: string, dtype: LoadMessage["dtype"], device: "webgpu" | "wasm") => {
  post({ type: "status", text: `Loading ${modelId} on ${device}...` });
  const pipe = await pipeline("text-generation", modelId, {
    device,
    dtype,
    progress_callback: (progressData: { status?: string; progress?: number }) => {
      post({
        type: "progress",
        text: progressData.status ?? "Downloading model files...",
        progress: clampProgress((progressData.progress ?? 0) * 100),
      });
    },
  });

  return pipe;
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

      try {
        generator = await loadWithDevice(message.modelId, message.dtype, "webgpu");
        activeDevice = "webgpu";
      } catch {
        post({ type: "status", text: "WebGPU unavailable. Falling back to WASM..." });
        generator = await loadWithDevice(message.modelId, message.dtype, "wasm");
        activeDevice = "wasm";
      }

      activeModel = message.modelId;
      post({ type: "loaded", modelId: activeModel, device: activeDevice });
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

      busy = true;
      const finalPrompt = `${message.systemPrompt}\n\nUser: ${message.prompt}\nAssistant:`;
      const streamer = new TextStreamer(generator.tokenizer, {
        skip_prompt: true,
        callback_function: (text: string) => {
          post({ type: "token", text });
        },
      });

      await generator(finalPrompt, {
        max_new_tokens: message.maxNewTokens,
        temperature: message.temperature,
        do_sample: message.temperature > 0,
        streamer,
      });

      post({ type: "done" });
      busy = false;
    }
  } catch (error) {
    busy = false;
    const text = error instanceof Error ? error.message : "Unknown error";
    post({ type: "error", error: text });
  }
};
