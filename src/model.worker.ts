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
  repetitionPenalty: number;
  topP: number;
  thinkingMode: boolean;
  webContext: string;
};

type WorkerInput = LoadMessage | GenerateMessage;

type TextGenerator = ((
  input: string,
  options: Record<string, unknown>,
) => Promise<unknown>) & { tokenizer: unknown };

let generator: TextGenerator | null = null;
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

const loadWithDevice = async (
  modelId: string,
  dtype: LoadMessage["dtype"],
  device: "webgpu" | "wasm",
) => {
  post({ type: "status", text: `Loading ${modelId} on ${device}...` });
  const pipe = (await pipeline("text-generation", modelId, {
    device,
    dtype,
    progress_callback: (progressData: { status?: string; progress?: number }) => {
      post({
        type: "progress",
        text: progressData.status ?? "Downloading model files...",
        progress: clampProgress((progressData.progress ?? 0) * 100),
      });
    },
  })) as TextGenerator;

  return pipe;
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
    }
  } catch (error) {
    busy = false;
    const text = error instanceof Error ? error.message : "Unknown error";
    post({ type: "error", error: text });
  }
};
