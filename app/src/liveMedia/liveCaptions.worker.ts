// @ts-nocheck
/// <reference lib="webworker" />

import { env, pipeline } from "@huggingface/transformers";

/** fp32 WASM is slower but avoids broken q8 ONNX weight scales on some builds. */
const MODEL_ID = "Xenova/whisper-tiny";

env.allowLocalModels = false;
env.useBrowserCache = true;

let transcriber: Awaited<ReturnType<typeof pipeline>> | null = null;
let loading: Promise<Awaited<ReturnType<typeof pipeline>>> | null = null;

const post = (data: unknown) => self.postMessage(data);

type HfProgress = {
  status?: string;
  file?: string;
  progress?: number;
  loaded?: number;
  total?: number;
};

type LoadAttempt = { device: "wasm" | "webgpu"; dtype: "fp32" | "q4" };

const LOAD_ATTEMPTS: LoadAttempt[] = [
  { device: "wasm", dtype: "fp32" },
  { device: "wasm", dtype: "q4" },
  { device: "webgpu", dtype: "q4" },
];

function progressFromHf(p: HfProgress): { pct: number; message: string } {
  const loaded = typeof p.loaded === "number" ? p.loaded : 0;
  const total = typeof p.total === "number" ? p.total : 0;
  let pct = 0;
  if (loaded > 0 && total > 0) pct = Math.round((loaded / total) * 100);
  else if (typeof p.progress === "number") {
    pct = p.progress <= 1 ? Math.round(p.progress * 100) : Math.round(p.progress);
  }
  const file = p.file ? (p.file.split("/").pop() ?? p.file) : "";
  const message = file ? `Downloading ${file}` : "Loading speech model…";
  return { pct, message };
}

function isLoadFailure(err: unknown): boolean {
  const msg = err instanceof Error ? err.message : String(err);
  return /session|onnx|scale|dequant|webgpu|ortrun|wgpu/i.test(msg);
}

async function loadTranscriber(attempt: LoadAttempt) {
  post({ type: "progress", pct: 1, message: `Loading on ${attempt.device}/${attempt.dtype}…` });
  return pipeline("automatic-speech-recognition", MODEL_ID, {
    device: attempt.device,
    dtype: attempt.dtype,
    progress_callback: (p: HfProgress) => {
      const st = p.status ?? "";
      if (st === "progress" || st === "progress_total" || st === "download" || st === "initiate") {
        const { pct, message } = progressFromHf(p);
        post({ type: "progress", pct, message });
      }
    },
  });
}

async function ensureTranscriber() {
  if (transcriber) return transcriber;
  if (!loading) {
    loading = (async () => {
      let lastErr: unknown = null;
      for (const attempt of LOAD_ATTEMPTS) {
        try {
          const pipe = await loadTranscriber(attempt);
          transcriber = pipe;
          return pipe;
        } catch (err) {
          lastErr = err;
          transcriber = null;
          if (!isLoadFailure(err)) throw err;
        }
      }
      throw lastErr instanceof Error ? lastErr : new Error("model-load-failed");
    })();
  }
  return loading;
}

function extractWordTimings(out: unknown): { text: string; words: Array<{ text: string; start: number; end: number }> } {
  if (typeof out === "string") {
    const parts = out.trim().split(/\s+/).filter(Boolean);
    return {
      text: out.trim(),
      words: parts.map((w, i) => ({ text: w, start: i * 0.28, end: (i + 1) * 0.28 })),
    };
  }
  const obj = out as { text?: string; chunks?: Array<{ text?: string; timestamp?: [number, number] }> };
  const text = String(obj?.text ?? "").trim();
  const words = (obj?.chunks ?? [])
    .map((ch) => ({
      text: String(ch.text ?? "").trim(),
      start: Array.isArray(ch.timestamp) ? ch.timestamp[0] : 0,
      end: Array.isArray(ch.timestamp) ? ch.timestamp[1] : 0,
    }))
    .filter((w) => w.text);
  if (words.length) return { text, words };
  const parts = text.split(/\s+/).filter(Boolean);
  return {
    text,
    words: parts.map((w, i) => ({ text: w, start: i * 0.28, end: (i + 1) * 0.28 })),
  };
}

self.onmessage = async (ev: MessageEvent) => {
  const msg = ev.data as {
    type: string;
    audio?: Float32Array;
    language?: string;
    id?: number;
  };

  try {
    if (msg.type === "load") {
      await ensureTranscriber();
      post({ type: "ready" });
      return;
    }

    if (msg.type === "transcribe" && msg.audio) {
      const pipe = await ensureTranscriber();
      const out = await pipe(msg.audio, {
        language: msg.language ?? "english",
        task: "transcribe",
        return_timestamps: "word",
        temperature: 0,
        compression_ratio_threshold: 1.8,
        logprob_threshold: -1,
        no_speech_threshold: 0.65,
      });
      const parsed = extractWordTimings(out);
      post({ type: "result", id: msg.id ?? 0, text: parsed.text, words: parsed.words });
      return;
    }

    if (msg.type === "dispose") {
      transcriber = null;
      loading = null;
      post({ type: "disposed" });
    }
  } catch (err) {
    post({
      type: "error",
      id: msg.id ?? 0,
      message: err instanceof Error ? err.message : "transcribe-failed",
    });
  }
};
