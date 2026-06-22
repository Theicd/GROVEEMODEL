import { markLocalTextReady } from "./localTextModels";
import type { LocalTextInferenceBackend } from "./localTextModelSettings";

export type LocalTextDownloadProgress = {
  pct: number;
  message: string;
  loaded: number;
  total: number;
};

type WorkerOut =
  | { type: "progress"; modelId: string; pct: number; message: string; loaded: number; total: number }
  | { type: "loaded"; modelId: string; device: string }
  | { type: "status"; modelId: string; text: string }
  | { type: "token"; text: string }
  | { type: "complete"; text: string }
  | { type: "aborted"; scope?: string }
  | { type: "error"; error: string; scope?: string };

let worker: Worker | null = null;

function getWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL("./textModel.worker.ts", import.meta.url), { type: "module" });
  }
  return worker;
}

export function downloadLocalTextModel(
  rackId: string,
  modelId: string,
  onProgress: (p: LocalTextDownloadProgress) => void,
  backend: LocalTextInferenceBackend = "auto",
): Promise<void> {
  const w = getWorker();
  return new Promise((resolve, reject) => {
    const onMessage = (ev: MessageEvent<WorkerOut>) => {
      const data = ev.data;
      if (data.type === "progress" && data.modelId === modelId) {
        onProgress({
          pct: data.pct,
          message: data.message,
          loaded: data.loaded,
          total: data.total,
        });
      }
      if (data.type === "loaded" && data.modelId === modelId) {
        markLocalTextReady(rackId);
        w.removeEventListener("message", onMessage);
        resolve();
      }
      if (data.type === "error") {
        w.removeEventListener("message", onMessage);
        reject(new Error(data.error));
      }
    };
    w.addEventListener("message", onMessage);
    w.postMessage({ type: "load", modelId, backend });
  });
}

export type LocalTextGenerateOptions = {
  modelId: string;
  systemPrompt: string;
  history: { role: "user" | "assistant"; content: string }[];
  prompt: string;
  maxNewTokens?: number;
  temperature?: number;
  topP?: number;
  backend?: LocalTextInferenceBackend;
  onToken: (text: string) => void;
  onStatus?: (text: string) => void;
};

export function generateLocalTextChat(opts: LocalTextGenerateOptions): Promise<string> {
  const w = getWorker();
  return new Promise((resolve, reject) => {
    let full = "";
    const onMessage = (ev: MessageEvent<WorkerOut>) => {
      const data = ev.data;
      if (data.type === "token") {
        full += data.text;
        opts.onToken(data.text);
      }
      if (data.type === "status") {
        opts.onStatus?.(data.text);
      }
      if (data.type === "complete") {
        w.removeEventListener("message", onMessage);
        resolve(data.text || full);
      }
      if (data.type === "aborted") {
        w.removeEventListener("message", onMessage);
        resolve(full);
      }
      if (data.type === "error" && data.scope === "chat") {
        w.removeEventListener("message", onMessage);
        reject(new Error(data.error));
      }
    };
    w.addEventListener("message", onMessage);
    w.postMessage({
      type: "generate",
      modelId: opts.modelId,
      systemPrompt: opts.systemPrompt,
      history: opts.history,
      prompt: opts.prompt,
      maxNewTokens: opts.maxNewTokens ?? 256,
      temperature: opts.temperature ?? 0.7,
      topP: opts.topP ?? 0.9,
      backend: opts.backend ?? "auto",
    });
  });
}

export function abortLocalTextGeneration(): void {
  worker?.postMessage({ type: "abort" });
}

export function terminateLocalTextWorker(): void {
  worker?.terminate();
  worker = null;
}
