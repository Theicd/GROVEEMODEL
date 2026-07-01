import { markLocalTextReady } from "./localTextModels";
import type { LocalTextInferenceBackend } from "./localTextModelSettings";
import { pushWorkerLog, type ConsoleLogLevel } from "../consoleLogStore";

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
  | { type: "error"; error: string; scope?: "load" | "chat" };

let worker: Worker | null = null;

function getWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL("./textModel.worker.ts", import.meta.url), { type: "module" });
    // Mirror the worker's forwarded console into the in-app console panel.
    worker.addEventListener("message", (ev: MessageEvent) => {
      const d = ev.data as { type?: string; level?: ConsoleLogLevel; text?: string } | null;
      if (d && d.type === "__wlog" && typeof d.text === "string") {
        pushWorkerLog(d.level ?? "log", `[SmolLM] ${d.text}`);
      }
    });
  }
  return worker;
}

/**
 * If the worker goes silent for this long (no progress/status/loaded/error), we treat
 * the load as hung and fail loudly. On mobile a crashed WASM instantiation (e.g. OOM)
 * can die without ever posting an "error", which previously left the UI stuck on
 * "loading" forever. The watchdog resets on every message, so slow-but-progressing
 * downloads on mobile networks are not affected.
 */
const LOCAL_TEXT_LOAD_STALL_MS = 120_000;

export function downloadLocalTextModel(
  rackId: string,
  modelId: string,
  onProgress: (p: LocalTextDownloadProgress) => void,
  backend: LocalTextInferenceBackend = "auto",
): Promise<void> {
  const w = getWorker();
  return new Promise((resolve, reject) => {
    let stallTimer: ReturnType<typeof setTimeout> | null = null;
    let settled = false;

    const cleanup = () => {
      settled = true;
      if (stallTimer) clearTimeout(stallTimer);
      stallTimer = null;
      w.removeEventListener("message", onMessage);
    };

    const armStall = () => {
      if (stallTimer) clearTimeout(stallTimer);
      stallTimer = setTimeout(() => {
        if (settled) return;
        cleanup();
        // A hung worker is unrecoverable — terminate so the next attempt starts clean.
        terminateLocalTextWorker();
        reject(
          new Error(
            "Model load stalled (no progress). The device may be low on memory — " +
              "try closing tabs/apps and loading again.",
          ),
        );
      }, LOCAL_TEXT_LOAD_STALL_MS);
    };

    const onMessage = (ev: MessageEvent<WorkerOut>) => {
      const data = ev.data;
      if (settled) return;
      // Any activity for this model counts as progress toward completion.
      if (
        (data.type === "progress" || data.type === "status" || data.type === "loaded") &&
        data.modelId === modelId
      ) {
        armStall();
      }
      if (data.type === "progress" && data.modelId === modelId) {
        onProgress({
          pct: data.pct,
          message: data.message,
          loaded: data.loaded,
          total: data.total,
        });
      }
      if (data.type === "status" && data.modelId === modelId) {
        onProgress({
          pct: 0,
          message: data.text,
          loaded: 0,
          total: 0,
        });
      }
      if (data.type === "loaded" && data.modelId === modelId) {
        console.info("[GROVEE:boot]", "SmolLM worker loaded", {
          modelId: data.modelId,
          device: data.device,
          usesGpu: data.device === "webgpu",
        });
        markLocalTextReady(rackId);
        cleanup();
        resolve();
      }
      if (data.type === "error" && data.scope !== "chat") {
        cleanup();
        reject(new Error(data.error || "SmolLM load failed"));
      }
    };
    w.addEventListener("message", onMessage);
    armStall();
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
