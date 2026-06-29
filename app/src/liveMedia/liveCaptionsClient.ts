export type TimedWord = { text: string; start: number; end: number };

export type TranscribeResult = {
  text: string;
  words: TimedWord[];
};

let worker: Worker | null = null;
let readyPromise: Promise<void> | null = null;
let nextId = 1;
let progressHandler: ((pct: number, message: string) => void) | null = null;

const pending = new Map<number, { resolve: (r: TranscribeResult) => void; reject: (err: Error) => void }>();

function dispatchWorkerMessage(msg: {
  type: string;
  id?: number;
  text?: string;
  words?: TimedWord[];
  message?: string;
  pct?: number;
}) {
  if (msg.type === "progress" && progressHandler) {
    progressHandler(msg.pct ?? 0, msg.message ?? "");
    return;
  }
  if (msg.type === "result" && msg.id != null) {
    pending.get(msg.id)?.resolve({
      text: msg.text ?? "",
      words: msg.words ?? [],
    });
    pending.delete(msg.id);
    return;
  }
  if (msg.type === "error" && msg.id != null) {
    pending.get(msg.id)?.reject(new Error(msg.message ?? "transcribe-failed"));
    pending.delete(msg.id);
  }
}

function getWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL("./liveCaptions.worker.ts", import.meta.url), { type: "module" });
    worker.onmessage = (ev) => dispatchWorkerMessage(ev.data);
  }
  return worker;
}

export function setLiveCaptionsLoadProgress(handler: ((pct: number, message: string) => void) | null): void {
  progressHandler = handler;
}

export async function ensureLiveCaptionsModel(): Promise<void> {
  if (readyPromise) return readyPromise;
  readyPromise = new Promise<void>((resolve, reject) => {
    const w = getWorker();
    const onMsg = (ev: MessageEvent) => {
      const msg = ev.data as { type: string; message?: string };
      if (msg.type === "ready") {
        w.removeEventListener("message", onMsg);
        resolve();
      } else if (msg.type === "error" && !("id" in msg)) {
        w.removeEventListener("message", onMsg);
        readyPromise = null;
        reject(new Error(msg.message ?? "model-load-failed"));
      }
    };
    w.addEventListener("message", onMsg);
    w.postMessage({ type: "load" });
  });
  return readyPromise;
}

export async function transcribeLiveAudio(audio: Float32Array, language: string): Promise<TranscribeResult> {
  await ensureLiveCaptionsModel();
  const id = nextId++;
  const copy = new Float32Array(audio);
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    getWorker().postMessage({ type: "transcribe", audio: copy, language, id }, [copy.buffer]);
  });
}

export function disposeLiveCaptionsModel(): void {
  if (worker) {
    worker.postMessage({ type: "dispose" });
    worker.terminate();
  }
  worker = null;
  readyPromise = null;
  pending.clear();
}
