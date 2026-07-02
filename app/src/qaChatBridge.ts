import type { ModelActivityEntry } from "./modelActivityLog";

export type QaReplySource =
  | "model"
  | "rack"
  | "canned-live"
  | "canned-globe"
  | "canned-game"
  | "local-time"
  | "meta-capabilities"
  | "disambiguation"
  | "reset"
  | "capabilities-only"
  | "local-text"
  | "session-memory"
  | "greeting"
  | "unknown";

export type QaTurnResult = {
  query: string;
  reply: string;
  replySource: QaReplySource;
  usedModel: boolean;
  webContextSent: string;
  modelPromptOut: string;
  modelResponseIn: string;
  searchProviders: string[];
  searchSummary: string;
  ms: number;
  error?: string;
};

type QaHandlers = {
  ready: () => boolean;
  newChat: () => void;
  submit: (text: string, forceLlm: boolean) => Promise<void>;
  getActivity: () => ModelActivityEntry[];
};

type PendingTurn = {
  query: string;
  forceLlm: boolean;
  started: number;
  resolve: (r: QaTurnResult) => void;
  reject: (e: Error) => void;
};

let handlers: QaHandlers | null = null;
let pending: PendingTurn | null = null;
let replySource: QaReplySource = "unknown";
let lastWebContext = "";

export const qaChatBridge = {
  register(h: QaHandlers) {
    handlers = h;
  },
  unregister() {
    handlers = null;
  },
  setReplySource(source: QaReplySource) {
    replySource = source;
  },
  setWebContext(ctx: string) {
    lastWebContext = ctx;
  },
  getReplySource(): QaReplySource {
    return replySource;
  },
  isForceLlmPending(): boolean {
    return pending?.forceLlm ?? false;
  },
  hasPending(): boolean {
    return pending !== null;
  },
  notifyTurnComplete(reply: string, searchSummary?: string, searchProviders?: string[]) {
    if (!pending) return;
    const activity = handlers?.getActivity() ?? [];
    const outGen = activity.find((e) => e.direction === "out" && e.kind === "generate");
    const inGen = activity.find((e) => e.direction === "in" && e.kind === "generate");
    const usedModel = replySource === "model";
    const webFromActivity = outGen?.detail?.includes("WEB CONTEXT:")
      ? outGen.detail.split("WEB CONTEXT:").slice(1).join("WEB CONTEXT:").trim()
      : "";
    const result: QaTurnResult = {
      query: pending.query,
      reply: reply.trim(),
      replySource,
      usedModel,
      webContextSent: (lastWebContext || webFromActivity).slice(0, 4000),
      modelPromptOut: (outGen?.detail ?? "").slice(0, 4000),
      modelResponseIn: (inGen?.detail ?? reply).slice(0, 4000),
      searchProviders: searchProviders ?? [],
      searchSummary: searchSummary ?? "",
      ms: Date.now() - pending.started,
    };
    pending.resolve(result);
    pending = null;
    replySource = "unknown";
    lastWebContext = "";
  },
  notifyTurnFailed(error: string) {
    if (!pending) return;
    pending.reject(new Error(error));
    pending = null;
    replySource = "unknown";
    lastWebContext = "";
  },
  async ask(query: string, opts?: { forceLlm?: boolean; newChat?: boolean }): Promise<QaTurnResult> {
    if (!handlers) throw new Error("QA bridge not registered — open ?qa=chat in dev mode");
    if (!handlers.ready()) throw new Error("Model not loaded yet");
    if (pending) throw new Error("Another QA turn is in progress");
    if (opts?.newChat) {
      handlers.newChat();
      await new Promise((r) => window.setTimeout(r, 300));
    }
    return new Promise<QaTurnResult>((resolve, reject) => {
      const timeoutMs = 360_000;
      const timeoutId = window.setTimeout(() => {
        if (!pending) return;
        pending.reject(new Error("QA timeout — לא התקבלה תשובה תוך 6 דקות"));
        pending = null;
      }, timeoutMs);

      const finish = (fn: () => void) => {
        window.clearTimeout(timeoutId);
        fn();
      };

      pending = {
        query: query.trim(),
        forceLlm: opts?.forceLlm ?? false,
        started: Date.now(),
        resolve: (r) => finish(() => resolve(r)),
        reject: (err) => finish(() => reject(err)),
      };
      replySource = "unknown";
      void handlers!
        .submit(query.trim(), opts?.forceLlm ?? false)
        .catch((err) => {
          if (!pending) return;
          pending.reject(err instanceof Error ? err : new Error(String(err)));
          pending = null;
          window.clearTimeout(timeoutId);
        });
    });
  },
  ready(): boolean {
    return handlers?.ready() ?? false;
  },
  newChat() {
    handlers?.newChat();
  },
};

declare global {
  interface Window {
    __groveeQa?: {
      ready: () => boolean;
      ask: (query: string, opts?: { forceLlm?: boolean; newChat?: boolean }) => Promise<QaTurnResult>;
      newChat: () => void;
    };
  }
}

export function exposeGroveeQaWindow() {
  window.__groveeQa = {
    ready: () => qaChatBridge.ready(),
    ask: (q, opts) => qaChatBridge.ask(q, opts),
    newChat: () => qaChatBridge.newChat(),
  };
}
