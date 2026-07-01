/**
 * In-app console capture. Mirrors browser console output (plus uncaught errors and
 * forwarded worker logs) into a ring buffer so users on devices without DevTools —
 * phones especially — can view and copy background errors from a settings panel.
 */
export type ConsoleLogLevel = "log" | "info" | "warn" | "error" | "debug";
export type ConsoleLogSource = "main" | "worker";

export type ConsoleLogEntry = {
  id: number;
  ts: number;
  level: ConsoleLogLevel;
  source: ConsoleLogSource;
  text: string;
};

const MAX_ENTRIES = 600;
const buffer: ConsoleLogEntry[] = [];
let seq = 0;
let installed = false;
const listeners = new Set<() => void>();

function notify(): void {
  listeners.forEach((fn) => {
    try {
      fn();
    } catch {
      /* ignore listener errors */
    }
  });
}

function argToText(arg: unknown): string {
  if (typeof arg === "string") return arg;
  if (arg instanceof Error) return arg.stack ?? `${arg.name}: ${arg.message}`;
  if (arg === undefined) return "undefined";
  if (arg === null) return "null";
  try {
    return JSON.stringify(arg);
  } catch {
    return String(arg);
  }
}

/** Append a log entry. Safe to call from anywhere on the main thread. */
export function pushLog(level: ConsoleLogLevel, text: string, source: ConsoleLogSource = "main"): void {
  buffer.push({ id: ++seq, ts: Date.now(), level, source, text });
  if (buffer.length > MAX_ENTRIES) buffer.splice(0, buffer.length - MAX_ENTRIES);
  notify();
}

/**
 * Patch the global console so every call is also recorded. Idempotent. Should be
 * called as early as possible (before other global error hooks) so their
 * console.error output is captured too.
 */
export function installConsoleCapture(): void {
  if (installed || typeof console === "undefined") return;
  installed = true;

  const levels: ConsoleLogLevel[] = ["log", "info", "warn", "error", "debug"];
  for (const level of levels) {
    const original = (console[level] as ((...args: unknown[]) => void) | undefined)?.bind(console);
    (console as unknown as Record<string, (...args: unknown[]) => void>)[level] = (
      ...args: unknown[]
    ) => {
      try {
        pushLog(level, args.map(argToText).join(" "), "main");
      } catch {
        /* never let capture break logging */
      }
      original?.(...args);
    };
  }
}

/** Record a log line forwarded from a Web Worker (see textModel.worker forwarding). */
export function pushWorkerLog(level: ConsoleLogLevel, text: string): void {
  pushLog(level, text, "worker");
}

export function getConsoleLogs(): ConsoleLogEntry[] {
  return buffer.slice();
}

export function clearConsoleLogs(): void {
  buffer.length = 0;
  notify();
}

export function subscribeConsoleLogs(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/** Render entries as plain text for copy/paste / sharing. */
export function formatConsoleLogs(entries: ConsoleLogEntry[] = buffer): string {
  const ua = typeof navigator !== "undefined" ? navigator.userAgent : "unknown";
  const header = [
    `GROVEE console log — ${new Date().toISOString()}`,
    `UA: ${ua}`,
    `crossOriginIsolated: ${
      typeof self !== "undefined" && "crossOriginIsolated" in self
        ? String((self as unknown as { crossOriginIsolated?: boolean }).crossOriginIsolated)
        : "n/a"
    }`,
    "".padEnd(48, "-"),
  ].join("\n");
  const body = entries
    .map((e) => {
      const t = new Date(e.ts).toISOString().slice(11, 23);
      const src = e.source === "worker" ? "worker" : "main";
      return `[${t}] ${e.level.toUpperCase().padEnd(5)} (${src}) ${e.text}`;
    })
    .join("\n");
  return `${header}\n${body}\n`;
}
