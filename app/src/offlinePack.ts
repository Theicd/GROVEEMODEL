/**
 * Offline pack metadata + pure helpers for the "Download all models for offline use"
 * action in Settings. The actual download orchestration runs in App.tsx (it has the
 * worker handle), but the size table and the progress/phase formatting live here so
 * they can be unit-tested directly.
 */

import { formatBytes } from "./storageReport";

export type OfflinePackComponentId = "chat" | "code" | "caption" | "image";

export interface OfflinePackComponent {
  id: OfflinePackComponentId;
  label: string;
  /** Approximate disk footprint after download. */
  approxBytes: number;
  /** What the user sees rendered next to the size. */
  description: string;
}

/**
 * Conservative size estimates for the bundled defaults. Real disk usage depends on
 * dtype/quantization and whether ONNX shards are deduplicated; these are the numbers
 * we display to the user before they click "Download".
 */
export const OFFLINE_PACK_COMPONENTS: ReadonlyArray<OfflinePackComponent> = [
  {
    id: "chat",
    label: "Gemma 4 — צ׳אט (ברירת מחדל)",
    approxBytes: 2.0 * 1024 ** 3,
    description: "מודל השפה הראשי (q4 ONNX) — תשובות בעברית/אנגלית.",
  },
  {
    id: "code",
    label: "Qwen 2.5 Coder — קוד",
    approxBytes: 0.5 * 1024 ** 3,
    description: "מודל קוד שמופעל אוטומטית כשמבקשים תוכנית/פונקציה.",
  },
  {
    id: "caption",
    label: "ViT-GPT2 — תיאור תמונה",
    approxBytes: 0.25 * 1024 ** 3,
    description: "מתאר תמונות שאתה מצרף לצ׳אט.",
  },
  {
    id: "image",
    label: "SD-Turbo — יצירת תמונות",
    approxBytes: 2.3 * 1024 ** 3,
    description: "מודל דיפוזיה לציור תמונות בדפדפן (web-txt2img).",
  },
] as const;

export const offlinePackTotalBytes = (): number =>
  OFFLINE_PACK_COMPONENTS.reduce((s, c) => s + c.approxBytes, 0);

export type OfflinePackPhase = "idle" | "models" | "image" | "done" | "error";

export interface OfflinePackUiState {
  phase: OfflinePackPhase;
  /** 0..100 across the whole pack (models phase = 0..70, image phase = 70..100). */
  overallPct: number;
  detail: string;
  error: string | null;
  completedAt: number | null;
}

export const initialOfflinePackState = (completedAt: number | null = null): OfflinePackUiState => ({
  phase: completedAt ? "done" : "idle",
  overallPct: completedAt ? 100 : 0,
  detail: "",
  error: null,
  completedAt,
});

/** Worker progress (`preload_all` 0..100) → overall pack progress, phase 1 only. */
export const mapModelsPhaseProgress = (workerPct: number): number => {
  const clamped = Math.max(0, Math.min(100, Math.round(workerPct)));
  return Math.round((clamped / 100) * 70);
};

/** SD-Turbo loader progress text → overall pack progress, phase 2 only. */
export const mapImagePhaseProgress = (status: string): number => {
  const m = status.match(/(\d{1,3})\s*%/);
  if (!m) return 70;
  const n = Math.max(0, Math.min(100, Number(m[1])));
  return 70 + Math.round((n / 100) * 30);
};

/**
 * Formatted "Last downloaded on …" text (Hebrew, locale `he-IL`). Returns empty
 * string for null / invalid timestamps.
 */
export const formatOfflinePackCompletedAt = (ts: number | null | undefined): string => {
  if (!ts || !Number.isFinite(ts)) return "";
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return "";
  return `הורדה אחרונה: ${d.toLocaleString("he-IL")}`;
};

export const offlinePackTotalLabel = (): string => formatBytes(offlinePackTotalBytes());

export const offlinePackComponentLabel = (c: OfflinePackComponent): string =>
  `${c.label} · ${formatBytes(c.approxBytes)}`;

export const OFFLINE_PACK_LOCAL_STORAGE_KEY = "grovee_offline_pack_completed_at_v1";

export function readOfflinePackCompletedAt(): number | null {
  try {
    const raw = localStorage.getItem(OFFLINE_PACK_LOCAL_STORAGE_KEY);
    if (!raw) return null;
    const n = Number(raw);
    return Number.isFinite(n) && n > 0 ? n : null;
  } catch {
    return null;
  }
}

export function writeOfflinePackCompletedAt(ts: number): void {
  try {
    localStorage.setItem(OFFLINE_PACK_LOCAL_STORAGE_KEY, String(ts));
  } catch {
    /* ignore quota / private mode */
  }
}

export function clearOfflinePackCompletedAt(): void {
  try {
    localStorage.removeItem(OFFLINE_PACK_LOCAL_STORAGE_KEY);
  } catch {
    /* ignore */
  }
}
