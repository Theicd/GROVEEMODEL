/** Gemma 4 E2B q4 multimodal pack — ~3.9 GB (matches model.worker). */
export const GEMMA_ESTIMATED_BYTES = 3_900_000_000;

export function downloadProgressPercent(loaded: number, total: number): number {
  if (!Number.isFinite(loaded) || !Number.isFinite(total) || total <= 0) return 0;
  return Math.min(100, Math.max(0, (loaded / total) * 100));
}

/** HF may send 0–1 (ratio) or 0–100 (percent), often per-file — not overall pack. */
export function normalizeHfProgressPercent(value: number): number {
  if (!Number.isFinite(value) || value <= 0) return 0;
  return value <= 1 ? value * 100 : value;
}

/** Prefer byte ratio; ignore HF progress field when bytes are known. */
export function resolveDownloadPercent(opts: {
  loaded: number;
  total: number;
  hfProgress?: number;
}): number {
  const { loaded, total, hfProgress } = opts;
  if (Number.isFinite(loaded) && Number.isFinite(total) && total > 0 && loaded > 0) {
    return downloadProgressPercent(loaded, total);
  }
  if (typeof hfProgress === "number") {
    return Math.min(100, normalizeHfProgressPercent(hfProgress));
  }
  return 0;
}

export function sumFileProgressMap(
  files: Record<string, { loaded: number; total: number }> | undefined,
): { loaded: number; total: number } {
  if (!files) return { loaded: 0, total: 0 };
  let loaded = 0;
  let total = 0;
  for (const st of Object.values(files)) {
    if (typeof st.loaded === "number") loaded += st.loaded;
    if (typeof st.total === "number") total += st.total;
  }
  return { loaded, total };
}

export function formatDownloadPercent(percent: number): string {
  if (!Number.isFinite(percent) || percent <= 0) return "0";
  if (percent < 0.1) return percent.toFixed(2);
  if (percent < 1) return percent.toFixed(1);
  return String(Math.round(percent));
}
