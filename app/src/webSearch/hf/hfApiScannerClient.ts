import { fetchJson } from "../fetchJson";
import { getHfScannerBaseUrl, getHfToken } from "./hfModelSettings";
import { readWorkingModelsCache, writeWorkingModelsCache } from "./hfScannerCache";
import type { HfAccessMode, HfProbeResult } from "./hfModelTypes";

export type ScannerModelRow = {
  model_id: string;
  provider?: string;
  access_mode?: string;
  category?: string;
  organization?: string;
  size_param?: string;
  pipeline?: string;
  downloads?: number;
  likes?: number;
  latency?: number;
  status?: string;
  endpoint?: string;
  error_text?: string;
  test_response?: string;
};

let scannerHealthyCache: { at: number; ok: boolean } | null = null;
const HEALTH_TTL_MS = 30_000;

export async function isHfScannerAvailable(): Promise<boolean> {
  const base = getHfScannerBaseUrl();
  if (!base) return false;
  if (scannerHealthyCache && Date.now() - scannerHealthyCache.at < HEALTH_TTL_MS) {
    return scannerHealthyCache.ok;
  }
  try {
    await fetchJson<{ ok?: boolean }>(`${base}/api/health`, undefined, { timeoutMs: 2500 });
    scannerHealthyCache = { at: Date.now(), ok: true };
    return true;
  } catch {
    scannerHealthyCache = { at: Date.now(), ok: false };
    return false;
  }
}

export function scannerRowToProbe(row: ScannerModelRow): HfProbeResult {
  const access = (row.access_mode || "UNKNOWN").toUpperCase();
  const accessMode: HfAccessMode =
    access === "FREE" ? "FREE" : access === "TOKEN" ? "TOKEN" : "UNKNOWN";
  return {
    modelId: row.model_id,
    status: row.status || "UNKNOWN",
    provider: row.provider || "Unknown",
    accessMode,
    latency: row.latency,
    endpoint: row.endpoint || "https://router.huggingface.co/v1/chat/completions",
    errorText: row.error_text || undefined,
    testResponse: row.test_response,
  };
}

export async function testModelViaScanner(
  modelId: string,
  hfToken?: string,
): Promise<HfProbeResult | null> {
  const base = getHfScannerBaseUrl();
  if (!base) return null;
  try {
    const token = hfToken ?? getHfToken();
    const out = await fetchJson<{ result?: ScannerModelRow }>(
      `${base}/api/test-model`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: modelId }),
      },
      { timeoutMs: 25_000, headers: token ? { Authorization: `Bearer ${token}` } : undefined },
    );
    if (!out.result?.model_id) return null;
    return scannerRowToProbe(out.result);
  } catch {
    return null;
  }
}

export async function fetchWorkingModelsFromScanner(limit = 120): Promise<ScannerModelRow[]> {
  const base = getHfScannerBaseUrl();
  if (!base) {
    const cached = await readWorkingModelsCache();
    return cached ?? [];
  }
  try {
    const rows = await fetchJson<ScannerModelRow[]>(
      `${base}/api/models/working?limit=${Math.min(limit, 500)}`,
      undefined,
      { timeoutMs: 8000 },
    );
    const list = Array.isArray(rows) ? rows : [];
    if (list.length) void writeWorkingModelsCache(list);
    return list;
  } catch {
    const cached = await readWorkingModelsCache();
    return cached ?? [];
  }
}

export function filterScannerModelsByQuery(rows: ScannerModelRow[], query: string): ScannerModelRow[] {
  const terms = query
    .toLowerCase()
    .replace(/hugging\s*face|huggingface|hf\.co|מודל|models?/gi, " ")
    .split(/\s+/)
    .map((t) => t.trim())
    .filter((t) => t.length >= 2);
  if (!terms.length) return rows.slice(0, 12);
  const scored = rows
    .map((row) => {
      const blob = `${row.model_id} ${row.category || ""} ${row.pipeline || ""}`.toLowerCase();
      let hits = 0;
      for (const t of terms) if (blob.includes(t)) hits++;
      return { row, hits };
    })
    .filter((s) => s.hits > 0)
    .sort((a, b) => b.hits - a.hits || (b.row.downloads || 0) - (a.row.downloads || 0));
  return scored.map((s) => s.row);
}

/** Reset health cache (tests). */
export function resetScannerHealthCache(): void {
  scannerHealthyCache = null;
}
