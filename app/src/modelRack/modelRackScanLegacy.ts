import { STATUS_WORKING } from "../webSearch/hf/hfModelProbe";
import {
  fetchWorkingModelsFromScanner,
  type ScannerModelRow,
} from "../webSearch/hf/hfApiScannerClient";
import type { HfHubModelSummary } from "../webSearch/hf/hfModelTypes";
import type { HfModelSerpHit } from "../webSearch/hf/hfModelTypes";
import { proxyAwareFetch } from "../webSearch/proxyFetch";
import { mapWithConcurrency, probeHfModelForRack, type HfProbeKind } from "./hfRackProbe";
import {
  loadModelRack,
  modalityFromPipeline,
  rackEntryFromHfHit,
  upsertFreeRackModel,
  type ModelModality,
  type RackModelEntry,
} from "./modelRack";

export const MAX_TOTAL_PROBES = 400;

const HF_PIPELINE_SCAN: {
  pipelineTag: string;
  probe: HfProbeKind;
  hubLimit: number;
  maxProbe: number;
}[] = [
  { pipelineTag: "text-generation", probe: "chat", hubLimit: 60, maxProbe: 12 },
  { pipelineTag: "text-to-image", probe: "image", hubLimit: 40, maxProbe: 10 },
];

const PROBE_CONCURRENCY = 3;
const PROBE_DELAY_MS = 350;

function hubModelId(m: HfHubModelSummary): string {
  return (m.id ?? m.modelId ?? "").trim();
}

function isScannerRowFreeWorking(row: ScannerModelRow): boolean {
  const status = (row.status || "").toUpperCase();
  const access = (row.access_mode || "").toUpperCase();
  return status === STATUS_WORKING && access === "FREE";
}

function entryFromScannerRow(row: ScannerModelRow): RackModelEntry | null {
  if (!isScannerRowFreeWorking(row)) return null;
  return rackEntryFromHfHit({
    modelId: row.model_id,
    category: row.category,
    pipelineTag: row.pipeline,
    status: STATUS_WORKING,
    accessMode: "FREE",
  });
}

async function fetchHubByPipelineTag(pipelineTag: string, limit: number): Promise<HfHubModelSummary[]> {
  const url = `https://huggingface.co/api/models?pipeline_tag=${encodeURIComponent(pipelineTag)}&limit=${limit}&sort=downloads&direction=-1`;
  try {
    const response = await proxyAwareFetch(url, {
      headers: { Accept: "application/json", "User-Agent": "GROVEEMODEL/1.0" },
    });
    if (!response.ok) return [];
    const data = (await response.json()) as HfHubModelSummary[];
    return Array.isArray(data) ? data : [];
  } catch {
    return [];
  }
}

type ProbeJob = { modelId: string; pipelineTag: string; probe: HfProbeKind };

async function runProbeJobs(
  jobs: ProbeJob[],
  seen: Set<string>,
  merged: Map<string, RackModelEntry>,
  budget: { left: number },
): Promise<void> {
  const pending = jobs.filter((j) => j.modelId && !seen.has(j.modelId)).slice(0, budget.left);
  if (!pending.length) return;
  budget.left -= pending.length;

  const entries = await mapWithConcurrency(pending, PROBE_CONCURRENCY, async (job) => {
    const probe = await probeHfModelForRack(job.modelId, job.probe);
    if (!probe.ok || probe.accessMode !== "FREE") return null;
    const entry = rackEntryFromHfHit({
      modelId: job.modelId,
      pipelineTag: job.pipelineTag,
      status: STATUS_WORKING,
      accessMode: "FREE",
    });
    return entry.status === "ready" ? entry : null;
  });

  for (const entry of entries) {
    if (!entry) continue;
    seen.add(entry.hfModelId!);
    merged.set(entry.id, entry);
  }

  if (PROBE_DELAY_MS > 0 && budget.left > 0) {
    await new Promise((r) => setTimeout(r, PROBE_DELAY_MS));
  }
}

/** Optional deep HF Inference scan (not used on app startup). */
export async function scanFreeWorkingHfModels(
  onProgress?: (p: { phase: string; found: number; modality?: ModelModality }) => void,
): Promise<RackModelEntry[]> {
  const merged = new Map<string, RackModelEntry>();
  const seen = new Set<string>();
  const budget = { left: MAX_TOTAL_PROBES };

  for (const row of await fetchWorkingModelsFromScanner(200)) {
    const entry = entryFromScannerRow(row);
    if (!entry?.hfModelId) continue;
    seen.add(entry.hfModelId);
    merged.set(entry.id, entry);
  }

  for (const cfg of HF_PIPELINE_SCAN) {
    if (budget.left <= 0) break;
    onProgress?.({ phase: cfg.pipelineTag, found: merged.size, modality: modalityFromPipeline(cfg.pipelineTag, "") });
    const hub = await fetchHubByPipelineTag(cfg.pipelineTag, cfg.hubLimit);
    const jobs = hub
      .map(hubModelId)
      .filter((id) => id && !seen.has(id))
      .slice(0, cfg.maxProbe)
      .map((modelId) => ({ modelId, pipelineTag: cfg.pipelineTag, probe: cfg.probe }));
    await runProbeJobs(jobs, seen, merged, budget);
  }

  return [...merged.values()];
}

export function addHfHitToRack(hit: HfModelSerpHit): RackModelEntry[] {
  if (hit.status !== STATUS_WORKING || hit.accessMode !== "FREE") {
    return loadModelRack();
  }
  const entry = rackEntryFromHfHit({
    modelId: hit.modelId,
    category: hit.category,
    pipelineTag: hit.pipelineTag,
    status: hit.status,
    accessMode: hit.accessMode,
  });
  if (entry.status !== "ready") return loadModelRack();
  return upsertFreeRackModel(entry);
}
