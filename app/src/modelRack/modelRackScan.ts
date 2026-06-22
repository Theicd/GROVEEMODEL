import {
  loadModelRack,
  markCloudHealthChecked,
  mergeWithBuiltinRack,
  saveScannedRackModels,
  summarizeRackCounts,
  type ModelModality,
  type RackCountSummary,
  type RackModelEntry,
} from "./modelRack";
import { scanCoreCloudImageModels } from "./pollinationsScan";

export type ScanProgress = {
  phase: string;
  found: number;
  cloud?: number;
};

export type RackScanSummary = RackCountSummary;

/** Health-check cloud image models only (no HF Hub / Inference scans). */
export async function refreshCloudModelRack(
  onProgress?: (p: ScanProgress & { modality?: ModelModality }) => void,
): Promise<RackModelEntry[]> {
  onProgress?.({ phase: "cloud-health", found: 0, cloud: 0 });

  const cloud = await scanCoreCloudImageModels((model, ok, count) => {
    onProgress?.({
      phase: `cloud:${model}`,
      found: count + (ok ? 1 : 0),
      cloud: count + (ok ? 1 : 0),
    });
  });

  markCloudHealthChecked();
  saveScannedRackModels(cloud);
  onProgress?.({ phase: "cloud-done", found: cloud.length, cloud: cloud.length });
  return mergeWithBuiltinRack(cloud);
}

/** @deprecated use refreshCloudModelRack */
export async function runAutoFreeRackScan(
  onProgress?: (p: ScanProgress & { modality?: ModelModality }) => void,
): Promise<RackModelEntry[]> {
  return refreshCloudModelRack(onProgress);
}

export function rackScanSummary(rack: RackModelEntry[]): RackScanSummary {
  return summarizeRackCounts(rack);
}

// --- Legacy HF scans (kept for search-panel add-to-rack; not used in auto startup) ---

export { addHfHitToRack, scanFreeWorkingHfModels } from "./modelRackScanLegacy";
