import { useEffect, useState } from "react";
import {
  getEngineStatus,
  subscribeEngineStatus,
} from "./engine/engine/pipeline";
import {
  getModelBootState,
  subscribeModelBoot,
  type ModelBootState,
} from "./engine/summarize/summarizerClient";
import type { EngineStatus } from "./engine/types";

export function useNewsEngineStatus(): { status: EngineStatus; modelBoot: ModelBootState } {
  const [status, setStatus] = useState<EngineStatus>(() => getEngineStatus());
  const [modelBoot, setModelBoot] = useState<ModelBootState>(() => getModelBootState());

  useEffect(() => subscribeEngineStatus(setStatus), []);
  useEffect(() => subscribeModelBoot(setModelBoot), []);

  return { status, modelBoot };
}

export function isNewsEngineBusy(status: EngineStatus): boolean {
  return (
    status.phase === "polling" ||
    status.phase === "extracting" ||
    status.phase === "summarizing" ||
    status.phase === "indexing"
  );
}

export function newsFeedScanPercent(status: EngineStatus): number {
  if (!status.feedsTotal) return 0;
  const done = status.feedsOk + status.feedsFailed;
  return Math.min(100, Math.round((done / status.feedsTotal) * 100));
}
