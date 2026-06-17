// @ts-nocheck
import type { StockImageResult } from "./stockImageSearch";

const MAX_CONCURRENT = 2;
const MIN_GAP_MS = 600;

type Job = {
  run: () => Promise<StockImageResult | null>;
  resolve: (hit: StockImageResult | null) => void;
};

const queue: Job[] = [];
let active = 0;
let lastStartedAt = 0;

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

function pump(): void {
  if (active >= MAX_CONCURRENT) return;
  const job = queue.shift();
  if (!job) return;

  active++;
  void (async () => {
    const wait = Math.max(0, MIN_GAP_MS - (Date.now() - lastStartedAt));
    if (wait > 0) await sleep(wait);
    lastStartedAt = Date.now();
    try {
      job.resolve(await job.run());
    } catch {
      job.resolve(null);
    } finally {
      active--;
      pump();
    }
  })();
}

export function enqueueStockImageSearch(
  run: () => Promise<StockImageResult | null>,
): Promise<StockImageResult | null> {
  return new Promise((resolve) => {
    queue.push({ run, resolve });
    pump();
  });
}

export function stockSearchQueueDepth(): number {
  return queue.length + active;
}
