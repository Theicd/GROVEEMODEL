// @ts-nocheck
import { fetchArticleImage } from "../extract/articleImage";

const MIN_GAP_MS = 2_800;
const MAX_CONCURRENT = 1;

type Job = {
  url: string;
  priority: number;
  resolve: (image: string) => void;
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
      job.resolve(await fetchArticleImage(job.url));
    } catch {
      job.resolve("");
    } finally {
      active--;
      pump();
    }
  })();
}

/** Throttled article-page image fetch — one request at a time with a minimum gap. */
export function enqueueArticleImageFetch(url: string, priority = 0): Promise<string> {
  return new Promise((resolve) => {
    queue.push({ url, priority, resolve });
    queue.sort((a, b) => b.priority - a.priority);
    pump();
  });
}

export function imageQueueDepth(): number {
  return queue.length + active;
}
