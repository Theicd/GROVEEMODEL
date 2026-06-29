import { MAX_CAPTION_WORDS } from "./liveCaptionRoll";

export type TimedWord = { text: string; start: number; end: number };

export class WordRevealScheduler {
  private shown: string[] = [];
  private timers = new Set<number>();
  private readonly minGapMs = 65;

  constructor(
    private readonly onLine: (line: string) => void,
    private readonly maxWords = MAX_CAPTION_WORDS,
  ) {}

  schedule(words: TimedWord[], audioStartMs: number) {
    const now = Date.now();
    let lateIndex = 0;
    words.forEach((w) => {
      const token = w.text.trim();
      if (!token) return;
      const target = audioStartMs + w.start * 1000;
      let delay = Math.max(0, target - now);
      if (delay === 0) {
        delay = this.minGapMs * lateIndex;
        lateIndex += 1;
      }
      const timer = window.setTimeout(() => {
        this.timers.delete(timer);
        this.shown.push(token);
        if (this.shown.length > this.maxWords) this.shown = this.shown.slice(-this.maxWords);
        this.onLine(this.shown.join(" "));
      }, delay);
      this.timers.add(timer);
    });
  }

  reset() {
    this.timers.forEach((t) => window.clearTimeout(t));
    this.timers.clear();
    this.shown = [];
    this.onLine("");
  }
}

export function novelTimedWords(prevChunk: string, words: TimedWord[]): TimedWord[] {
  if (!prevChunk.trim()) return words;
  const prevWords = prevChunk.trim().split(/\s+/).filter(Boolean);
  const newWords = words.map((w) => w.text.trim()).filter(Boolean);
  let overlap = 0;
  const maxK = Math.min(prevWords.length, newWords.length, 8);
  for (let k = maxK; k >= 1; k -= 1) {
    const suffix = prevWords.slice(-k).join(" ").toLowerCase();
    const prefix = newWords.slice(0, k).join(" ").toLowerCase();
    if (suffix === prefix) {
      overlap = k;
      break;
    }
  }
  return words.slice(overlap);
}
