export const MAX_CAPTION_WORDS = 24;
export const LIVE_CHUNK_SECONDS = 1.0;
export const LIVE_CHUNK_OVERLAP_SECONDS = 0.15;
export const SILENCE_CLEAR_MS = 2000;
export const SPEECH_RMS_MIN = 0.012;
export const SPEECH_ZCR_MIN = 0.012;
export const SPEECH_ZCR_MAX = 0.28;

const HALLUCINATION_PHRASES = [
  "thank you",
  "thanks for watching",
  "thanks for listening",
  "subscribe",
  "subtitle",
  "amara.org",
  "you",
  "the",
  "silence",
  "music",
  "applause",
  "blank audio",
];

export type ChunkMetrics = { rms: number; zcr: number };

export function tokenizeCaptionWords(text: string): string[] {
  return text.trim().split(/\s+/).filter(Boolean);
}

export function cleanTranscript(raw: string): string {
  return raw
    .replace(/[\u200B-\u200D\uFEFF]/g, "")
    .replace(/\s+/g, " ")
    .replace(/^[\s.,…!?'"\-–—]+/g, "")
    .replace(/[\s.,…]+$/g, "")
    .trim();
}

export function trimCaptionWords(text: string, maxWords = MAX_CAPTION_WORDS): string {
  const words = tokenizeCaptionWords(text);
  if (words.length <= maxWords) return words.join(" ");
  return words.slice(-maxWords).join(" ");
}

export function chunkAudioMetrics(samples: Float32Array): ChunkMetrics {
  if (!samples.length) return { rms: 0, zcr: 0 };
  let sumSq = 0;
  let crossings = 0;
  for (let i = 0; i < samples.length; i += 1) {
    sumSq += samples[i] * samples[i];
    if (i > 0 && (samples[i] >= 0) !== (samples[i - 1] >= 0)) crossings += 1;
  }
  return {
    rms: Math.sqrt(sumSq / samples.length),
    zcr: crossings / samples.length,
  };
}

/** Speech-like gate — filters silence, loud music beds, and steady noise. */
export function chunkHasSpeech(samples: Float32Array, threshold = SPEECH_RMS_MIN): boolean {
  const { rms, zcr } = chunkAudioMetrics(samples);
  if (rms < threshold) return false;
  if (zcr < SPEECH_ZCR_MIN || zcr > SPEECH_ZCR_MAX) return false;
  return true;
}

export function isHallucinatedTranscript(text: string, metrics?: ChunkMetrics): boolean {
  const t = cleanTranscript(text).toLowerCase();
  if (!t || t.length < 4) return true;
  if (HALLUCINATION_PHRASES.includes(t)) return true;
  if (/^(thank|thanks|subscribe|you|the|um+|uh+|hmm+|so+|and|but|now|we|it|i)\.?$/i.test(t)) return true;

  const words = tokenizeCaptionWords(t);
  if (!words.length) return true;
  if (words.length === 1 && words[0].length <= 3) return true;

  const avgLen = words.reduce((n, w) => n + w.length, 0) / words.length;
  if (words.length >= 3 && avgLen < 2.2) return true;

  // Loud non-speech bed: high energy but tiny nonsense transcript.
  if (metrics && metrics.rms > 0.055 && words.length <= 2 && t.length < 12) return true;

  return false;
}

export function hasStuckRepetition(text: string): boolean {
  const words = tokenizeCaptionWords(text.toLowerCase());
  if (words.length < 5) return false;
  const tail = words.slice(-4);
  const unique = new Set(tail);
  if (unique.size <= 1) return true;
  const last = words[words.length - 1];
  return words.filter((w) => w === last).length >= 3;
}

/** Words in incoming that are not already in the tail of prevChunk. */
export function diffChunkWords(prevChunk: string, incoming: string): string {
  const prevWords = tokenizeCaptionWords(cleanTranscript(prevChunk));
  const newWords = tokenizeCaptionWords(cleanTranscript(incoming));
  if (!newWords.length) return "";
  if (!prevWords.length) return newWords.join(" ");

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
  const novel = newWords.slice(overlap);
  if (!novel.length) return "";
  return novel.join(" ");
}

export function shouldAcceptTranscript(
  roll: string,
  lastChunk: string,
  incoming: string,
  metrics: ChunkMetrics,
): boolean {
  const cleaned = cleanTranscript(incoming);
  if (isHallucinatedTranscript(cleaned, metrics)) return false;

  const novel = diffChunkWords(lastChunk || roll, cleaned);
  if (!novel) return false;

  const novelWords = tokenizeCaptionWords(novel);
  if (!novelWords.length) return false;
  if (novelWords.every((w) => w.length <= 2)) return false;

  const rollWords = tokenizeCaptionWords(roll);
  if (rollWords.length && novelWords.length === 1 && rollWords.slice(-3).includes(novelWords[0])) {
    return false;
  }

  return true;
}

export type AppendResult = {
  roll: string;
  lastChunk: string;
  accepted: boolean;
};

/** Append only fresh words; reset on repetition loops. */
export function appendLiveCaption(
  roll: string,
  lastChunk: string,
  incoming: string,
  metrics: ChunkMetrics,
  maxWords = MAX_CAPTION_WORDS,
): AppendResult {
  const cleaned = cleanTranscript(incoming);
  if (!shouldAcceptTranscript(roll, lastChunk, cleaned, metrics)) {
    return { roll, lastChunk, accepted: false };
  }

  const novel = diffChunkWords(lastChunk || roll, cleaned);
  let nextRoll = roll;
  if (novel) {
    nextRoll = trimCaptionWords(roll ? `${roll} ${novel}` : novel, maxWords);
  } else if (!roll) {
    nextRoll = trimCaptionWords(cleaned, maxWords);
  }

  if (hasStuckRepetition(nextRoll)) {
    nextRoll = trimCaptionWords(cleaned, maxWords);
  }

  return {
    roll: nextRoll,
    lastChunk: cleaned,
    accepted: Boolean(nextRoll),
  };
}

export function formatRollingCaption(roll: string, pending: boolean): string {
  const base = cleanTranscript(roll);
  if (!base) return pending ? "" : "";
  return base;
}
