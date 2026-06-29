import { useCallback, useEffect, useRef, useState } from "react";
import { resampleTo16kMono, startTabAudioTap } from "./liveAudioCapture";
import {
  appendLiveCaption,
  chunkAudioMetrics,
  formatRollingCaption,
  SILENCE_CLEAR_MS,
} from "./liveCaptionRoll";
import { novelTimedWords, WordRevealScheduler } from "./liveCaptionStream";
import { setLiveCaptionsLoadProgress, ensureLiveCaptionsModel, transcribeLiveAudio } from "./liveCaptionsClient";
import { isTabSpeechRecognitionAvailable, startTabSpeechCaptions } from "./liveSpeechRecognition";
import { speechLangToWhisperLanguage, shouldTranslateCaptions, translateLiveCaption } from "./liveTranslate";

function sleep(ms: number): Promise<void> {
  return new Promise((r) => window.setTimeout(r, ms));
}

async function ensureVideoAudible(video: HTMLVideoElement | null): Promise<void> {
  if (!video) return;
  video.muted = false;
  if (video.volume < 0.05) video.volume = 0.75;
  try {
    await video.play();
  } catch {
    /* autoplay policy */
  }
  await sleep(300);
}

export function isLiveCaptionsSupported(): boolean {
  return (
    typeof window !== "undefined" &&
    (isTabSpeechRecognitionAvailable() || typeof Worker !== "undefined") &&
    typeof AudioContext !== "undefined" &&
    Boolean(navigator.mediaDevices?.getDisplayMedia)
  );
}

export type LiveCaptionsStatus = "off" | "starting" | "on" | "error";

type StartOpts = {
  video: HTMLVideoElement | null;
  sourceLang: string;
  targetLang: string;
};

type ChunkJob = {
  samples: Float32Array;
  sampleRate: number;
  audioStartMs: number;
};

const MAX_QUEUED_CHUNKS = 2;

export function useLiveCaptions() {
  const [status, setStatus] = useState<LiveCaptionsStatus>("off");
  const [statusMessage, setStatusMessage] = useState("");
  const [loadPct, setLoadPct] = useState(0);
  const [original, setOriginal] = useState("");
  const [translated, setTranslated] = useState("");
  const activeRef = useRef(false);
  const audioCleanupRef = useRef<(() => void) | null>(null);
  const speechCleanupRef = useRef<(() => void) | null>(null);
  const engineRef = useRef<"speech" | "whisper">("speech");
  const targetLangRef = useRef("he");
  const sourceLangRef = useRef("en-US");
  const whisperLangRef = useRef("english");
  const rollRef = useRef("");
  const lastChunkRef = useRef("");
  const lastSpeechAtRef = useRef(0);
  const chunkQueueRef = useRef<ChunkJob[]>([]);
  const drainingRef = useRef(false);
  const translateTailRef = useRef("");
  const schedulerRef = useRef<WordRevealScheduler | null>(null);
  const whisperStartedRef = useRef(false);

  const syncDisplay = useCallback(() => {
    setOriginal(formatRollingCaption(rollRef.current, false));
  }, []);

  const clearRoll = useCallback(() => {
    rollRef.current = "";
    lastChunkRef.current = "";
    translateTailRef.current = "";
    schedulerRef.current?.reset();
    setOriginal("");
    setTranslated("");
  }, []);

  const cleanupAudio = useCallback(() => {
    audioCleanupRef.current?.();
    audioCleanupRef.current = null;
    speechCleanupRef.current?.();
    speechCleanupRef.current = null;
  }, []);

  const stop = useCallback(() => {
    activeRef.current = false;
    chunkQueueRef.current = [];
    drainingRef.current = false;
    whisperStartedRef.current = false;
    rollRef.current = "";
    lastChunkRef.current = "";
    lastSpeechAtRef.current = 0;
    translateTailRef.current = "";
    schedulerRef.current?.reset();
    schedulerRef.current = null;
    cleanupAudio();
    setLoadPct(0);
    setStatus("off");
    setStatusMessage("");
    setOriginal("");
    setTranslated("");
  }, [cleanupAudio]);

  const touchSpeech = useCallback(
    (line: string) => {
      if (!line.trim()) return;
      rollRef.current = line;
      lastSpeechAtRef.current = Date.now();
      syncDisplay();
    },
    [syncDisplay],
  );

  const maybeTranslateTail = useCallback(async (text: string) => {
    if (!shouldTranslateCaptions(sourceLangRef.current, targetLangRef.current)) {
      setTranslated("");
      return;
    }
    const tail = text.trim();
    if (!tail || tail === translateTailRef.current) return;
    translateTailRef.current = tail;
    try {
      const out = await translateLiveCaption(tail, targetLangRef.current);
      if (activeRef.current && translateTailRef.current === tail) setTranslated(out);
    } catch (err) {
      if (import.meta.env.DEV) console.warn("[live-captions] translate failed", err);
    }
  }, []);

  const applyWhisperResult = useCallback(
    (incoming: string, words: Array<{ text: string; start: number; end: number }>, metrics: ReturnType<typeof chunkAudioMetrics>, audioStartMs: number) => {
      const prevChunk = lastChunkRef.current;
      const result = appendLiveCaption(rollRef.current, prevChunk, incoming, metrics);
      if (!result.accepted) return;
      lastChunkRef.current = result.lastChunk;
      lastSpeechAtRef.current = Date.now();

      const novel = novelTimedWords(prevChunk, words);
      if (!novel.length) return;
      if (!schedulerRef.current) {
        schedulerRef.current = new WordRevealScheduler((line) => {
          rollRef.current = line;
          syncDisplay();
          void maybeTranslateTail(line);
        });
      }
      schedulerRef.current.schedule(novel, audioStartMs);
    },
    [maybeTranslateTail, syncDisplay],
  );

  const drainChunkQueue = useCallback(async () => {
    if (drainingRef.current || !activeRef.current) return;
    drainingRef.current = true;

    while (chunkQueueRef.current.length > 0 && activeRef.current) {
      const job = chunkQueueRef.current.shift();
      if (!job) break;
      try {
        const metrics = chunkAudioMetrics(job.samples);
        const pcm = resampleTo16kMono(job.samples, job.sampleRate);
        const out = await transcribeLiveAudio(pcm, whisperLangRef.current);
        if (out.text.trim() && activeRef.current) {
          applyWhisperResult(out.text, out.words, metrics, job.audioStartMs);
        }
      } catch (err) {
        if (import.meta.env.DEV) console.warn("[live-captions] transcribe chunk failed", err);
      }
    }

    drainingRef.current = false;
  }, [applyWhisperResult]);

  const enqueueChunk = useCallback(
    (samples: Float32Array, sampleRate: number, audioStartMs: number) => {
      if (!activeRef.current) return;
      chunkQueueRef.current.push({ samples, sampleRate, audioStartMs });
      while (chunkQueueRef.current.length > MAX_QUEUED_CHUNKS) chunkQueueRef.current.shift();
      void drainChunkQueue();
    },
    [drainChunkQueue],
  );

  const startWhisperTap = useCallback(async () => {
    if (whisperStartedRef.current || !activeRef.current) return;
    whisperStartedRef.current = true;
    setStatusMessage("whisper-fallback");
    await ensureLiveCaptionsModel();
    const tap = await startTabAudioTap(
      (samples, rate, audioStartMs) => enqueueChunk(samples, rate, audioStartMs),
      () => {
        if (activeRef.current) stop();
      },
    );
    audioCleanupRef.current = tap.cleanup;
  }, [enqueueChunk, stop]);

  const startSpeechEngine = useCallback(
    async (sourceLang: string) => {
      if (!isTabSpeechRecognitionAvailable()) {
        engineRef.current = "whisper";
        await startWhisperTap();
        return;
      }

      engineRef.current = "speech";
      setStatusMessage("speech-live");

      const stopSpeech = await startTabSpeechCaptions({
        lang: sourceLang,
        onText: (line, _hasInterim) => {
          touchSpeech(line);
        },
        onError: (code, fatal) => {
          if (!activeRef.current) return;
          if (import.meta.env.DEV) console.warn("[live-captions] speech error", code, fatal);
          if (fatal && engineRef.current === "speech") {
            engineRef.current = "whisper";
            speechCleanupRef.current?.();
            speechCleanupRef.current = null;
            void startWhisperTap();
          }
        },
      });
      speechCleanupRef.current = stopSpeech;
    },
    [startWhisperTap, touchSpeech],
  );

  const start = useCallback(
    async (opts: StartOpts) => {
      if (!isLiveCaptionsSupported()) {
        setStatus("error");
        setStatusMessage("unsupported");
        return;
      }

      stop();
      await sleep(150);

      activeRef.current = true;
      targetLangRef.current = opts.targetLang;
      whisperLangRef.current = speechLangToWhisperLanguage(opts.sourceLang);
      sourceLangRef.current = opts.sourceLang;
      engineRef.current = "speech";
      rollRef.current = "";
      lastChunkRef.current = "";
      lastSpeechAtRef.current = Date.now();
      translateTailRef.current = "";
      setStatus("starting");
      setStatusMessage("pick-tab");
      setLoadPct(0);

      setLiveCaptionsLoadProgress((pct, message) => {
        if (!activeRef.current) return;
        setLoadPct(pct);
        if (message) setStatusMessage(`loading-model:${message}`);
      });

      try {
        await ensureVideoAudible(opts.video);
        await startSpeechEngine(opts.sourceLang);
        setStatus("on");
        setLoadPct(100);
      } catch (err) {
        activeRef.current = false;
        cleanupAudio();
        if (import.meta.env.DEV) console.warn("[live-captions] start failed", err);
        const msg = err instanceof Error ? err.message : "failed";
        setStatus("error");
        if (err instanceof DOMException && err.name === "NotAllowedError") {
          setStatusMessage("share-denied");
        } else if (msg === "no-audio") {
          setStatusMessage("no-audio");
        } else if (msg.includes("model-load")) {
          setStatusMessage("model-load-failed");
        } else {
          setStatusMessage(msg);
        }
      } finally {
        setLiveCaptionsLoadProgress(null);
      }
    },
    [cleanupAudio, startSpeechEngine, stop],
  );

  useEffect(() => {
    if (status !== "on") return undefined;
    const id = window.setInterval(() => {
      if (!activeRef.current || !rollRef.current) return;
      if (Date.now() - lastSpeechAtRef.current > SILENCE_CLEAR_MS) clearRoll();
    }, 300);
    return () => window.clearInterval(id);
  }, [clearRoll, status]);

  useEffect(() => () => {
    setLiveCaptionsLoadProgress(null);
    stop();
  }, [stop]);

  return {
    status,
    statusMessage,
    loadPct,
    original,
    translated,
    active: status === "on" || status === "starting",
    start,
    stop,
  };
}
