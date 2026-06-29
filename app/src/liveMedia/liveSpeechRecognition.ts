import { MAX_CAPTION_WORDS, cleanTranscript, trimCaptionWords } from "./liveCaptionRoll";

type SpeechRecognitionResultLike = {
  isFinal: boolean;
  [index: number]: { transcript: string };
};

type SpeechRecognitionEventLike = {
  resultIndex: number;
  results: {
    length: number;
    [index: number]: SpeechRecognitionResultLike;
  };
};

type BrowserSpeechRecognition = {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: { error?: string }) => void) | null;
  onend: (() => void) | null;
  start: (audioTrack?: MediaStreamTrack) => void;
  stop: () => void;
};

type SpeechRecognitionCtor = new () => BrowserSpeechRecognition;

function getSpeechRecognition(): SpeechRecognitionCtor | null {
  const w = window as Window & {
    SpeechRecognition?: SpeechRecognitionCtor;
    webkitSpeechRecognition?: SpeechRecognitionCtor;
  };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

export function isTabSpeechRecognitionAvailable(): boolean {
  return getSpeechRecognition() != null;
}

const FATAL_SPEECH_ERRORS = new Set(["network", "not-allowed", "service-not-allowed", "audio-capture"]);

export type TabSpeechOpts = {
  lang: string;
  onText: (line: string, hasInterim: boolean) => void;
  onError: (code: string, fatal: boolean) => void;
};

/** Fast interim captions from Chrome speech service (word-by-word). */
export async function startTabSpeechCaptions(opts: TabSpeechOpts): Promise<() => void> {
  const Ctor = getSpeechRecognition();
  if (!Ctor) throw new Error("no-speech-api");

  const displayStream = await navigator.mediaDevices.getDisplayMedia({
    video: true,
    audio: true,
    preferCurrentTab: true,
    selfBrowserSurface: "include",
  } as MediaStreamConstraints);

  for (const vt of displayStream.getVideoTracks()) vt.enabled = false;
  const audioTrack = displayStream.getAudioTracks()[0];
  if (!audioTrack) {
    displayStream.getTracks().forEach((t) => t.stop());
    throw new Error("no-audio");
  }

  let committed = "";
  let stopped = false;
  let restarting = false;

  const recognition = new Ctor();
  recognition.lang = opts.lang;
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.maxAlternatives = 1;

  const pushLine = (interim: string, hasInterim: boolean) => {
    const line = trimCaptionWords(cleanTranscript(`${committed}${interim}`), MAX_CAPTION_WORDS);
    opts.onText(line, hasInterim);
  };

  recognition.onresult = (event) => {
    if (stopped) return;
    let interim = "";
    for (let i = event.resultIndex; i < event.results.length; i += 1) {
      const piece = event.results[i]?.[0]?.transcript ?? "";
      if (event.results[i]?.isFinal) committed += piece;
      else interim += piece;
    }
    pushLine(interim, Boolean(interim));
  };

  recognition.onerror = (e) => {
    const code = e.error ?? "recognition-error";
    if (code === "aborted" || code === "no-speech") return;
    opts.onError(code, FATAL_SPEECH_ERRORS.has(code));
  };

  recognition.onend = () => {
    if (stopped || restarting) return;
    restarting = true;
    window.setTimeout(() => {
      restarting = false;
      if (stopped || audioTrack.readyState !== "live") return;
      try {
        recognition.start(audioTrack);
      } catch {
        try {
          recognition.start();
        } catch {
          opts.onError("recognition-stopped", true);
        }
      }
    }, 120);
  };

  try {
    recognition.start(audioTrack);
  } catch {
    recognition.start();
  }

  return () => {
    stopped = true;
    try {
      recognition.stop();
    } catch {
      /* ignore */
    }
    displayStream.getTracks().forEach((t) => t.stop());
  };
}
