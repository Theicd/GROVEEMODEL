import { useCallback, useEffect, useRef, useState } from "react";

type SpeechRecognitionCtor = new () => SpeechRecognition;

function getSpeechRecognition(): SpeechRecognitionCtor | null {
  const w = window as Window & {
    SpeechRecognition?: SpeechRecognitionCtor;
    webkitSpeechRecognition?: SpeechRecognitionCtor;
  };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

type Props = {
  disabled?: boolean;
  onTranscript: (text: string) => void;
};

export function ComposerVoiceMic({ disabled, onTranscript }: Props) {
  const [listening, setListening] = useState(false);
  const [supported, setSupported] = useState(false);
  const recognitionRef = useRef<SpeechRecognition | null>(null);

  useEffect(() => {
    setSupported(getSpeechRecognition() !== null);
  }, []);

  const stopListening = useCallback(() => {
    recognitionRef.current?.stop();
    recognitionRef.current = null;
    setListening(false);
  }, []);

  const startListening = useCallback(() => {
    const Ctor = getSpeechRecognition();
    if (!Ctor || disabled) return;

    const recognition = new Ctor();
    recognition.lang = document.documentElement.lang === "he" ? "he-IL" : "he-IL";
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onresult = (event) => {
      const last = event.results[event.results.length - 1];
      if (last?.isFinal) {
        const text = last[0]?.transcript?.trim();
        if (text) onTranscript(text);
      }
    };

    recognition.onerror = () => {
      stopListening();
    };

    recognition.onend = () => {
      recognitionRef.current = null;
      setListening(false);
    };

    recognitionRef.current = recognition;
    recognition.start();
    setListening(true);
  }, [disabled, onTranscript, stopListening]);

  useEffect(() => () => stopListening(), [stopListening]);

  const toggle = () => {
    if (listening) stopListening();
    else startListening();
  };

  if (!supported) return null;

  return (
    <button
      type="button"
      className={`in-act in-mic ${listening ? "in-mic--listening" : ""}`}
      onClick={toggle}
      disabled={disabled}
      aria-label={listening ? "עצור הכתבה קולית" : "הכתבה קולית"}
      aria-pressed={listening}
      title={listening ? "עצור הכתבה" : "הכתבה קולית"}
    >
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
        <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
        <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
        <line x1="12" y1="19" x2="12" y2="22" />
      </svg>
    </button>
  );
}
