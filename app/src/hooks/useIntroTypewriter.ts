import { useEffect, useState } from "react";

const INTRO_PHRASES = [
  "שיחה חינמית — ישירות מהדפדפן",
  "Gemma 4 E2B · טקסט, ראייה וחיפוש",
  "הכל נשאר אצלך — בלי שרת AI",
  "טעינה ראשונה ~3.9GB · אחר כך מהיר מהמטמון",
] as const;

/** Typewriter cycling through intro phrases (Hebrew + English mix). */
export function useIntroTypewriter(intervalMs = 3200, enabled = true): string {
  const phrases = INTRO_PHRASES;
  const [index, setIndex] = useState(0);
  const [text, setText] = useState("");
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    if (!enabled) return;
    setIndex(0);
    setText("");
    setDeleting(false);
  }, [enabled]);

  useEffect(() => {
    if (!enabled) {
      setText("");
      return;
    }

    const phrase = phrases[index % phrases.length] ?? "";
    const doneTyping = text === phrase;
    const doneDeleting = deleting && text === "";

    let delay = deleting ? 28 : 42;
    if (doneTyping && !deleting) delay = intervalMs;
    if (doneDeleting) delay = 400;

    const timer = window.setTimeout(() => {
      if (doneTyping && !deleting) {
        setDeleting(true);
        return;
      }
      if (doneDeleting) {
        setDeleting(false);
        setIndex((i) => (i + 1) % phrases.length);
        return;
      }
      if (deleting) {
        setText(phrase.slice(0, Math.max(0, text.length - 1)));
      } else {
        setText(phrase.slice(0, text.length + 1));
      }
    }, delay);

    return () => window.clearTimeout(timer);
  }, [text, deleting, index, phrases, intervalMs, enabled]);

  return enabled ? text : "";
}
