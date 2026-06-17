import { useEffect, useMemo, useState } from "react";

import {
  INTRO_FOOTER_MESSAGES,
  footerPerfMessage,
  type IntroFooterMessage,
} from "../introFooterMessages";

/** Minimum time each line stays fully visible after typing finishes. */
const HOLD_MS = 5000;
const EXIT_MS = 450;
const CHAR_MS = 46;
const CHAR_MS_PUNCT = 140;
const BUTTON_ROTATE_MS = 3600;

type CarouselPhase = "typing" | "holding" | "exiting";

function charDelay(ch: string | undefined): number {
  if (!ch) return CHAR_MS;
  if (ch === "·" || ch === "—" || ch === "," || ch === ".") return CHAR_MS_PUNCT;
  return CHAR_MS;
}

export function useIntroFooterCarousel(active: boolean, webgpu: boolean) {
  const messages = useMemo(() => {
    const perf = footerPerfMessage(webgpu);
    const list: IntroFooterMessage[] = [...INTRO_FOOTER_MESSAGES];
    const idx = list.findIndex((m) => m.id === "browser");
    if (idx >= 0) list.splice(idx + 1, 0, perf);
    else list.push(perf);
    return list;
  }, [webgpu]);

  const [index, setIndex] = useState(0);
  const [phase, setPhase] = useState<CarouselPhase>("typing");
  const [typedText, setTypedText] = useState("");
  const [chipIn, setChipIn] = useState(false);
  const [altButtonLabel, setAltButtonLabel] = useState(false);

  const current = messages[index] ?? messages[0];
  const fullText = current.text;
  const isTyping = phase === "typing";

  useEffect(() => {
    if (!active) {
      setIndex(0);
      setPhase("typing");
      setTypedText("");
      setChipIn(false);
      setAltButtonLabel(false);
      return;
    }
    setTypedText("");
    setPhase("typing");
    setChipIn(false);
    const enter = window.setTimeout(() => setChipIn(true), 80);
    return () => window.clearTimeout(enter);
  }, [active, index, current.id]);

  useEffect(() => {
    if (!active || phase !== "typing") return;

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduced) {
      setTypedText(fullText);
      setPhase("holding");
      return;
    }

    if (typedText === fullText) {
      setPhase("holding");
      return;
    }

    const next = fullText[typedText.length];
    const id = window.setTimeout(() => {
      setTypedText(fullText.slice(0, typedText.length + 1));
    }, charDelay(next));

    return () => window.clearTimeout(id);
  }, [active, phase, typedText, fullText]);

  useEffect(() => {
    if (!active || phase !== "holding") return;

    const id = window.setTimeout(() => setPhase("exiting"), HOLD_MS);
    return () => window.clearTimeout(id);
  }, [active, phase, index]);

  useEffect(() => {
    if (!active || phase !== "exiting") return;

    setChipIn(false);
    const id = window.setTimeout(() => {
      setIndex((i) => (i + 1) % messages.length);
    }, EXIT_MS);
    return () => window.clearTimeout(id);
  }, [active, phase, messages.length]);

  useEffect(() => {
    if (!active) return;
    const id = window.setInterval(() => setAltButtonLabel((v) => !v), BUTTON_ROTATE_MS);
    return () => window.clearInterval(id);
  }, [active]);

  return {
    current,
    typedText: active ? typedText : "",
    chipIn,
    phase,
    isTyping,
    showCursor: active && (phase === "typing" || phase === "holding"),
    altButtonLabel,
  };
}
