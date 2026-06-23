import { useEffect, useMemo, useRef, useState } from "react";

import {
  INTRO_FOOTER_MESSAGES,
  footerPerfMessage,
  type IntroFooterMessage,
} from "../introFooterMessages";

const HOLD_MS = 5000;
const TAG_ENTER_DELAY_MS = 80;
const TAG_TO_TEXT_MS = 520;
const TEXT_EXIT_MS = 300;
const TAG_SWAP_MS = 360;
const CHAR_MS = 46;
const CHAR_MS_PUNCT = 140;
const BUTTON_ROTATE_MS = 3600;

type CarouselPhase =
  | "tag-enter"
  | "tag-swap"
  | "typing"
  | "holding"
  | "text-exit";

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
  const [phase, setPhase] = useState<CarouselPhase>("tag-enter");
  const [typedText, setTypedText] = useState("");
  const [tagSlotReady, setTagSlotReady] = useState(false);
  const [tagAnim, setTagAnim] = useState<"enter" | "idle" | "swap-out" | "swap-in">("enter");
  const [textIn, setTextIn] = useState(false);
  const [textWarpOut, setTextWarpOut] = useState(false);
  const [altButtonLabel, setAltButtonLabel] = useState(false);
  const firstCycleRef = useRef(true);

  const current = messages[index] ?? messages[0];
  const fullText = current.text;
  const isTyping = phase === "typing";

  useEffect(() => {
    if (!active) {
      setIndex(0);
      setPhase("tag-enter");
      setTypedText("");
      setTagSlotReady(false);
      setTagAnim("enter");
      setTextIn(false);
      setTextWarpOut(false);
      setAltButtonLabel(false);
      firstCycleRef.current = true;
      return;
    }

    setPhase("tag-enter");
    setTypedText("");
    setTextIn(false);
    setTextWarpOut(false);
    setTagAnim(firstCycleRef.current ? "enter" : "swap-in");

    const showTag = window.setTimeout(() => {
      setTagSlotReady(true);
      if (!firstCycleRef.current) setTagAnim("swap-in");
    }, firstCycleRef.current ? TAG_ENTER_DELAY_MS : 0);

    const settleTag = window.setTimeout(() => {
      setTagAnim("idle");
      firstCycleRef.current = false;
    }, (firstCycleRef.current ? TAG_ENTER_DELAY_MS : 0) + TAG_SWAP_MS);

    const startText = window.setTimeout(() => {
      setTextIn(true);
      setPhase("typing");
    }, (firstCycleRef.current ? TAG_ENTER_DELAY_MS : 0) + TAG_SWAP_MS + TAG_TO_TEXT_MS);

    return () => {
      window.clearTimeout(showTag);
      window.clearTimeout(settleTag);
      window.clearTimeout(startText);
    };
  }, [active, index, current.id]);

  useEffect(() => {
    if (!active || phase !== "typing") return;

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const mobile = window.matchMedia("(max-width: 820px)").matches;
    if (reduced || mobile) {
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

    const id = window.setTimeout(() => setPhase("text-exit"), HOLD_MS);
    return () => window.clearTimeout(id);
  }, [active, phase, index]);

  useEffect(() => {
    if (!active || phase !== "text-exit") return;

    setTextWarpOut(true);
    setTextIn(false);

    const id = window.setTimeout(() => {
      setTextWarpOut(false);
      setTypedText("");
      setTagAnim("swap-out");
      setPhase("tag-swap");
    }, TEXT_EXIT_MS);

    return () => window.clearTimeout(id);
  }, [active, phase]);

  useEffect(() => {
    if (!active || phase !== "tag-swap") return;

    const id = window.setTimeout(() => {
      setIndex((i) => (i + 1) % messages.length);
    }, TAG_SWAP_MS);

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
    tagSlotReady,
    tagAnim,
    textIn,
    phase,
    isTyping,
    textWarpOut,
    showCursor: active && textIn && (phase === "typing" || phase === "holding"),
    altButtonLabel,
  };
}
