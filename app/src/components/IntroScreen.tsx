import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { GroveeHudCanvas } from "../GroveeHudCanvas";
import { formatBytes } from "../storageReport";
import { GEMMA_ESTIMATED_BYTES } from "../introProgressFormat";
import {
  SMOLLM_ESTIMATED_BYTES,
  startupChoiceLabelHe,
  type StartupModelChoice,
} from "../startupModelProfile";
import { stageAtLeast } from "../introCinematicLines";
import { useIntroCinematicSequence } from "../hooks/useIntroCinematicSequence";
import { useIntroTypewriter } from "../hooks/useIntroTypewriter";
import { CircularProgress } from "./CircularProgress";
import { IntroEngageTab } from "./IntroEngageTab";
import { IntroMarqueeFooter } from "./IntroMarqueeFooter";
import { IntroTopBar } from "./IntroTopBar";
import { IntroUiGuard } from "./IntroUiGuard";

type IntroScreenProps = {
  phase: "start" | "loading";
  progress: number;
  status: string;
  loadingPhase: "download" | "init";
  loadingByteLine: string;
  loadingFile: string;
  loadingTip: string;
  showWasmRetry: boolean;
  cacheClearing: boolean;
  isLoading: boolean;
  isGenerating: boolean;
  onLoad: () => void;
  onRetryWasm: () => void;
  onOpenInfo: () => void;
  onClearCache: () => void;
  onContinueWithoutChat?: () => void;
  /** Resolved startup target (Gemma vs SmolLM) shown on landing + loading copy. */
  startupTarget?: StartupModelChoice;
  recommendedReasonHe?: string;
};

const STANDBY = "> STANDBY";
const HINT_DISMISS_KEY = "grovee-intro-hints-dismissed";

function readDismissedHints(): Set<string> {
  try {
    const raw = sessionStorage.getItem(HINT_DISMISS_KEY);
    if (!raw) return new Set();
    const parsed = JSON.parse(raw) as string[];
    return new Set(Array.isArray(parsed) ? parsed : []);
  } catch {
    return new Set();
  }
}

export function IntroScreen({
  phase,
  progress,
  status,
  loadingPhase,
  loadingByteLine,
  loadingFile,
  loadingTip,
  showWasmRetry,
  cacheClearing,
  isLoading,
  isGenerating,
  onLoad,
  onRetryWasm,
  onOpenInfo,
  onClearCache,
  onContinueWithoutChat,
  startupTarget = "gemma",
  recommendedReasonHe,
}: IntroScreenProps) {
  const [webgpu, setWebgpu] = useState(false);
  const [dismissedHints, setDismissedHints] = useState<Set<string>>(readDismissedHints);
  const [consoleLines, setConsoleLines] = useState<string[]>([STANDBY]);
  const lastLogRef = useRef("");
  const isStart = phase === "start";
  const { stage } = useIntroCinematicSequence(isStart);
  const typewriterText = useIntroTypewriter(3200, isStart && stageAtLeast(stage, "typewriter"));

  useEffect(() => {
    void navigator.gpu?.requestAdapter().then((adapter) => setWebgpu(!!adapter));
  }, []);

  const displayPct = Math.min(100, Math.max(0, progress));
  const indeterminate = phase === "loading" && loadingPhase === "download" && displayPct < 0.5;
  const compilePulse = loadingPhase === "init";

  const logLine = useMemo(() => {
    if (compilePulse) return "> INIT: ONNX / WebGPU…";
    if (loadingFile) return `> FETCH: ${loadingFile}`;
    if (status && status !== "Not loaded") return `> ${status.toUpperCase()}`;
    return STANDBY;
  }, [compilePulse, loadingFile, status]);

  useEffect(() => {
    if (phase !== "loading" || logLine === lastLogRef.current) return;
    lastLogRef.current = logLine;
    setConsoleLines((prev) => {
      const last = prev[prev.length - 1];
      if (last?.startsWith("> FETCH:") && logLine.startsWith("> FETCH:")) {
        return [...prev.slice(0, -1), logLine].slice(-3);
      }
      return [...prev, logLine].slice(-3);
    });
  }, [phase, logLine]);

  useEffect(() => {
    if (phase === "start") {
      setConsoleLines([STANDBY]);
      lastLogRef.current = "";
    }
  }, [phase]);

  const ringLabel = compilePulse ? "INIT" : startupTarget === "local-text" ? "SMOL" : "GEMMA";
  const modelLabel = startupChoiceLabelHe(startupTarget);
  const estimatedBytes =
    startupTarget === "local-text" ? SMOLLM_ESTIMATED_BYTES : GEMMA_ESTIMATED_BYTES;
  const landingSubtitle = startupTarget === "local-text" ? "SMOLLM2 360M" : "GEMMA 4 E2B";

  const dismissHint = useCallback((id: string) => {
    setDismissedHints((prev) => {
      const next = new Set(prev);
      next.add(id);
      try {
        sessionStorage.setItem(HINT_DISMISS_KEY, JSON.stringify([...next]));
      } catch {
        /* private mode */
      }
      return next;
    });
  }, []);

  const showWebgpuHint = !webgpu && !dismissedHints.has("webgpu-warn");

  return (
    <div
      id="intro-screen"
      className="intro-screen hal-landing hal-landing--cinema"
      data-testid="intro-screen"
      data-ui="hal-space-v2"
      data-phase={phase}
      data-cinematic-stage={stage}
      aria-busy={phase === "loading"}
      aria-live="polite"
      dir="rtl"
    >
      <GroveeHudCanvas />
      <IntroUiGuard />
      <div className="scanlines" aria-hidden="true" />
      <IntroTopBar webgpu={webgpu} />

      <div className="hal-landing__stage">
        {isStart ? (
          <>
            <div className="hal-landing__top-row">
              <div className="hal-float hal-float--brand" aria-label="GROVEE">
                <p
                  className={`hal-float__line hal-landing__eyebrow${stageAtLeast(stage, "eyebrow") ? " hal-float__line--in" : ""}`}
                  dir="ltr"
                >
                  LOCAL AI · BROWSER · FREE
                </p>
                <h1
                  className={`hal-float__line hal-landing__title${stageAtLeast(stage, "title") ? " hal-float__line--in" : ""}`}
                  dir="ltr"
                >
                  GROVEE
                </h1>
                <p
                  className={`hal-float__line hal-landing__subtitle${stageAtLeast(stage, "subtitle") ? " hal-float__line--in" : ""}`}
                  dir="ltr"
                >
                  {landingSubtitle}
                </p>

                <div
                  className={`hal-landing__command-row${stageAtLeast(stage, "typewriter") ? " hal-landing__command-row--in" : ""}`}
                  dir="rtl"
                >
                  <p className="hal-landing__typewriter" aria-live="polite">
                    {typewriterText}
                    <span className="hal-cursor" aria-hidden="true">
                      ▌
                    </span>
                  </p>
                </div>

                {showWebgpuHint ? (
                  <div className="hal-hint-banner hal-hint-banner--warn hal-float__line hal-float__line--in" role="alert">
                    <p className="hal-hint-banner__text" dir="rtl">
                      WebGPU לא זמין — הטעינה תמשיך ב-WASM (CPU), איטי יותר.
                    </p>
                    <button
                      type="button"
                      className="hal-hint-banner__close"
                      onClick={() => dismissHint("webgpu-warn")}
                      aria-label="סגור אזהרה"
                    >
                      ×
                    </button>
                  </div>
                ) : null}
              </div>
            </div>
          </>
        ) : (
          <div
            className="hal-loading-dock"
            data-testid="download-progress"
            data-loading-phase={loadingPhase}
            aria-live="polite"
          >
            <div className="hal-loading-dock__head">
              <CircularProgress
                percent={displayPct}
                size={96}
                indeterminate={indeterminate || compilePulse}
                label={ringLabel}
              />

              <div className="hal-loading-dock__summary">
                <p className="hal-loading-dock__title" dir="rtl">
                  {compilePulse ? "מאתחל מודל בדפדפן…" : `מוריד ${modelLabel}`}
                </p>
                <div
                  className={`hal-loading-dock__bar${indeterminate || compilePulse ? " hal-loading-dock__bar--pulse" : ""}`}
                  role="progressbar"
                  aria-valuenow={Math.round(displayPct)}
                  aria-valuemin={0}
                  aria-valuemax={100}
                >
                  <div
                    className="hal-loading-dock__bar-fill"
                    style={{ width: indeterminate || compilePulse ? undefined : `${displayPct}%` }}
                  />
                </div>
                <p className="hal-loading-dock__meta" dir="ltr" data-testid="download-bytes">
                  {loadingByteLine ||
                    (compilePulse
                      ? `${formatBytes(estimatedBytes)} · INIT ONNX / WebGPU`
                      : status && status !== "Not loaded"
                        ? status
                        : "מכין הורדה…")}
                </p>
              </div>
            </div>

            <div className="hal-loading-dock__console hal-console" dir="ltr">
              {consoleLines.map((line, i) => (
                <div key={`${line}-${i}`} className="hal-console__line">
                  {line}
                </div>
              ))}
            </div>

            <p className="hal-loading-dock__tip" key={loadingTip} dir="rtl">
              {loadingTip}
            </p>
          </div>
        )}
      </div>

      <IntroEngageTab
        active={isStart}
        actionReady={stageAtLeast(stage, "action")}
        isLoading={isLoading}
        isGenerating={isGenerating}
        onLoad={onLoad}
      />

      <IntroMarqueeFooter
        webgpu={webgpu}
        phase={phase}
        isLoading={isLoading}
        isGenerating={isGenerating}
        cacheClearing={cacheClearing}
        showWasmRetry={showWasmRetry}
        displayPct={displayPct}
        loadingByteLine={loadingByteLine}
        compilePulse={compilePulse}
        indeterminate={indeterminate}
        onRetryWasm={onRetryWasm}
        onOpenInfo={onOpenInfo}
        onClearCache={onClearCache}
        onContinueWithoutChat={onContinueWithoutChat}
      />
    </div>
  );
}
