import { formatDownloadPercent } from "../introProgressFormat";
import { useIntroFooterCarousel } from "../hooks/useIntroFooterCarousel";

type IntroMarqueeFooterProps = {
  webgpu: boolean;
  phase: "start" | "loading";
  isLoading: boolean;
  isGenerating: boolean;
  cacheClearing: boolean;
  showWasmRetry: boolean;
  displayPct: number;
  loadingByteLine: string;
  compilePulse: boolean;
  indeterminate: boolean;
  onRetryWasm: () => void;
  onOpenInfo: () => void;
  onClearCache: () => void;
};

export function IntroMarqueeFooter({
  webgpu,
  phase,
  isLoading,
  isGenerating,
  cacheClearing,
  showWasmRetry,
  displayPct,
  loadingByteLine,
  compilePulse,
  indeterminate,
  onRetryWasm,
  onOpenInfo,
  onClearCache,
}: IntroMarqueeFooterProps) {
  const isStart = phase === "start";
  const { current, typedText, chipIn, phase: carouselPhase, showCursor } =
    useIntroFooterCarousel(isStart, webgpu);
  const chipWarpOut = carouselPhase === "exiting";
  const chipWarpJump = !chipIn || chipWarpOut;
  const pctLabel = indeterminate || compilePulse ? "…" : `${formatDownloadPercent(displayPct)}%`;

  return (
    <footer
      className="lcars-footer"
      data-testid="intro-vital-footer"
      dir="rtl"
      role="region"
      aria-label="מידע והפעלה — GROVEE"
      data-phase={phase}
    >
      <div className="lcars-footer__caps" aria-hidden="true">
        <span className="lcars-cap lcars-cap--sun" />
        <span className="lcars-cap lcars-cap--peach" />
        <span className="lcars-cap lcars-cap--tan" />
        <span className="lcars-cap lcars-cap--lilac" />
        <span className="lcars-cap lcars-cap--blue" />
      </div>

      <div className="lcars-footer__main">
        <div
          className={`lcars-footer__info-stage${chipWarpJump ? " lcars-footer__info-stage--jump" : ""}`}
          aria-live="polite"
        >
          <div className="lcars-footer__warp" aria-hidden="true">
            <span className="lcars-footer__warp-streak" />
            <span className="lcars-footer__warp-streak lcars-footer__warp-streak--alt" />
            <span className="lcars-footer__warp-flash" />
          </div>
          {isStart ? (
            <div
              key={current.id}
              className={`lcars-footer__chip lcars-footer__chip--solo${chipIn ? " lcars-footer__chip--in" : ""}${chipWarpOut ? " lcars-footer__chip--warp-out" : ""}${current.warn ? " lcars-footer__chip--warn" : ""}`}
            >
              <span className="lcars-footer__chip-tag">{current.tag}</span>
              <span className="lcars-footer__chip-text">
                {typedText}
                {showCursor ? (
                  <span className="lcars-footer__cursor" aria-hidden="true">
                    ▌
                  </span>
                ) : null}
              </span>
            </div>
          ) : (
            <div className="lcars-footer__chip lcars-footer__chip--solo lcars-footer__chip--in">
              <span className="lcars-footer__chip-tag">טעינה</span>
              <span className="lcars-footer__chip-text">מוריד ומאתחל Gemma 4 E2B…</span>
            </div>
          )}
        </div>

        <div className="lcars-footer__command">
          {!isStart ? (
            <div className="lcars-footer__progress" data-testid="footer-load-progress">
              <span className="lcars-footer__progress-pct" dir="ltr">
                {pctLabel}
              </span>
              <div
                className={`lcars-footer__progress-bar${indeterminate || compilePulse ? " lcars-footer__progress-bar--pulse" : ""}`}
                role="progressbar"
                aria-valuenow={Math.round(displayPct)}
                aria-valuemin={0}
                aria-valuemax={100}
              >
                <div
                  className="lcars-footer__progress-fill"
                  style={{ width: indeterminate || compilePulse ? undefined : `${displayPct}%` }}
                />
              </div>
              <span className="lcars-footer__progress-meta" dir="ltr">
                {loadingByteLine || (compilePulse ? "INIT ONNX / WebGPU" : "טוען Gemma 4 E2B…")}
              </span>
            </div>
          ) : null}

          {isStart ? (
            <div className="lcars-footer__links">
              {showWasmRetry ? (
                <button
                  type="button"
                  className="lcars-link"
                  onClick={onRetryWasm}
                  disabled={isLoading || isGenerating}
                >
                  WASM
                </button>
              ) : null}
              <button type="button" className="lcars-link" onClick={onOpenInfo}>
                מידע
              </button>
              <button
                type="button"
                className="lcars-link lcars-link--muted"
                onClick={() => void onClearCache()}
                disabled={isGenerating || isLoading || cacheClearing}
              >
                {cacheClearing ? "מנקה…" : "מטמון"}
              </button>
            </div>
          ) : null}
        </div>
      </div>
    </footer>
  );
}
