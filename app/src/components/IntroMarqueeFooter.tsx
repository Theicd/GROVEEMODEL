import { formatDownloadPercent } from "../introProgressFormat";
import { useIntroFooterCarousel } from "../hooks/useIntroFooterCarousel";
import { startupChoiceLabelHe, type StartupModelChoice } from "../startupModelProfile";

type IntroMarqueeFooterProps = {
  webgpu: boolean;
  phase: "start" | "loading";
  startupTarget?: StartupModelChoice;
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
  onContinueWithoutChat?: () => void;
};

export function IntroMarqueeFooter({
  webgpu,
  phase,
  startupTarget = "gemma",
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
  onContinueWithoutChat,
}: IntroMarqueeFooterProps) {
  const isStart = phase === "start";
  const modelLabel = startupChoiceLabelHe(startupTarget);
  const {
    current,
    typedText,
    tagSlotReady,
    tagAnim,
    textIn,
    textWarpOut,
    showCursor,
  } = useIntroFooterCarousel(isStart, webgpu);
  const textWarpJump = textWarpOut;
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
          className={`lcars-footer__info-stage${textWarpJump ? " lcars-footer__info-stage--jump" : ""}`}
          aria-live="polite"
        >
          <div className="lcars-footer__warp" aria-hidden="true">
            <span className="lcars-footer__warp-streak" />
            <span className="lcars-footer__warp-streak lcars-footer__warp-streak--alt" />
            <span className="lcars-footer__warp-flash" />
          </div>
          {isStart ? (
            <div
              className={`lcars-footer__chip lcars-footer__chip--solo${current.warn ? " lcars-footer__chip--warn" : ""}`}
            >
              <div className="lcars-footer__tag-slot" aria-hidden={!tagSlotReady}>
                <span
                  key={current.id}
                  className={`lcars-footer__chip-tag lcars-footer__chip-tag--${tagAnim}${tagSlotReady ? " lcars-footer__chip-tag--ready" : ""}`}
                >
                  {current.tag}
                </span>
              </div>
              <div className="lcars-footer__text-slot">
                <span
                  className={`lcars-footer__chip-text${textIn ? " lcars-footer__chip-text--in" : ""}${textWarpOut ? " lcars-footer__chip-text--out" : ""}`}
                >
                  {typedText}
                  {showCursor ? (
                    <span className="lcars-footer__cursor" aria-hidden="true">
                      ▌
                    </span>
                  ) : null}
                </span>
              </div>
            </div>
          ) : (
            <div className="lcars-footer__chip lcars-footer__chip--solo">
              <div className="lcars-footer__tag-slot">
                <span className="lcars-footer__chip-tag lcars-footer__chip-tag--ready lcars-footer__chip-tag--idle">
                  טעינה
                </span>
              </div>
              <div className="lcars-footer__text-slot">
                <span className="lcars-footer__chip-text lcars-footer__chip-text--in">
                  מוריד ומאתחל {modelLabel}…
                </span>
              </div>
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
                {loadingByteLine || (compilePulse ? "INIT ONNX / WebGPU" : `טוען ${modelLabel}…`)}
              </span>
            </div>
          ) : null}

          {(isStart || phase === "loading") && (onContinueWithoutChat || isStart) ? (
            <div className="lcars-footer__links">
              {onContinueWithoutChat ? (
                <button
                  type="button"
                  className="lcars-link"
                  onClick={onContinueWithoutChat}
                  disabled={isGenerating}
                >
                  בלי מודל שיחה
                </button>
              ) : null}
              {isStart ? (
                <>
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
                </>
              ) : null}
            </div>
          ) : null}
        </div>
      </div>
    </footer>
  );
}
