import { useEffect, useState } from "react";

const VOYAGE_STEP_MS = 3200;

const VOYAGE_LINES = [
  { id: "engage", text: "ENGAGE", dir: "ltr" as const, code: true },
  { id: "start", text: "התחל", dir: "rtl" as const, code: false },
  { id: "load", text: "טען מודל לדפדפן", dir: "rtl" as const, code: false },
];

type IntroEngageTabProps = {
  active: boolean;
  actionReady: boolean;
  isLoading: boolean;
  isGenerating: boolean;
  onLoad: () => void;
};

export function IntroEngageTab({
  active,
  actionReady,
  isLoading,
  isGenerating,
  onLoad,
}: IntroEngageTabProps) {
  const [voyageIdx, setVoyageIdx] = useState(0);

  useEffect(() => {
    if (!active) {
      setVoyageIdx(0);
      return;
    }
    const id = window.setInterval(() => {
      setVoyageIdx((i) => (i + 1) % VOYAGE_LINES.length);
    }, VOYAGE_STEP_MS);
    return () => window.clearInterval(id);
  }, [active]);

  if (!active) return null;

  return (
    <aside
      className={`intro-engage-tab${actionReady ? " intro-engage-tab--ready" : ""}`}
      data-testid="intro-engage-tab"
      aria-label="הפעלת מודל"
    >
      <div className="intro-engage-tab__spine" aria-hidden="true">
        <span className="intro-engage-tab__spine-elbow" />
        <span className="intro-engage-tab__spine-rail" />
        <span className="intro-engage-tab__spine-node" />
      </div>

      <div className="intro-engage-tab__dock">
        <span className="intro-engage-tab__tab-cap" aria-hidden="true" />
        <span className="intro-engage-tab__tab-notch" aria-hidden="true" />
        <div className="intro-engage-tab__btn-wrap">
          <span className="intro-engage-tab__btn-halo" aria-hidden="true" />
          <button
            type="button"
            className={`lcars-btn lcars-btn--engage lcars-btn--tab${actionReady ? " lcars-btn--engage-in" : ""}`}
            data-testid="load-model"
            onClick={onLoad}
            disabled={isLoading || isGenerating || !actionReady}
            aria-label="טען מודל לדפדפן / התחל"
          >
            <span className="lcars-btn__shine" aria-hidden="true" />
            <span className="lcars-btn__voyage" aria-live="polite">
              {VOYAGE_LINES.map((line, i) => (
                <span
                  key={line.id}
                  className={`lcars-btn__voyage-line${line.code ? " lcars-btn__voyage-line--code" : ""}${voyageIdx === i ? " is-visible" : ""}`}
                  dir={line.dir}
                >
                  {line.text}
                </span>
              ))}
            </span>
          </button>
        </div>
      </div>
    </aside>
  );
}
