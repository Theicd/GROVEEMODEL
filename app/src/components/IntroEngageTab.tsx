import { useEffect, useState } from "react";

const BUTTON_ROTATE_MS = 3600;

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
  const [altButtonLabel, setAltButtonLabel] = useState(false);

  useEffect(() => {
    if (!active) {
      setAltButtonLabel(false);
      return;
    }
    const id = window.setInterval(() => setAltButtonLabel((v) => !v), BUTTON_ROTATE_MS);
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
            <span className="lcars-btn__code" dir="ltr">
              ENGAGE
            </span>
            <span className="lcars-btn__label" aria-live="polite">
              <span className={`lcars-btn__label-line${altButtonLabel ? "" : " is-visible"}`}>
                טען מודל לדפדפן
              </span>
              <span
                className={`lcars-btn__label-line lcars-btn__label-line--alt${altButtonLabel ? " is-visible" : ""}`}
              >
                התחל
              </span>
            </span>
          </button>
        </div>
      </div>
    </aside>
  );
}
