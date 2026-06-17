import { useEffect, useState } from "react";
import { formatBytes } from "./engine/model/modelInfo";
import { isAiDeepReadEnabled, setAiDeepReadEnabled } from "./engine/settings/aiMode";
import { GEMMA_ARTICLE_MODEL_ID, GEMMA_ESTIMATED_BYTES } from "./gemmaModelInfo";
import { CircularProgress } from "./CircularProgress";

export type GemmaDeepReadPanelProps = {
  compact?: boolean;
  gemmaReady: boolean;
  gemmaLoading: boolean;
  gemmaLoadPct?: number;
  gemmaLoadDetail?: string;
  onRequestGemmaLoad?: () => void;
};

type LampState = "off" | "connecting" | "connected" | "waiting";

function lampState(enabled: boolean, gemmaReady: boolean, gemmaLoading: boolean): LampState {
  if (!enabled) return "off";
  if (gemmaReady) return "connected";
  if (gemmaLoading) return "connecting";
  return "waiting";
}

const LAMP_LABEL: Record<LampState, string> = {
  off: "סיכום כתבות כבוי",
  connecting: "טוען Gemma לדפדפן…",
  connected: "Gemma מוכן לסיכום",
  waiting: "נדרש טעינת מודל",
};

export function AiDeepReadPanel({
  compact = false,
  gemmaReady,
  gemmaLoading,
  gemmaLoadPct = 0,
  gemmaLoadDetail,
  onRequestGemmaLoad,
}: GemmaDeepReadPanelProps) {
  const [enabled, setEnabled] = useState(isAiDeepReadEnabled);

  useEffect(() => {
    const sync = () => setEnabled(isAiDeepReadEnabled());
    const onStorage = (e: StorageEvent) => {
      if (e.key === "gn-ai-deep-read") sync();
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  const lamp = lampState(enabled, gemmaReady, gemmaLoading);
  const busy = gemmaLoading;

  const connect = () => {
    setEnabled(true);
    setAiDeepReadEnabled(true);
    if (!gemmaReady && !gemmaLoading) onRequestGemmaLoad?.();
  };

  const disconnect = () => {
    setEnabled(false);
    setAiDeepReadEnabled(false);
  };

  return (
    <section className="gn-ai-ignition" aria-label="סיכום כתבות עם Gemma">
      <div className="gn-ai-ignition__row">
        <span
          className={`gn-model-lamp gn-model-lamp--${lamp === "waiting" ? "connecting" : lamp}`}
          role="status"
          aria-label={LAMP_LABEL[lamp]}
          title={LAMP_LABEL[lamp]}
        >
          <span className="gn-model-lamp__bulb" aria-hidden="true" />
        </span>

        <div className="gn-ai-ignition__copy">
          <strong>חקירה עמוקה (Gemma 4)</strong>
          <span className={`gn-ai-ignition__status gn-ai-ignition__status--${lamp === "waiting" ? "connecting" : lamp}`}>
            {lamp === "off"
              ? "כבוי"
              : lamp === "connecting"
                ? busy
                  ? `טוען${gemmaLoadPct > 0 ? ` · ${Math.round(gemmaLoadPct)}%` : ""}`
                  : "מתחיל…"
                : lamp === "connected"
                  ? "מוכן לסיכום בעברית"
                  : "לחץ «טען מודל לדפדפן»"}
          </span>
        </div>

        {!enabled ? (
          <button type="button" className="gn-ai-ignition__btn gn-ai-ignition__btn--on" onClick={connect}>
            הפעל
          </button>
        ) : (
          <button type="button" className="gn-ai-ignition__btn gn-ai-ignition__btn--off" onClick={disconnect}>
            כבה
          </button>
        )}
      </div>

      <p className="gn-ai-ignition__hint">
        {compact
          ? "«סכם כתבה» שולף טקסט מהמקור ומנסח בעברית דרך Gemma — לא סריקה אוטומטית של כל המאגר."
          : "ברירת מחדל: כבוי. «סכם כתבה» בפאנל החדשות שולף את הכתבה ומנסח תקציר בעברית עם Gemma 4 (אותו מודל כמו הצ'אט)."}
      </p>

      {!compact && enabled && !gemmaReady && lamp !== "off" ? (
        <div className="gn-ai-panel__install">
          <p className="gn-ai-panel__model" dir="ltr">
            {GEMMA_ARTICLE_MODEL_ID}
          </p>
          {busy ? (
            <>
              <CircularProgress
                percent={gemmaLoadPct}
                label="GEMMA"
                indeterminate={gemmaLoadPct < 2}
              />
              <p className="gn-ai-panel__status">
                {gemmaLoadDetail || `~${formatBytes(GEMMA_ESTIMATED_BYTES)}`}
              </p>
            </>
          ) : (
            <>
              <p className="gn-ai-panel__status">
                המודל לא נטען — לחץ «טען מודל לדפדפן» במסך הראשי או «הפעל» כאן.
              </p>
              {onRequestGemmaLoad ? (
                <button type="button" className="gn-ai-panel__retry" onClick={onRequestGemmaLoad}>
                  טען Gemma
                </button>
              ) : null}
            </>
          )}
        </div>
      ) : null}

      {!compact && enabled && gemmaReady ? (
        <p className="gn-ai-panel__ready">
          Gemma מוכן — לחץ «סכם כתבה» בכרטיס ידיעה לתקציר בעברית (כותרת + תקציר).
        </p>
      ) : null}
    </section>
  );
}
