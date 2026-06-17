import { useEffect, useState } from "react";
import { formatBytes, QWEN_ESTIMATED_BYTES, QWEN_MODEL_ID } from "./engine/model/modelInfo";
import { isAiDeepReadEnabled, setAiDeepReadEnabled } from "./engine/settings/aiMode";
import {
  bootSummarizer,
  getModelBootState,
  isSummarizerReady,
  subscribeModelBoot,
  type ModelBootState,
} from "./engine/summarize/summarizerClient";
import { CircularProgress } from "./CircularProgress";

type LampState = "off" | "connecting" | "connected" | "error";

function lampState(enabled: boolean, boot: ModelBootState, ready: boolean): LampState {
  if (!enabled) return "off";
  if (boot.phase === "error") return "error";
  if (ready) return "connected";
  return "connecting";
}

const LAMP_LABEL: Record<LampState, string> = {
  off: "מודל מנותק",
  connecting: "מתחבר למודל…",
  connected: "מודל מחובר",
  error: "שגיאת מודל",
};

export function AiDeepReadPanel({ compact = false }: { compact?: boolean }) {
  const [enabled, setEnabled] = useState(isAiDeepReadEnabled);
  const [boot, setBoot] = useState<ModelBootState>(getModelBootState());

  useEffect(() => subscribeModelBoot(setBoot), []);

  const ready = isSummarizerReady() || boot.phase === "ready";
  const busy = boot.phase === "downloading" || boot.phase === "loading";
  const lamp = lampState(enabled, boot, ready);

  const connect = () => {
    setEnabled(true);
    setAiDeepReadEnabled(true);
    if (!ready && boot.phase !== "downloading" && boot.phase !== "loading") {
      bootSummarizer();
    }
  };

  const disconnect = () => {
    setEnabled(false);
    setAiDeepReadEnabled(false);
  };

  return (
    <section className="gn-ai-ignition" aria-label="שליטה במודל Qwen לסיכום ידיעות">
      <div className="gn-ai-ignition__row">
        <span
          className={`gn-model-lamp gn-model-lamp--${lamp}`}
          role="status"
          aria-label={LAMP_LABEL[lamp]}
          title={LAMP_LABEL[lamp]}
        >
          <span className="gn-model-lamp__bulb" aria-hidden="true" />
        </span>

        <div className="gn-ai-ignition__copy">
          <strong>חקירה עמוקה (Qwen)</strong>
          <span className={`gn-ai-ignition__status gn-ai-ignition__status--${lamp}`}>
            {lamp === "off"
              ? "כבוי"
              : lamp === "connecting"
                ? busy
                  ? `${boot.phase === "downloading" ? "מוריד" : "טוען"}${boot.pct > 0 ? ` · ${boot.pct}%` : ""}`
                  : "מתחיל…"
                : lamp === "connected"
                  ? `מוכן${boot.device ? ` · ${boot.device}` : ""}`
                  : "החיבור נכשל"}
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
          ? "סיכום לפי בקשה מכרטיסיית ידיעה — לא סריקה אוטומטית של המאגר."
          : "ברירת מחדל: כבוי. «סכם כתבה» בפאנל החדשות מפעיל Qwen לכתבה אחת בלבד (~380MB)."}
      </p>

      {!compact && enabled && !ready && lamp !== "error" ? (
        <div className="gn-ai-panel__install">
          <p className="gn-ai-panel__model" dir="ltr">
            {QWEN_MODEL_ID}
          </p>
          {busy || boot.phase === "idle" ? (
            <>
              <CircularProgress
                percent={boot.pct}
                label="QWEN"
                indeterminate={boot.pct < 2 && boot.phase !== "idle"}
              />
              <p className="gn-ai-panel__status">
                {boot.message || `~${formatBytes(QWEN_ESTIMATED_BYTES)}`}
              </p>
            </>
          ) : null}
        </div>
      ) : null}

      {!compact && enabled && lamp === "error" ? (
        <div className="gn-ai-panel__install">
          <p className="gn-ai-panel__status gn-ai-panel__status--error">
            {boot.message || "לא ניתן לטעון את המודל."}
          </p>
          <button type="button" className="gn-ai-panel__retry" onClick={() => bootSummarizer(true)}>
            נסה שוב
          </button>
        </div>
      ) : null}

      {!compact && enabled && ready ? (
        <p className="gn-ai-panel__ready">המודל מוכן — סיכום עמוק רק בלחיצה על «סכם כתבה».</p>
      ) : null}
    </section>
  );
}
