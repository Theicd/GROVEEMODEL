import { useEffect, useState, type RefObject } from "react";
import { createPortal } from "react-dom";
import type { ChatUiLanguage } from "../ui/useUiLanguage";
import { CAPTION_SOURCE_LANGS, CAPTION_TARGET_LANGS } from "./liveTranslate";
import type { LiveCaptionsStatus } from "./useLiveCaptions";

type MenuPos = { left: number; bottom: number };

type Props = {
  uiLang: ChatUiLanguage;
  anchorRef: RefObject<HTMLElement | null>;
  open: boolean;
  captionsActive: boolean;
  captionsStatus: LiveCaptionsStatus;
  statusMessage: string;
  loadPct?: number;
  sourceLang: string;
  targetLang: string;
  captionsSupported: boolean;
  onSourceLang: (code: string) => void;
  onTargetLang: (code: string) => void;
  onToggleCaptions: () => void;
  onClose: () => void;
};

function computeMenuPos(anchor: HTMLElement): MenuPos {
  const r = anchor.getBoundingClientRect();
  const menuW = Math.min(280, window.innerWidth * 0.88);
  const half = menuW / 2;
  const pad = 8;
  const centerX = r.left + r.width / 2;
  const left = Math.max(pad + half, Math.min(window.innerWidth - pad - half, centerX));
  return {
    left,
    bottom: window.innerHeight - r.top + 10,
  };
}

export function CableTunerGearMenu({
  uiLang,
  anchorRef,
  open,
  captionsActive,
  captionsStatus,
  statusMessage,
  loadPct = 0,
  sourceLang,
  targetLang,
  captionsSupported,
  onSourceLang,
  onTargetLang,
  onToggleCaptions,
  onClose,
}: Props) {
  const [pos, setPos] = useState<MenuPos | null>(null);

  useEffect(() => {
    if (!open) {
      setPos(null);
      return;
    }
    const update = () => {
      const el = anchorRef.current;
      if (!el) return;
      setPos(computeMenuPos(el));
    };
    update();
    window.addEventListener("resize", update);
    window.addEventListener("scroll", update, true);
    return () => {
      window.removeEventListener("resize", update);
      window.removeEventListener("scroll", update, true);
    };
  }, [anchorRef, open]);

  if (!open || !pos) return null;

  const he = uiLang === "he";
  const L = he
    ? {
        title: "הגדרות",
        captions: "כתוביות חיות ותרגום",
        captionsOn: "עצור כתוביות",
        captionsOff: "הפעל כתוביות",
        srcLang: "שפת מקור",
        targetLang: "תרגם ל",
        unsupported: "לא נתמך — נדרש Chrome עם Web Audio ושיתוף טאב",
        pickTab: "בחר את הטאב הזה וסמן ״שתף אודיו״",
        starting: "מאתחל…",
        tabAudio: "מאזין לאודיו מהטאב",
        localListening: "מאזין — תמלול מקומי (Whisper), ללא שירות Google",
        speechLive: "מאזין — כתוביות חיות מילה-מילה",
        whisperFallback: "עבר לתמלול מקומי (Whisper)",
        loadingModel: "טוען מודל תמלול מקומי…",
        stream: "מאזין לשידור",
        shareDenied: "שיתוף המסך בוטל — נדרש אודיו מהטאב",
        micFallback: "מצב מיקרופון — ודא שהרמקול פועל ליד המיקרופון",
        trackRecFailed: "הדפדפן לא תומך בתמלול מאודיו הטאב — נסה Chrome",
        modelLoadFailed: "טעינת מודל התמלול נכשלה — בדוק חיבור לאינטרנט (הורדה חד-פעמית)",
        noAudio: "לא נבחר אודיו — סמן ״שתף אודיו״",
        network: "שגיאת תמלול ענן — עברנו לתמלול מקומי; רענן ונסה שוב",
        recognitionStopped: "התמלול נעצר — נסה שוב",
        error: "שגיאה בהפעלה",
      }
    : {
        title: "Settings",
        captions: "Live captions & translate",
        captionsOn: "Stop captions",
        captionsOff: "Start captions",
        srcLang: "Source language",
        targetLang: "Translate to",
        unsupported: "Not supported — Chrome with Web Audio and tab share required",
        pickTab: "Select this tab and check Share tab audio",
        starting: "Starting…",
        tabAudio: "Listening to tab audio",
        localListening: "Listening — local Whisper STT (no Google speech)",
        speechLive: "Listening — live word-by-word captions",
        whisperFallback: "Switched to local Whisper STT",
        loadingModel: "Loading local speech model…",
        stream: "Listening to stream",
        shareDenied: "Screen share cancelled — tab audio required",
        micFallback: "Microphone mode — play audio near the mic",
        trackRecFailed: "Browser cannot transcribe tab audio — try Chrome",
        modelLoadFailed: "Speech model failed to load — need internet once for download",
        noAudio: "No audio — enable Share tab audio",
        network: "Cloud STT error — using local model; refresh and retry",
        recognitionStopped: "Transcription stopped — try again",
        error: "Failed to start",
      };

  let hint = "";
  if (captionsStatus === "starting" && statusMessage === "pick-tab") hint = L.pickTab;
  else if (captionsStatus === "starting" && statusMessage.startsWith("loading-model")) {
    hint = loadPct > 0 ? `${L.loadingModel} ${loadPct}%` : L.loadingModel;
  } else if (captionsStatus === "starting") hint = L.starting;
  else if (statusMessage === "tab-audio") hint = L.tabAudio;
  else if (statusMessage === "local-listening") hint = L.localListening;
  else if (statusMessage === "speech-live") hint = L.speechLive;
  else if (statusMessage === "whisper-fallback") hint = L.whisperFallback;
  else if (statusMessage === "stream") hint = L.stream;
  else if (statusMessage === "share-denied") hint = L.shareDenied;
  else if (statusMessage === "mic-fallback") hint = L.micFallback;
  else if (statusMessage === "track-rec-failed") hint = L.trackRecFailed;
  else if (statusMessage === "model-load-failed") hint = L.modelLoadFailed;
  else if (statusMessage === "no-audio") hint = L.noAudio;
  else if (statusMessage === "network") hint = L.network;
  else if (statusMessage === "recognition-stopped") hint = L.recognitionStopped;
  else if (statusMessage === "unsupported") hint = L.unsupported;
  else if (captionsStatus === "error") hint = L.error;

  const menu = (
    <div
      className="lm-cable-gear-menu lm-cable-gear-menu--portal"
      dir={he ? "rtl" : "ltr"}
      role="dialog"
      aria-label={L.title}
      style={{
        position: "fixed",
        left: pos.left,
        bottom: pos.bottom,
        transform: "translateX(-50%)",
        zIndex: 20000,
      }}
    >
      <div className="lm-cable-gear-menu__head">
        <span className="lm-cable-gear-menu__title">{L.title}</span>
        <button type="button" className="lm-cable-gear-menu__close" onClick={onClose} aria-label={he ? "סגור" : "Close"}>
          ×
        </button>
      </div>

      <section className="lm-cable-gear-section">
        <h3 className="lm-cable-gear-section__title">{L.captions}</h3>
        {!captionsSupported ? (
          <p className="lm-cable-gear-hint">{L.unsupported}</p>
        ) : (
          <>
            <label className="lm-cable-gear-field">
              <span>{L.srcLang}</span>
              <select value={sourceLang} onChange={(e) => onSourceLang(e.target.value)} disabled={captionsActive}>
                {CAPTION_SOURCE_LANGS.map((opt) => (
                  <option key={opt.code} value={opt.code}>
                    {he ? opt.labelHe : opt.labelEn}
                  </option>
                ))}
              </select>
            </label>
            <label className="lm-cable-gear-field">
              <span>{L.targetLang}</span>
              <select value={targetLang} onChange={(e) => onTargetLang(e.target.value)} disabled={captionsActive}>
                {CAPTION_TARGET_LANGS.map((opt) => (
                  <option key={opt.code} value={opt.code}>
                    {he ? opt.labelHe : opt.labelEn}
                  </option>
                ))}
              </select>
            </label>
            <button
              type="button"
              className={`lm-cable-gear-btn${captionsActive ? " lm-cable-gear-btn--stop" : ""}`}
              onClick={onToggleCaptions}
              disabled={captionsStatus === "starting"}
            >
              {captionsActive ? L.captionsOn : L.captionsOff}
            </button>
            {hint ? <p className="lm-cable-gear-hint">{hint}</p> : null}
          </>
        )}
      </section>
    </div>
  );

  return createPortal(menu, document.body);
}
