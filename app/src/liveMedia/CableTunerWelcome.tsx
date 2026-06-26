import type { ChatUiLanguage } from "../ui/useUiLanguage";

const WELCOME_DISMISS_KEY = "grovee-tv-welcome-dismissed";

export function readTvWelcomeDismissed(): boolean {
  if (typeof sessionStorage === "undefined") return false;
  try {
    return sessionStorage.getItem(WELCOME_DISMISS_KEY) === "1";
  } catch {
    return false;
  }
}

export function dismissTvWelcome(): void {
  try {
    sessionStorage.setItem(WELCOME_DISMISS_KEY, "1");
  } catch {
    /* private mode */
  }
}

type Props = {
  uiLang: ChatUiLanguage;
  booting: boolean;
  channelCount: number;
  onStart: () => void;
};

export function CableTunerWelcome({ uiLang, booting, channelCount, onStart }: Props) {
  const he = uiLang === "he";
  const L = he
    ? {
        kicker: "TV LIVE",
        title: "ברוכים הבאים לשידור חי",
        lead: "ערוצי הטלוויזיה והרדיו שלך — מסך מפוצל, מדריך שידורים וחיפוש.",
        loading: "טוען מועדפים…",
        ready: (n: number) => `${n} ערוצים מוכנים לצפייה`,
        tips: [
          { icon: "▲▼", label: "למעלה / למטה", hint: "מעבר בין ערוצים" },
          { icon: "⊞", label: "תפריט", hint: "מסך מפוצל — 4 ערוצים" },
          { icon: "▦", label: "EPG", hint: "לוח שידורים" },
          { icon: "⌕", label: "חיפוש", hint: "מצא ערוצים והוסף ☆" },
          { icon: "⛶", label: "מסך מלא", hint: "הרחבת וידאו" },
          { icon: "🔊", label: "ווליום", hint: "עוצמת שמע" },
        ],
        start: "התחל לצפות",
        wait: "מכין ערוצים…",
        close: "סגור",
      }
    : {
        kicker: "TV LIVE",
        title: "Welcome to live TV",
        lead: "Your TV and radio channels — split view, TV guide, and search.",
        loading: "Loading favorites…",
        ready: (n: number) => `${n} channels ready`,
        tips: [
          { icon: "▲▼", label: "Up / Down", hint: "Change channel" },
          { icon: "⊞", label: "Menu", hint: "4-up split screen" },
          { icon: "▦", label: "EPG", hint: "TV guide" },
          { icon: "⌕", label: "Search", hint: "Find channels & star ☆" },
          { icon: "⛶", label: "Full screen", hint: "Expand video" },
          { icon: "🔊", label: "Volume", hint: "Audio level" },
        ],
        start: "Start watching",
        wait: "Preparing channels…",
        close: "Close",
      };

  const canDismiss = !booting && channelCount > 0;

  return (
    <div className="lm-cable-welcome" role="dialog" aria-modal="true" aria-labelledby="lm-cable-welcome-title">
      <div className="lm-cable-welcome__card" dir={he ? "rtl" : "ltr"}>
        <button
          type="button"
          className="lm-cable-welcome__close"
          onClick={onStart}
          disabled={!canDismiss}
          aria-label={L.close}
          title={L.close}
        >
          ×
        </button>
        <p className="lm-cable-welcome__kicker">{L.kicker}</p>
        <h2 id="lm-cable-welcome-title" className="lm-cable-welcome__title">
          {L.title}
        </h2>
        <p className="lm-cable-welcome__lead">{L.lead}</p>
        <p className="lm-cable-welcome__status" aria-live="polite">
          {booting ? L.loading : channelCount > 0 ? L.ready(channelCount) : L.loading}
        </p>
        <ul className="lm-cable-welcome__tips">
          {L.tips.map((tip) => (
            <li key={tip.label} className="lm-cable-welcome__tip">
              <span className="lm-cable-welcome__tip-icon" aria-hidden="true">
                {tip.icon}
              </span>
              <span className="lm-cable-welcome__tip-body">
                <strong>{tip.label}</strong>
                <span>{tip.hint}</span>
              </span>
            </li>
          ))}
        </ul>
        <button
          type="button"
          className="lm-cable-welcome__start"
          onClick={onStart}
          disabled={!canDismiss}
        >
          {canDismiss ? L.start : L.wait}
        </button>
      </div>
    </div>
  );
}
