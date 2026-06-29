import { useCallback, useEffect, useState } from "react";
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

type TourStep = {
  icon: string;
  title: string;
  body: string;
  bullets?: string[];
};

type Props = {
  uiLang: ChatUiLanguage;
  booting: boolean;
  channelCount: number;
  onStart: () => void;
};

function tourSteps(he: boolean, channelCount: number): TourStep[] {
  if (he) {
    return [
      {
        icon: "📺",
        title: "ברוכים הבאים לשידור חי",
        body: "כאן צופים בערוצי הטלוויזיה והרדיו מהמועדפים שלך — עם מסך מפוצל, מדריך שידורים וחיפוש.",
        bullets: [
          channelCount > 0 ? `${channelCount} ערוצים מוכנים לצפייה` : "טוען מועדפים…",
          "הסיור לוקח כדקה — אפשר לדלג בכל שלב",
        ],
      },
      {
        icon: "⊞",
        title: "מסך מפוצל — 4 ערוצים",
        body: "בפתיחה רואים 4 ערוצים במקביל. כל ריבוע מציג שידור חי; אחד מהם מנגן שמע.",
        bullets: [
          "לחיצה על ערוץ בבקר = בחירת שמע",
          "לחיצה כפולה או ▶ = מעבר לערוץ במסך מלא",
        ],
      },
      {
        icon: "▲▼",
        title: "חצים למעלה / למטה",
        body: "במסך מפוצל — מחליפים את כל 4 הערוצים קדימה או אחורה ברשימה. בערוץ בודד — עוברים לערוץ הבא או הקודם.",
        bullets: ["או השתמשו בכפתורי ▲ ▼ בתחתית המסך"],
      },
      {
        icon: "◀▶",
        title: "חצים ימין / שמאל",
        body: "במסך מפוצל בלבד — בוחרים איזה מהריבועים מנגן שמע, בלי לעזוב את המסך המפוצל.",
      },
      {
        icon: "▦",
        title: "מדריך שידורים (EPG)",
        body: "כפתור EPG בתחתית פותח לוח שידורים לפי קטגוריות. רואים מה משודר עכשיו ומה בהמשך.",
        bullets: ["זמין בערוץ בודד או כשבוחרים ערוץ במסך מפוצל"],
      },
      {
        icon: "⌕",
        title: "חיפוש ומועדפים",
        body: "כפתור החיפוש מחזיר לרשימת הערוצים — הוסיפו ☆ למועדפים, ערכו שם וקטגוריה, וסדרו קטגוריות בהגדרות.",
      },
      {
        icon: "✓",
        title: "מוכנים?",
        body: "הזיזו עכבר או לחצו על המסך כדי להציג את הבקר. תהנו!",
      },
    ];
  }
  return [
    {
      icon: "📺",
      title: "Welcome to live TV",
      body: "Watch your favorite TV and radio channels — split view, TV guide, and search.",
      bullets: [
        channelCount > 0 ? `${channelCount} channels ready` : "Loading favorites…",
        "This tour takes about a minute — skip anytime",
      ],
    },
    {
      icon: "⊞",
      title: "Split screen — 4 channels",
      body: "You start with four live channels at once. One tile plays audio.",
      bullets: ["Tap a tile in the bar to pick audio", "Double-tap or ▶ for full-screen channel"],
    },
    {
      icon: "▲▼",
      title: "Up / Down arrows",
      body: "On split screen — shift all four channels forward or back. On a single channel — next or previous.",
      bullets: ["Or use the ▲ ▼ buttons on the bottom bar"],
    },
    {
      icon: "◀▶",
      title: "Left / Right arrows",
      body: "On split screen only — choose which tile plays audio without leaving quad view.",
    },
    {
      icon: "▦",
      title: "TV guide (EPG)",
      body: "The EPG button opens a program guide by category — now playing and upcoming.",
      bullets: ["Works on single channel or when a quad tile is focused"],
    },
    {
      icon: "⌕",
      title: "Search & favorites",
      body: "Search finds channels — star ☆ to favorite, edit names and categories, reorder categories in settings.",
    },
    {
      icon: "✓",
      title: "Ready?",
      body: "Move the mouse or tap the screen to show controls. Enjoy!",
    },
  ];
}

export function CableTunerWelcome({ uiLang, booting, channelCount, onStart }: Props) {
  const he = uiLang === "he";
  const steps = tourSteps(he, channelCount);
  const [step, setStep] = useState(0);
  const last = step >= steps.length - 1;
  const canAdvance = !booting && channelCount > 0;

  const L = he
    ? {
        kicker: "TV LIVE",
        back: "הקודם",
        next: "הבא",
        skip: "דלג",
        start: "התחל לצפות",
        wait: "מכין ערוצים…",
        close: "סגור",
        stepOf: (a: number, b: number) => `שלב ${a} מתוך ${b}`,
      }
    : {
        kicker: "TV LIVE",
        back: "Back",
        next: "Next",
        skip: "Skip",
        start: "Start watching",
        wait: "Preparing channels…",
        close: "Close",
        stepOf: (a: number, b: number) => `Step ${a} of ${b}`,
      };

  const finish = useCallback(() => {
    if (!canAdvance) return;
    onStart();
  }, [canAdvance, onStart]);

  const goNext = useCallback(() => {
    if (!canAdvance) return;
    if (last) finish();
    else setStep((s) => Math.min(s + 1, steps.length - 1));
  }, [canAdvance, finish, last, steps.length]);

  const goBack = useCallback(() => {
    setStep((s) => Math.max(0, s - 1));
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        finish();
        return;
      }
      if (e.key === "ArrowRight" || e.key === "Enter") {
        e.preventDefault();
        goNext();
        return;
      }
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        if (step > 0) goBack();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [finish, goBack, goNext, step]);

  const current = steps[step]!;

  return (
    <div className="lm-cable-welcome" role="dialog" aria-modal="true" aria-labelledby="lm-cable-welcome-title">
      <div className="lm-cable-welcome__card lm-cable-welcome__card--tour" dir={he ? "rtl" : "ltr"}>
        <button
          type="button"
          className="lm-cable-welcome__close"
          onClick={finish}
          disabled={!canAdvance}
          aria-label={L.close}
          title={L.close}
        >
          ×
        </button>
        <p className="lm-cable-welcome__kicker">{L.kicker}</p>
        <p className="lm-cable-welcome__step-label" aria-live="polite">
          {L.stepOf(step + 1, steps.length)}
        </p>
        <div className="lm-cable-welcome__tour-icon" aria-hidden="true">
          {current.icon}
        </div>
        <h2 id="lm-cable-welcome-title" className="lm-cable-welcome__title">
          {current.title}
        </h2>
        <p className="lm-cable-welcome__lead">{current.body}</p>
        {current.bullets?.length ? (
          <ul className="lm-cable-welcome__bullets">
            {current.bullets.map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
        ) : null}
        <div className="lm-cable-welcome__dots" role="tablist" aria-label={L.stepOf(step + 1, steps.length)}>
          {steps.map((_, i) => (
            <button
              key={i}
              type="button"
              role="tab"
              className={`lm-cable-welcome__dot${i === step ? " is-active" : ""}`}
              aria-selected={i === step}
              aria-label={L.stepOf(i + 1, steps.length)}
              onClick={() => setStep(i)}
            />
          ))}
        </div>
        <div className="lm-cable-welcome__nav">
          {step > 0 ? (
            <button type="button" className="lm-cable-welcome__nav-btn lm-cable-welcome__nav-btn--ghost" onClick={goBack}>
              {L.back}
            </button>
          ) : (
            <button
              type="button"
              className="lm-cable-welcome__nav-btn lm-cable-welcome__nav-btn--ghost"
              onClick={finish}
              disabled={!canAdvance}
            >
              {L.skip}
            </button>
          )}
          <button
            type="button"
            className="lm-cable-welcome__nav-btn lm-cable-welcome__nav-btn--primary"
            onClick={goNext}
            disabled={!canAdvance}
          >
            {last ? (canAdvance ? L.start : L.wait) : L.next}
          </button>
        </div>
      </div>
    </div>
  );
}
