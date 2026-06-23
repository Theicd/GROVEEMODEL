import { useUiLanguage, setChatUiLanguage, type ChatUiLanguage } from "./useUiLanguage";

type UiLanguageToggleProps = {
  className?: string;
};

export function UiLanguageToggle({ className }: UiLanguageToggleProps = {}) {
  const lang = useUiLanguage();

  const pick = (next: ChatUiLanguage) => {
    if (next !== lang) setChatUiLanguage(next);
  };

  return (
    <div className={`ui-lang-toggle${className ? ` ${className}` : ""}`} role="group" aria-label="שפת ממשק">
      <button
        type="button"
        className={`ui-lang-toggle-btn${lang === "he" ? " ui-lang-toggle-btn--active" : ""}`}
        aria-pressed={lang === "he"}
        onClick={() => pick("he")}
      >
        עברית
      </button>
      <button
        type="button"
        className={`ui-lang-toggle-btn${lang === "en" ? " ui-lang-toggle-btn--active" : ""}`}
        aria-pressed={lang === "en"}
        onClick={() => pick("en")}
      >
        EN
      </button>
    </div>
  );
}
