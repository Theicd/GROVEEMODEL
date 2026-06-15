import { useEffect, useRef, useState } from "react";

type Props = {
  attachDisabled: boolean;
  onAttachClick: () => void;
  thinkingMode: boolean;
  onThinkingToggle: () => void;
  thinkingDisabled: boolean;
  cameraMode: boolean;
  onCameraToggle: () => void;
  cameraDisabled: boolean;
};

function MenuCheckbox({ checked }: { checked: boolean }) {
  return (
    <span
      className={`composer-plus-menu-box ${checked ? "composer-plus-menu-box--checked" : ""}`}
      aria-hidden="true"
    >
      {checked ? (
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
          <path d="M20 6 9 17l-5-5" />
        </svg>
      ) : null}
    </span>
  );
}

export function ComposerPlusMenu({
  attachDisabled,
  onAttachClick,
  thinkingMode,
  onThinkingToggle,
  thinkingDisabled,
  cameraMode,
  onCameraToggle,
  cameraDisabled,
}: Props) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div className="composer-plus-root" ref={rootRef}>
      <button
        type="button"
        className={`in-act in-plus ${open ? "in-plus--open" : ""}`}
        onClick={() => setOpen((v) => !v)}
        aria-label="אפשרויות נוספות"
        aria-expanded={open}
        aria-haspopup="menu"
        title="הוסף קובץ, חשיבה או מצלמה"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
          <path d="M12 5v14" />
          <path d="M5 12h14" />
        </svg>
      </button>

      {open ? (
        <div className="composer-plus-menu" role="menu" aria-label="אפשרויות קלט">
          <button
            type="button"
            role="menuitemcheckbox"
            aria-checked={thinkingMode}
            className={`composer-plus-menu-item composer-plus-menu-item--toggle ${thinkingMode ? "composer-plus-menu-item--active" : ""}`}
            disabled={thinkingDisabled}
            onClick={() => onThinkingToggle()}
          >
            <span className="composer-plus-menu-icon" aria-hidden="true">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M9 18h6" />
                <path d="M10 22h4" />
                <path d="M12 2a7 7 0 0 0-4 12v2h8v-2a7 7 0 0 0-4-12z" />
              </svg>
            </span>
            <span className="composer-plus-menu-label">חשיבה</span>
            <MenuCheckbox checked={thinkingMode} />
          </button>

          <button
            type="button"
            role="menuitem"
            className="composer-plus-menu-item"
            disabled={attachDisabled}
            onClick={() => {
              if (attachDisabled) return;
              onAttachClick();
              setOpen(false);
            }}
          >
            <span className="composer-plus-menu-icon" aria-hidden="true">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="3" y="3" width="18" height="18" rx="2" />
                <circle cx="8.5" cy="8.5" r="1.5" />
                <path d="M21 15l-5-5L5 21" />
              </svg>
            </span>
            <span className="composer-plus-menu-label">הוסף תמונה או קובץ</span>
          </button>

          <button
            type="button"
            role="menuitemcheckbox"
            aria-checked={cameraMode}
            className={`composer-plus-menu-item ${cameraMode ? "composer-plus-menu-item--active" : ""}`}
            disabled={cameraDisabled}
            onClick={() => {
              onCameraToggle();
              setOpen(false);
            }}
          >
            <span className="composer-plus-menu-icon" aria-hidden="true">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z" />
                <circle cx="12" cy="13" r="3" />
              </svg>
            </span>
            <span className="composer-plus-menu-label">מצלמה</span>
          </button>
        </div>
      ) : null}
    </div>
  );
}
