import { useEffect, useState } from "react";
import { buildCapabilitiesWelcomeMessage } from "../capabilitiesOnlyMode";

const SESSION_DISMISS_KEY = "grovee_capabilities_welcome_dismissed";

type Props = {
  failureReason?: string | null;
};

export function CapabilitiesWelcomeToast({ failureReason }: Props) {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    try {
      if (sessionStorage.getItem(SESSION_DISMISS_KEY) === "1") return;
    } catch {
      /* private mode */
    }
    const showTimer = window.setTimeout(() => setVisible(true), 500);
    return () => window.clearTimeout(showTimer);
  }, []);

  useEffect(() => {
    if (!visible) return;
    const hideTimer = window.setTimeout(() => dismiss(), 14_000);
    return () => window.clearTimeout(hideTimer);
  }, [visible]);

  const dismiss = () => {
    setVisible(false);
    try {
      sessionStorage.setItem(SESSION_DISMISS_KEY, "1");
    } catch {
      /* ignore */
    }
  };

  if (!visible) return null;

  return (
    <div className="grovee-capabilities-toast" role="status" dir="rtl" aria-live="polite">
      <span className="grovee-capabilities-toast-glow" aria-hidden="true" />
      <button
        type="button"
        className="grovee-capabilities-toast-close"
        onClick={dismiss}
        aria-label="סגור"
      >
        ×
      </button>
      <span className="grovee-capabilities-toast-icon" aria-hidden="true">
        ✦
      </span>
      <p className="grovee-capabilities-toast-body">
        {buildCapabilitiesWelcomeMessage(failureReason)}
      </p>
      <button type="button" className="grovee-capabilities-toast-ok" onClick={dismiss}>
        הבנתי
      </button>
    </div>
  );
}
