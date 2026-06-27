import { useEffect } from "react";

/** Sync visual viewport height + keyboard inset for mobile chat composer. */
export function useMobileKeyboardInset(enabled = true): void {
  useEffect(() => {
    if (!enabled || typeof window === "undefined") return;
    const mq = window.matchMedia("(max-width: 768px)");
    const root = document.documentElement;
    const vv = window.visualViewport;

    const clear = () => {
      root.style.removeProperty("--app-height");
      root.style.removeProperty("--keyboard-inset");
    };

    const sync = () => {
      if (!mq.matches || !vv) {
        clear();
        return;
      }
      const keyboardInset = Math.max(0, window.innerHeight - vv.height - vv.offsetTop);
      root.style.setProperty("--app-height", `${Math.round(vv.height)}px`);
      root.style.setProperty("--keyboard-inset", `${Math.round(keyboardInset)}px`);
    };

    sync();
    vv?.addEventListener("resize", sync);
    vv?.addEventListener("scroll", sync);
    window.addEventListener("resize", sync);
    mq.addEventListener("change", sync);

    return () => {
      vv?.removeEventListener("resize", sync);
      vv?.removeEventListener("scroll", sync);
      window.removeEventListener("resize", sync);
      mq.removeEventListener("change", sync);
      clear();
    };
  }, [enabled]);
}
