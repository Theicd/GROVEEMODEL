/** Install once: logs and keeps a small audit trail for support (F12 Console). */
export function installGlobalErrorHooks(): void {
  const seen = new Set<string>();

  window.addEventListener(
    "error",
    (ev) => {
      const msg = ev.error instanceof Error ? ev.error.stack ?? ev.error.message : ev.message;
      console.error("[GROVEE] window.error:", msg, ev.filename, ev.lineno);
    },
    true,
  );

  window.addEventListener("unhandledrejection", (ev) => {
    const r = ev.reason;
    const text = r instanceof Error ? r.stack ?? r.message : String(r);
    if (seen.has(text)) return;
    seen.add(text);
    console.error("[GROVEE] unhandledrejection:", text);
    // Extension noise — optional filter
    if (/message channel closed|Extension context/i.test(text)) {
      console.warn("[GROVEE] (ignored: likely browser extension)");
    }
  });
}
