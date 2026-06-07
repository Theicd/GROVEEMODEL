import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.tsx";
import { ErrorBoundary } from "./ErrorBoundary.tsx";
import { installGlobalErrorHooks } from "./bootHelpers.ts";

installGlobalErrorHooks();

const rootEl = document.getElementById("root");
if (!rootEl) {
  document.body.insertAdjacentHTML(
    "beforeend",
    `<div style="padding:24px;font-family:system-ui;background:#121824;color:#e8eeff">Missing #root — broken HTML bundle.</div>`,
  );
} else {
  try {
    createRoot(rootEl).render(
      <StrictMode>
        <ErrorBoundary>
          <App />
        </ErrorBoundary>
      </StrictMode>,
    );
  } catch (e) {
    console.error("[GROVEE] createRoot failed:", e);
    const msg = e instanceof Error ? e.message : String(e);
    rootEl.innerHTML = "";
    rootEl.innerHTML = `<div style="padding:28px;max-width:520px;margin:48px auto;font-family:system-ui;background:#1a1f2e;color:#e8eeff;border-radius:14px;border:1px solid #3d4f77"><h1 style="font-size:1.15rem;margin:0 0 12px">לא ניתן להפעיל את React</h1><p style="opacity:.92;line-height:1.5">ייתכן שקובץ JS לא נטען (רשת, 404, אדבלוק).</p><pre style="white-space:pre-wrap;font-size:12px;background:#0f141f;padding:14px;border-radius:10px;margin-top:14px;opacity:.85">${msg.replace(/</g, "&lt;")}</pre><p style="margin-top:18px"><button type="button" onclick="location.reload()" style="padding:12px 20px;border-radius:10px;border:none;background:#3d7fff;color:#fff;font-weight:700;cursor:pointer">רענון</button></p></div>`;
  }
}
