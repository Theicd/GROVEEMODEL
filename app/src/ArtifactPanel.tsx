import { useMemo, useState } from "react";

export type Artifact = {
  kind: "code" | "html";
  lang?: string;
  content: string;
  title: string;
};

const normalizeHtmlForIframe = (fragmentOrDoc: string): string => {
  const t = fragmentOrDoc.trim();
  const headSample = t.slice(0, 600).toLowerCase();
  if (headSample.includes("<!doctype") || headSample.startsWith("<html")) {
    if (!headSample.includes("charset")) {
      if (/<head\b/i.test(t)) {
        return t.replace(/<head\b[^>]*>/i, (h) => `${h}<meta charset="utf-8">`);
      }
      return `<!DOCTYPE html><html lang="he" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><body>${t}</body></html>`;
    }
    return t;
  }
  return `<!DOCTYPE html><html lang="he" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>html,body{min-height:100%;margin:0;}</style></head><body>${t}</body></html>`;
};

export function ArtifactPanel({
  artifact,
  onClose,
  streaming,
}: {
  artifact: Artifact;
  onClose: () => void;
  streaming?: boolean;
}) {
  const [tab, setTab] = useState<"preview" | "source">("preview");
  const srcDoc = useMemo(
    () => (artifact.kind === "html" ? normalizeHtmlForIframe(artifact.content) : ""),
    [artifact.content, artifact.kind],
  );

  const panelTitle =
    artifact.kind === "html" ? "HTML · תצוגה" : `${artifact.title || artifact.lang || "code"} · קוד`;

  return (
    <div className="artifact-panel-inner">
      <header className="artifact-panel-head">
        <div className="artifact-panel-title">
          <span className="artifact-panel-dot" aria-hidden="true" />
          {panelTitle}
          {streaming ? <span className="artifact-stream-badge">מתעדכן…</span> : null}
        </div>
        <button type="button" className="artifact-panel-close" onClick={onClose} aria-label="סגור חלונית">
          ×
        </button>
      </header>

      {artifact.kind === "html" ? (
        <>
          <div className="artifact-panel-tabs" role="tablist">
            <button
              type="button"
              role="tab"
              aria-selected={tab === "preview"}
              className={`artifact-tab ${tab === "preview" ? "active" : ""}`}
              onClick={() => setTab("preview")}
            >
              תצוגה חיה
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={tab === "source"}
              className={`artifact-tab ${tab === "source" ? "active" : ""}`}
              onClick={() => setTab("source")}
            >
              מקור
            </button>
            <button
              type="button"
              className="artifact-copy-tab"
              onClick={() => void navigator.clipboard.writeText(artifact.content)}
            >
              העתק
            </button>
          </div>
          <div className="artifact-panel-body">
            {tab === "preview" ? (
              <iframe
                className="artifact-html-frame"
                title="HTML preview"
                sandbox="allow-scripts allow-forms"
                srcDoc={srcDoc}
              />
            ) : (
              <pre className="artifact-code-scroll">
                <code>{artifact.content}</code>
              </pre>
            )}
          </div>
        </>
      ) : (
        <>
          <div className="artifact-panel-tabs">
            <span className="artifact-lang-tag">{artifact.lang || "text"}</span>
            <button
              type="button"
              className="artifact-copy-tab"
              onClick={() => void navigator.clipboard.writeText(artifact.content)}
            >
              העתק
            </button>
          </div>
          <div className="artifact-panel-body">
            <pre className="artifact-code-scroll">
              <code className={artifact.lang ? `lang-${artifact.lang}` : undefined}>{artifact.content}</code>
            </pre>
          </div>
        </>
      )}
    </div>
  );
}
