import { useState } from "react";
import type { ChatUiLanguage } from "../ui/useUiLanguage";
import { accessModeLabel, statusBadgeLabel } from "../webSearch/hf/hfConnectionSnippets";
import type { UnifiedSearchHit } from "./types";

type Props = {
  hit: UnifiedSearchHit;
  uiLang: ChatUiLanguage;
};

const labels = {
  he: {
    pill: "מודל HF",
    open: "פתח ב-Hub",
    connect: "איך מתחברים",
    curl: "cURL",
    python: "Python",
    copy: "העתק",
    copied: "הועתק",
    provider: "ספק",
    latency: "זמן תגובה",
  },
  en: {
    pill: "HF Model",
    open: "Open on Hub",
    connect: "How to connect",
    curl: "cURL",
    python: "Python",
    copy: "Copy",
    copied: "Copied",
    provider: "Provider",
    latency: "Latency",
  },
} as const;

export function isHfModelHit(hit: UnifiedSearchHit): boolean {
  return hit.kind === "hfmodel";
}

export function HfModelSearchResultRow({ hit, uiLang }: Props) {
  const L = labels[uiLang];
  const [showConnect, setShowConnect] = useState(false);
  const [tab, setTab] = useState<"curl" | "python">("curl");
  const [copied, setCopied] = useState(false);

  const status = hit.meta?.hfStatus || "NOT PROBED";
  const statusClass =
    status === "WORKING"
      ? "serp-hf-status--ok"
      : status === "PROVIDER REQUIRED"
        ? "serp-hf-status--token"
        : "serp-hf-status--muted";

  const snippet = hit.snippetOriginal || hit.snippet;
  const code = tab === "curl" ? hit.meta?.hfCurl : hit.meta?.hfPython;

  const onCopy = async () => {
    if (!code) return;
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* ignore */
    }
  };

  return (
    <article className="serp-row serp-row--hfmodel" dir={uiLang === "he" ? "rtl" : "ltr"}>
      <div className="serp-row-site">
        <div className="serp-row-site-main">
          <img
            className="serp-row-favicon"
            src={hit.imageUrl || hit.faviconUrl}
            alt=""
            width={16}
            height={16}
          />
          <span className="serp-row-site-name">{hit.sourceLabel}</span>
        </div>
        <span className="serp-hf-pill">{L.pill}</span>
      </div>

      <a className="serp-row-title" href={hit.url} target="_blank" rel="noopener noreferrer" title={hit.title}>
        {hit.title}
      </a>

      <div className="serp-hf-badges">
        <span className={`serp-hf-status ${statusClass}`}>{statusBadgeLabel(status, uiLang)}</span>
        {hit.meta?.hfAccess ? (
          <span className="serp-hf-chip">{accessModeLabel(hit.meta.hfAccess, uiLang)}</span>
        ) : null}
        {hit.meta?.hfCategory ? <span className="serp-hf-chip">{hit.meta.hfCategory}</span> : null}
        {hit.meta?.hfProvider && hit.meta.hfProvider !== "Unknown" ? (
          <span className="serp-hf-chip">
            {L.provider}: {hit.meta.hfProvider}
          </span>
        ) : null}
        {hit.meta?.hfLatency != null ? (
          <span className="serp-hf-chip" dir="ltr">
            {L.latency}: {hit.meta.hfLatency}s
          </span>
        ) : null}
      </div>

      {snippet ? <p className="serp-row-snippet">{snippet}</p> : null}

      <div className="serp-hf-actions">
        <a className="serp-btn" href={hit.url} target="_blank" rel="noopener noreferrer">
          {L.open}
        </a>
        {hit.meta?.hfCurl ? (
          <button type="button" className="serp-btn serp-btn--ghost" onClick={() => setShowConnect((v) => !v)}>
            {L.connect}
          </button>
        ) : null}
      </div>

      {showConnect && code ? (
        <div className="serp-hf-connect">
          <div className="serp-hf-connect-tabs">
            <button
              type="button"
              className={`serp-hf-tab${tab === "curl" ? " serp-hf-tab--active" : ""}`}
              onClick={() => setTab("curl")}
            >
              {L.curl}
            </button>
            <button
              type="button"
              className={`serp-hf-tab${tab === "python" ? " serp-hf-tab--active" : ""}`}
              onClick={() => setTab("python")}
            >
              {L.python}
            </button>
            <button type="button" className="serp-hf-copy" onClick={() => void onCopy()}>
              {copied ? L.copied : L.copy}
            </button>
          </div>
          <pre className="serp-hf-code" dir="ltr">
            {code}
          </pre>
        </div>
      ) : null}
    </article>
  );
}
