import { useState } from "react";
import type { ChatUiLanguage } from "../ui/useUiLanguage";
import { accessModeLabel, statusBadgeLabel } from "../webSearch/hf/hfConnectionSnippets";
import { addHfHitToRack } from "../modelRack/modelRackScan";
import type { UnifiedSearchHit } from "./types";

type Props = {
  hit: UnifiedSearchHit;
  uiLang: ChatUiLanguage;
  onAddedToRack?: () => void;
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
    addRack: "הוסף לשיחה",
    addedRack: "נוסף",
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
    addRack: "Add to chat",
    addedRack: "Added",
  },
} as const;

export function isHfModelHit(hit: UnifiedSearchHit): boolean {
  return hit.kind === "hfmodel";
}

export function HfModelSearchResultRow({ hit, uiLang, onAddedToRack }: Props) {
  const L = labels[uiLang];
  const [showConnect, setShowConnect] = useState(false);
  const [tab, setTab] = useState<"curl" | "python">("curl");
  const [copied, setCopied] = useState(false);
  const [rackAdded, setRackAdded] = useState(false);

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

  const onAddToRack = () => {
    const modelId = (hit.titleOriginal || hit.title).trim();
    if (!modelId.includes("/")) return;
    const access = hit.meta?.hfAccess;
    if (hit.meta?.hfStatus !== "WORKING" || access !== "FREE") return;
    addHfHitToRack({
      id: `hf-${modelId.replace(/\//g, "--")}`,
      modelId,
      url: hit.url,
      title: modelId,
      snippet: hit.snippetOriginal || hit.snippet,
      pipelineTag: hit.meta?.hfPipeline,
      category: hit.meta?.hfCategory,
      status: hit.meta?.hfStatus || "NOT PROBED",
      provider: hit.meta?.hfProvider || "Unknown",
      accessMode: hit.meta?.hfAccess === "FREE" ? "FREE" : hit.meta?.hfAccess === "TOKEN" ? "TOKEN" : "UNKNOWN",
      endpoint: "",
      curlSnippet: hit.meta?.hfCurl || "",
      pythonSnippet: hit.meta?.hfPython || "",
      probed: !!hit.meta?.hfStatus,
      probeSource: "none",
    });
    setRackAdded(true);
    onAddedToRack?.();
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
        {hit.titleOriginal?.includes("/") &&
        hit.meta?.hfStatus === "WORKING" &&
        hit.meta?.hfAccess === "FREE" ? (
          <button
            type="button"
            className="serp-btn serp-btn--ghost"
            disabled={rackAdded}
            onClick={onAddToRack}
          >
            {rackAdded ? L.addedRack : L.addRack}
          </button>
        ) : null}
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
