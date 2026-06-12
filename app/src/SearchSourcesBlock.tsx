import type { SearchSourceResult } from "./webSearch/types";

type Props = {
  sources: SearchSourceResult[];
  summary?: string;
};

export function SearchSourcesBlock({ sources, summary }: Props) {
  const ok = sources.filter((s) => s.ok && s.text.trim());
  const failed = sources.filter((s) => !s.ok);
  if (!ok.length && !failed.length) return null;

  return (
    <div className="search-sources-block" dir="rtl">
      <div className="search-sources-header">
        <span className="search-sources-icon" aria-hidden="true">
          🔍
        </span>
        <span>{summary ?? `מקורות חיפוש (${ok.length})`}</span>
      </div>
      {ok.map((s) => (
        <details key={s.provider} className="search-source-item search-source-item--ok" open={ok.length === 1}>
          <summary>
            {s.label}
            <span className="search-source-latency">{s.latencyMs}ms</span>
          </summary>
          <pre className="search-source-body">{s.text}</pre>
          {s.url ? (
            <a className="search-source-link" href={s.url} target="_blank" rel="noopener noreferrer">
              {s.url}
            </a>
          ) : null}
        </details>
      ))}
      {failed.length ? (
        <p className="search-sources-failed">
          לא נטענו: {failed.map((f) => `${f.label}${f.error ? ` (${f.error})` : ""}`).join(" · ")}
        </p>
      ) : null}
    </div>
  );
}
