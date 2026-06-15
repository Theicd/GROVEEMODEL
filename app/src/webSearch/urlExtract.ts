/** Detect and parse URLs pasted into chat (GitHub, Hugging Face, arXiv, …). */

export const URL_IN_TEXT_RE = /https?:\/\/[^\s<>"')\]\u0590-\u05FF]+/gi;

export const extractUrlsFromText = (text: string): string[] => {
  const matches = text.match(URL_IN_TEXT_RE) ?? [];
  const seen = new Set<string>();
  const out: string[] = [];
  for (const raw of matches) {
    const cleaned = raw.replace(/[.,;:!?)]+$/, "").trim();
    if (!cleaned || seen.has(cleaned)) continue;
    seen.add(cleaned);
    out.push(cleaned);
  }
  return out;
};

export const hasUrlInQuery = (text: string): boolean => extractUrlsFromText(text).length > 0;

export type ParsedUrl =
  | { kind: "github-repo"; owner: string; repo: string; url: string }
  | { kind: "huggingface-model"; id: string; url: string }
  | { kind: "huggingface-dataset"; id: string; url: string }
  | { kind: "arxiv"; id: string; url: string }
  | { kind: "wikipedia"; lang: string; title: string; url: string }
  | { kind: "generic"; url: string };

export const parseUrl = (rawUrl: string): ParsedUrl => {
  let url: URL;
  try {
    url = new URL(rawUrl);
  } catch {
    return { kind: "generic", url: rawUrl };
  }

  const host = url.hostname.replace(/^www\./, "");

  if (host === "github.com") {
    const parts = url.pathname.split("/").filter(Boolean);
    if (parts.length >= 2 && !["orgs", "organizations", "settings", "marketplace"].includes(parts[0])) {
      return { kind: "github-repo", owner: parts[0], repo: parts[1], url: rawUrl };
    }
  }

  if (host === "huggingface.co") {
    const parts = url.pathname.split("/").filter(Boolean);
    if (parts[0] === "datasets" && parts.length >= 3) {
      return { kind: "huggingface-dataset", id: `${parts[1]}/${parts[2]}`, url: rawUrl };
    }
    if (parts[0] === "models" && parts.length >= 3) {
      return { kind: "huggingface-model", id: `${parts[1]}/${parts[2]}`, url: rawUrl };
    }
    if (parts.length >= 2 && !["docs", "blog", "pricing", "join", "settings"].includes(parts[0])) {
      return { kind: "huggingface-model", id: `${parts[0]}/${parts[1]}`, url: rawUrl };
    }
  }

  if (host === "arxiv.org") {
    const m = url.pathname.match(/\/(?:abs|pdf)\/([\d.]+v?\d*)/i);
    if (m?.[1]) return { kind: "arxiv", id: m[1].replace(/\.pdf$/i, ""), url: rawUrl };
  }

  if (host.endsWith("wikipedia.org")) {
    const lang = host.split(".")[0] ?? "en";
    const title = decodeURIComponent(url.pathname.replace(/^\/wiki\//, "").replace(/_/g, " "));
    if (title) return { kind: "wikipedia", lang, title, url: rawUrl };
  }

  return { kind: "generic", url: rawUrl };
};

export const isGitHubRepoUrlInQuery = (text: string): boolean =>
  extractUrlsFromText(text).some((u) => parseUrl(u).kind === "github-repo");

export const primaryUrlFromQuery = (text: string): string | null => extractUrlsFromText(text)[0] ?? null;
