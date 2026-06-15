import { fetchJson, fetchText } from "../fetchJson";
import type { SearchSourceResult } from "../types";
import {
  extractUrlsFromText,
  parseUrl,
  primaryUrlFromQuery,
  type ParsedUrl,
} from "../urlExtract";
import { fetchWikipediaSearch } from "./wikipedia";

const USER_AGENT = "GROVEEMODEL/1.0 (browser chat; url context)";
const README_MAX = 2400;
const GENERIC_MAX = 3200;

const ghHeaders = {
  Accept: "application/vnd.github+json",
  "X-GitHub-Api-Version": "2022-11-28",
  "User-Agent": USER_AGENT,
};

const decodeBase64Utf8 = (b64: string): string => {
  try {
    return decodeURIComponent(
      atob(b64.replace(/\s/g, ""))
        .split("")
        .map((c) => `%${(`00${c.charCodeAt(0).toString(16)}`).slice(-2)}`)
        .join(""),
    );
  } catch {
    return "";
  }
};

const stripHtmlToText = (html: string): string =>
  html
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/\s+/g, " ")
    .trim();

const truncate = (s: string, max: number): string =>
  s.length <= max ? s : `${s.slice(0, max - 1).trim()}…`;

const fetchGitHubRepo = async (parsed: Extract<ParsedUrl, { kind: "github-repo" }>): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "url-context" as const;
  const label = "GitHub — פרויקט (קישור)";
  const { owner, repo, url } = parsed;

  try {
    const meta = await fetchJson<{
      full_name?: string;
      description?: string | null;
      html_url?: string;
      stargazers_count?: number;
      forks_count?: number;
      language?: string | null;
      topics?: string[];
      pushed_at?: string;
      open_issues_count?: number;
      license?: { spdx_id?: string | null } | null;
      default_branch?: string;
    }>(`https://api.github.com/repos/${owner}/${repo}`, { headers: ghHeaders });

    let readme = "";
    try {
      readme = await fetchText(
        `https://raw.githubusercontent.com/${owner}/${repo}/${meta.default_branch ?? "main"}/README.md`,
        { headers: { "User-Agent": USER_AGENT } },
        { timeoutMs: 10_000 },
      );
    } catch {
      try {
        const readmeData = await fetchJson<{ content?: string }>(
          `https://api.github.com/repos/${owner}/${repo}/readme`,
          { headers: ghHeaders },
        );
        if (readmeData.content) readme = decodeBase64Utf8(readmeData.content);
      } catch {
        /* no readme */
      }
    }

    const lines = [
      `ANSWER (link preview): ${meta.full_name ?? `${owner}/${repo}`}`,
      `קישור: ${url}`,
      ...(meta.description ? [`תיאור: ${meta.description}`] : []),
      `★ ${meta.stargazers_count ?? 0} · forks ${meta.forks_count ?? 0}${meta.language ? ` · ${meta.language}` : ""}`,
      ...(meta.topics?.length ? [`נושאים: ${meta.topics.slice(0, 8).join(", ")}`] : []),
      ...(meta.pushed_at ? [`עודכן: ${meta.pushed_at.slice(0, 10)}`] : []),
      ...(meta.license?.spdx_id ? [`רישיון: ${meta.license.spdx_id}`] : []),
      ...(meta.open_issues_count != null ? [`issues פתוחים: ${meta.open_issues_count}`] : []),
    ];

    if (readme.trim()) {
      lines.push("README (קטע):");
      lines.push(truncate(readme.replace(/^#+\s+/gm, "").trim(), README_MAX));
    }

    return {
      provider,
      label,
      ok: true,
      text: lines.join("\n"),
      url: meta.html_url ?? url,
      latencyMs: Math.round(performance.now() - started),
    };
  } catch (err) {
    return {
      provider,
      label,
      ok: false,
      text: "",
      error: err instanceof Error ? err.message : "לא ניתן לטעון מאגר GitHub",
      latencyMs: Math.round(performance.now() - started),
    };
  }
};

const fetchHuggingFaceModel = async (
  parsed: Extract<ParsedUrl, { kind: "huggingface-model" }>,
): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "url-context" as const;
  const label = "Hugging Face — מודל (קישור)";
  try {
    const m = await fetchJson<{
      id?: string;
      modelId?: string;
      pipeline_tag?: string;
      library_name?: string;
      downloads?: number;
      likes?: number;
      tags?: string[];
      cardData?: { language?: string[]; license?: string };
    }>(`https://huggingface.co/api/models/${parsed.id}`, { headers: { "User-Agent": USER_AGENT } });

    const lines = [
      `ANSWER (link preview): ${m.id ?? m.modelId ?? parsed.id}`,
      `קישור: ${parsed.url}`,
      ...(m.pipeline_tag ? [`משימה: ${m.pipeline_tag}`] : []),
      ...(m.library_name ? [`ספרייה: ${m.library_name}`] : []),
      `⬇ ${m.downloads ?? 0} · ♥ ${m.likes ?? 0}`,
      ...(m.tags?.length ? [`תגיות: ${m.tags.slice(0, 10).join(", ")}`] : []),
      ...(m.cardData?.language?.length ? [`שפות: ${m.cardData.language.join(", ")}`] : []),
      ...(m.cardData?.license ? [`רישיון: ${m.cardData.license}`] : []),
    ];

    return {
      provider,
      label,
      ok: true,
      text: lines.join("\n"),
      url: parsed.url,
      latencyMs: Math.round(performance.now() - started),
    };
  } catch (err) {
    return {
      provider,
      label,
      ok: false,
      text: "",
      error: err instanceof Error ? err.message : "לא ניתן לטעון מודל Hugging Face",
      latencyMs: Math.round(performance.now() - started),
    };
  }
};

const fetchHuggingFaceDataset = async (
  parsed: Extract<ParsedUrl, { kind: "huggingface-dataset" }>,
): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "url-context" as const;
  const label = "Hugging Face — dataset (קישור)";
  try {
    const d = await fetchJson<{ id?: string; downloads?: number; likes?: number; tags?: string[] }>(
      `https://huggingface.co/api/datasets/${parsed.id}`,
      { headers: { "User-Agent": USER_AGENT } },
    );
    const lines = [
      `ANSWER (link preview): ${d.id ?? parsed.id}`,
      `קישור: ${parsed.url}`,
      `⬇ ${d.downloads ?? 0} · ♥ ${d.likes ?? 0}`,
      ...(d.tags?.length ? [`תגיות: ${d.tags.slice(0, 10).join(", ")}`] : []),
    ];
    return {
      provider,
      label,
      ok: true,
      text: lines.join("\n"),
      url: parsed.url,
      latencyMs: Math.round(performance.now() - started),
    };
  } catch (err) {
    return {
      provider,
      label,
      ok: false,
      text: "",
      error: err instanceof Error ? err.message : "לא ניתן לטעון dataset",
      latencyMs: Math.round(performance.now() - started),
    };
  }
};

const fetchArxivLink = async (parsed: Extract<ParsedUrl, { kind: "arxiv" }>): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "url-context" as const;
  const label = "arXiv — מאמר (קישור)";
  try {
    const xml = await fetchText(
      `https://export.arxiv.org/api/query?id_list=${encodeURIComponent(parsed.id)}`,
      undefined,
      { timeoutMs: 12_000 },
    );
    const title = xml.match(/<title>([^<]+)<\/title>/i)?.[1]?.replace(/\s+/g, " ").trim();
    const summary = xml.match(/<summary>([\s\S]*?)<\/summary>/i)?.[1]?.replace(/\s+/g, " ").trim();
    const published = xml.match(/<published>([^<]+)<\/published>/i)?.[1]?.slice(0, 10);
    const lines = [
      `ANSWER (link preview): ${title ?? parsed.id}`,
      `קישור: ${parsed.url}`,
      ...(published ? [`פורסם: ${published}`] : []),
      ...(summary ? [`תקציר: ${truncate(summary, 900)}`] : []),
    ];
    return {
      provider,
      label,
      ok: true,
      text: lines.join("\n"),
      url: parsed.url,
      latencyMs: Math.round(performance.now() - started),
    };
  } catch (err) {
    return {
      provider,
      label,
      ok: false,
      text: "",
      error: err instanceof Error ? err.message : "לא ניתן לטעון arXiv",
      latencyMs: Math.round(performance.now() - started),
    };
  }
};

const fetchWikipediaLink = async (
  parsed: Extract<ParsedUrl, { kind: "wikipedia" }>,
): Promise<SearchSourceResult> => {
  const lang = parsed.lang === "he" || parsed.lang === "en" ? parsed.lang : "en";
  const result = await fetchWikipediaSearch(parsed.title, lang);
  return {
    ...result,
    provider: "url-context",
    label: `Wikipedia — ${parsed.title}`,
    url: parsed.url,
  };
};

const fetchGenericPage = async (parsed: Extract<ParsedUrl, { kind: "generic" }>): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "url-context" as const;
  const label = "דף אינטרנט (קישור)";
  try {
    const html = await fetchText(parsed.url, { headers: { "User-Agent": USER_AGENT } }, { timeoutMs: 14_000 });
    const text = stripHtmlToText(html);
    if (!text || text.length < 40) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: "לא ניתן לחלץ טקסט מהדף (CORS או דף ריק)",
        latencyMs: Math.round(performance.now() - started),
      };
    }
    const titleMatch = html.match(/<title[^>]*>([^<]+)<\/title>/i)?.[1]?.trim();
    const lines = [
      `ANSWER (link preview): ${titleMatch ?? parsed.url}`,
      `קישור: ${parsed.url}`,
      `תוכן (קטע): ${truncate(text, GENERIC_MAX)}`,
    ];
    return {
      provider,
      label,
      ok: true,
      text: lines.join("\n"),
      url: parsed.url,
      latencyMs: Math.round(performance.now() - started),
    };
  } catch (err) {
    return {
      provider,
      label,
      ok: false,
      text: "",
      error: err instanceof Error ? err.message : "לא ניתן לטעון את הדף",
      latencyMs: Math.round(performance.now() - started),
    };
  }
};

const fetchByParsedUrl = (parsed: ParsedUrl): Promise<SearchSourceResult> => {
  switch (parsed.kind) {
    case "github-repo":
      return fetchGitHubRepo(parsed);
    case "huggingface-model":
      return fetchHuggingFaceModel(parsed);
    case "huggingface-dataset":
      return fetchHuggingFaceDataset(parsed);
    case "arxiv":
      return fetchArxivLink(parsed);
    case "wikipedia":
      return fetchWikipediaLink(parsed);
    default:
      return fetchGenericPage(parsed);
  }
};

/** Fetch structured context for URLs pasted into chat. */
export const fetchUrlContextSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "url-context" as const;
  const label = "קישור — תצוגה מקדימה";
  const url = primaryUrlFromQuery(query);
  if (!url) {
    return {
      provider,
      label,
      ok: false,
      text: "",
      error: "לא נמצא קישור בשאלה",
      latencyMs: Math.round(performance.now() - started),
    };
  }

  const parsed = parseUrl(url);
  const result = await fetchByParsedUrl(parsed);

  const extraUrls = extractUrlsFromText(query).slice(1, 2);
  if (extraUrls.length && result.ok) {
    const extras = await Promise.all(
      extraUrls.map(async (u) => {
        const r = await fetchByParsedUrl(parseUrl(u));
        return r.ok ? `\n---\n${r.text.split("\n").slice(0, 6).join("\n")}` : "";
      }),
    );
    result.text = result.text + extras.filter(Boolean).join("");
  }

  return result;
};
