import { faviconForUrl } from "./sourceBranding";
import type { UnifiedSearchHit } from "./types";

const parseGithubLine = (trimmed: string): Omit<UnifiedSearchHit, "id" | "provider"> | null => {
  const urlMatch = trimmed.match(/\((https?:\/\/[^)]+)\)/);
  const url = urlMatch?.[1] ?? "";
  if (!url) return null;

  const starsMatch = trimmed.match(/★([\d,]+)/);
  const stars = starsMatch ? parseInt(starsMatch[1].replace(/,/g, ""), 10) : undefined;

  let body = trimmed
    .replace(/^\d+\.\s*/, "")
    .replace(/\s*\(https?:\/\/[^)]+\)/, "")
    .replace(/\s*★[\d,]+$/, "")
    .trim();

  const colonIdx = body.indexOf(":");
  let repoPart = body;
  let description = "";
  if (colonIdx > 0) {
    repoPart = body.slice(0, colonIdx).trim();
    description = body.slice(colonIdx + 1).trim();
  }

  const langMatch = repoPart.match(/\s*\[([^\]]+)\]\s*$/);
  const language = langMatch?.[1];
  const repo = repoPart.replace(/\s*\[[^\]]+\]\s*$/, "").trim();

  const title = description ? `${repo}: ${description}` : repo;

  return {
    kind: "github",
    title,
    url,
    snippet: "",
    sourceLabel: "GitHub",
    faviconUrl: faviconForUrl("https://github.com"),
    score: stars ?? 50,
    meta: { stars, engine: language },
    summarizable: false,
  };
};

const parseHackerNewsLine = (trimmed: string): Omit<UnifiedSearchHit, "id" | "provider"> | null => {
  if (!/^\d+\./.test(trimmed)) return null;
  const urlMatch = trimmed.match(/(https?:\/\/\S+)/);
  const url = urlMatch?.[1] ?? "https://news.ycombinator.com";
  const scoreMatch = trimmed.match(/\(★([\d,]+)\)/);
  const score = scoreMatch ? parseInt(scoreMatch[1].replace(/,/g, ""), 10) : undefined;
  const title = trimmed
    .replace(/^\d+\.\s*/, "")
    .replace(/\(★[\d,]+\)/, "")
    .replace(url, "")
    .trim();

  return {
    kind: "hackernews",
    title: title || "Hacker News",
    url,
    snippet: score != null ? `${score.toLocaleString("en-US")} נקודות ב-HN` : "",
    sourceLabel: "Hacker News",
    faviconUrl: faviconForUrl("https://news.ycombinator.com"),
    score: 40,
    meta: score != null ? { stars: score } : undefined,
    summarizable: /^https?:\/\//.test(url) && !url.includes("ycombinator.com"),
  };
};

export const parseArxivText = (text: string): UnifiedSearchHit[] => {
  const hits: UnifiedSearchHit[] = [];
  const lines = text.split("\n");
  let current: { title: string; url: string; snippet: string; published?: string } | null = null;

  const flush = () => {
    if (!current?.url) {
      current = null;
      return;
    }
    hits.push({
      id: `arxiv-${hits.length}-${current.url}`,
      kind: "arxiv",
      title: current.title,
      url: current.url,
      snippet: current.snippet,
      sourceLabel: "arXiv",
      faviconUrl: faviconForUrl("https://arxiv.org"),
      provider: "arxiv",
      score: 35,
      summarizable: false,
      meta: current.published ? { engine: current.published } : undefined,
    });
    current = null;
  };

  for (const raw of lines) {
    const line = raw.trim();
    if (!line) continue;
    if (/^ANSWER\s*\(/i.test(line) || /^חיפוש arxiv:/i.test(line)) continue;

    if (/^\d+\.\s/.test(line)) {
      flush();
      const titleLine = line.replace(/^\d+\.\s*/, "");
      const dateMatch = titleLine.match(/^(.+?)\s*\((\d{4}-\d{2}-\d{2})\)\s*$/);
      current = {
        title: dateMatch ? dateMatch[1].trim() : titleLine,
        url: "",
        snippet: "",
        published: dateMatch?.[2],
      };
      continue;
    }

    if (!current) continue;
    if (/^https?:\/\/arxiv/i.test(line)) {
      current.url = line.startsWith("http://") ? line.replace("http://", "https://") : line;
      continue;
    }
    if (!current.snippet) {
      current.snippet = line.replace(/…$/, "").trim();
    }
  }
  flush();
  return hits;
};

export const parseGithubLines = (text: string): UnifiedSearchHit[] => {
  const hits: UnifiedSearchHit[] = [];
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (!/^\d+\./.test(trimmed)) continue;
    const parsed = parseGithubLine(trimmed);
    if (!parsed) continue;
    hits.push({
      ...parsed,
      id: `github-${hits.length}-${parsed.url}`,
      provider: "github",
    });
  }
  return hits;
};

export const parseHackerNewsLines = (text: string): UnifiedSearchHit[] => {
  const hits: UnifiedSearchHit[] = [];
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    const parsed = parseHackerNewsLine(trimmed);
    if (!parsed) continue;
    hits.push({
      ...parsed,
      id: `hn-${hits.length}-${parsed.url}`,
      provider: "hacker-news",
    });
  }
  return hits;
};

export const parseWikipediaText = (text: string, provider: "wikipedia-en" | "wikipedia-he"): UnifiedSearchHit[] => {
  const hits: UnifiedSearchHit[] = [];
  const lang = provider === "wikipedia-he" ? "he" : "en";
  const sourceLabel = lang === "he" ? "ויקיפדיה" : "Wikipedia";

  for (const section of text.split(/\n(?=- )/)) {
    const trimmed = section.trim();
    if (!trimmed.startsWith("- ")) continue;
    const urlMatch = trimmed.match(/(https:\/\/\w+\.wikipedia\.org\/wiki\/\S+)/);
    const url = urlMatch?.[1] ?? "";
    if (!url) continue;

    const lines = trimmed.split("\n").map((l) => l.trim()).filter(Boolean);
    const title = lines[0]
      .replace(/^-\s+/, "")
      .replace(/\s*\(מלא\):?\s*$/, "")
      .replace(/:\s*$/, "")
      .trim();

    let snippet = "";
    let imageUrl: string | undefined;
    for (let i = 1; i < lines.length; i++) {
      const imgMatch = lines[i].match(/^IMAGE:\s*(https?:\/\/\S+)/i);
      if (imgMatch) {
        imageUrl = imgMatch[1];
        continue;
      }
      if (/^https?:\/\//.test(lines[i])) break;
      snippet += (snippet ? " " : "") + lines[i];
    }
    snippet = snippet.replace(/<[^>]+>/g, "").trim().slice(0, 280);

    hits.push({
      id: `wiki-${lang}-${hits.length}-${url}`,
      kind: "web",
      title,
      url,
      snippet,
      imageUrl,
      sourceLabel,
      faviconUrl: faviconForUrl(url),
      provider,
      score: lang === "he" ? 44 : 43,
      meta: { engine: sourceLabel },
      summarizable: true,
    });
  }
  return hits;
};
