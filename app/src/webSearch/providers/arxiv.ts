import { fetchText } from "../fetchJson";
import type { SearchSourceResult } from "../types";

const parseArxivEntries = (xml: string, limit = 6): Array<{ title: string; id: string; summary: string }> => {
  const entries: Array<{ title: string; id: string; summary: string }> = [];
  const entryRe = /<entry>([\s\S]*?)<\/entry>/gi;
  let m: RegExpExecArray | null;
  while ((m = entryRe.exec(xml)) && entries.length < limit) {
    const block = m[1];
    const title = block.match(/<title>([\s\S]*?)<\/title>/i)?.[1]?.replace(/\s+/g, " ").trim() ?? "";
    const id = block.match(/<id>([\s\S]*?)<\/id>/i)?.[1]?.trim() ?? "";
    const summary =
      block
        .match(/<summary>([\s\S]*?)<\/summary>/i)?.[1]
        ?.replace(/\s+/g, " ")
        .trim()
        .slice(0, 280) ?? "";
    if (title) entries.push({ title, id, summary });
  }
  return entries;
};

export const fetchArxivSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "arxiv" as const;
  const label = "arXiv";
  const q = query
    .trim()
    .replace(/arxiv|arXiv|מאמר(?:ים)?|paper|papers|מחקר(?:ים)?/gi, " ")
    .trim();
  const searchQ = q.length >= 2 ? q : query.trim();
  if (!searchQ) {
    return {
      provider,
      label,
      ok: false,
      text: "",
      error: "שאילתה ריקה",
      latencyMs: Math.round(performance.now() - started),
    };
  }

  try {
    const url =
      `https://export.arxiv.org/api/query?search_query=all:${encodeURIComponent(searchQ)}` +
      `&start=0&max_results=6&sortBy=relevance&sortOrder=descending`;
    const xml = await fetchText(url);
    const entries = parseArxivEntries(xml);
    if (!entries.length) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: "אין מאמרים",
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const text = entries
      .map((e) => `- ${e.title}\n  ${e.summary}${e.summary ? "…" : ""}\n  ${e.id}`)
      .join("\n\n");

    return {
      provider,
      label,
      ok: true,
      text: `arXiv — "${searchQ}":\n${text}`,
      url: `https://arxiv.org/search/?query=${encodeURIComponent(searchQ)}&searchtype=all`,
      latencyMs: Math.round(performance.now() - started),
    };
  } catch (err) {
    return {
      provider,
      label,
      ok: false,
      text: "",
      error: err instanceof Error ? err.message : "שגיאה",
      latencyMs: Math.round(performance.now() - started),
    };
  }
};
