import { fetchText } from "../fetchJson";
import { topicalProviderQuery } from "../topicalEnrichment";
import type { SearchSourceResult } from "../types";

const decodeXml = (s: string): string =>
  s
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'");

const extractArxivSearchTerms = (query: string): string => {
  const raw = query.trim();
  const latin = raw.match(/[a-zA-Z][a-zA-Z0-9_.+\- ]{1,}/g)?.join(" ").trim();
  if (latin && latin.length >= 3) return latin.slice(0, 120);

  if (/transformer|טרנספורמר/i.test(raw)) return "transformer attention";
  if (/llm|שפה\s+גדול/i.test(raw)) return "large language model";
  if (/vision|ראייה/i.test(raw)) return "computer vision";
  if (/quantum|קוונט/i.test(raw)) return "quantum computing";
  if (/machine\s+learning|למידת\s+מכונה/i.test(raw)) return "machine learning";
  if (/רובוט|robotics/i.test(raw)) return "robotics";
  if (/גיימינג|gaming|esports/i.test(raw)) return "gaming";

  const topical = topicalProviderQuery(raw);
  if (topical && topical !== "technology trends news" && /[a-z]/i.test(topical)) {
    return topical.slice(0, 120);
  }

  const stripped = raw
    .replace(/(?:arxiv|ארxiv|מאמר(?:י|ים)?|paper|papers|preprint|חפש|search|find|על|about)/gi, " ")
    .trim();
  return stripped.slice(0, 80) || "artificial intelligence";
};

const parseArxivAtom = (xml: string): Array<{ title: string; id: string; summary: string; published: string }> => {
  const entries: Array<{ title: string; id: string; summary: string; published: string }> = [];
  const blocks = xml.match(/<entry>[\s\S]*?<\/entry>/g) ?? [];
  for (const block of blocks) {
    const title = decodeXml(block.match(/<title[^>]*>([\s\S]*?)<\/title>/)?.[1]?.trim() ?? "");
    const id = block.match(/<id>([\s\S]*?)<\/id>/)?.[1]?.trim() ?? "";
    const summary = decodeXml(
      (block.match(/<summary[^>]*>([\s\S]*?)<\/summary>/)?.[1] ?? "")
        .replace(/\s+/g, " ")
        .trim()
        .slice(0, 180),
    );
    const published = block.match(/<published>([\s\S]*?)<\/published>/)?.[1]?.trim().slice(0, 10) ?? "";
    if (title) entries.push({ title, id, summary, published });
  }
  return entries;
};

/** arXiv Atom API — free, no key. */
export const fetchArxivSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "arxiv" as const;
  const label = "arXiv (מאמרים)";

  try {
    const terms = extractArxivSearchTerms(query);
    const searchQuery = encodeURIComponent(`all:${terms}`);
    const url = `https://export.arxiv.org/api/query?search_query=${searchQuery}&start=0&max_results=5&sortBy=submittedDate&sortOrder=descending`;

    const xml = await fetchText(url, undefined, { timeoutMs: 15_000 });
    const entries = parseArxivAtom(xml);

    if (!entries.length) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: "לא נמצאו מאמרים",
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const lines = [
      `חיפוש arXiv: ${terms}`,
      ...entries.map((e, i) => {
        const link = e.id.replace("http://", "https://");
        return `${i + 1}. ${e.title}${e.published ? ` (${e.published})` : ""}\n   ${link}${e.summary ? `\n   ${e.summary}…` : ""}`;
      }),
      `ANSWER (arxiv top): ${entries[0].title}`,
    ];

    return {
      provider,
      label,
      ok: true,
      text: lines.join("\n"),
      url: entries[0].id.replace("http://", "https://"),
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
