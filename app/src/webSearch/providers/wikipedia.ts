import { fetchJson } from "../fetchJson";
import type { SearchSourceResult } from "../types";

export const fetchWikipediaSearch = async (
  query: string,
  lang: "en" | "he",
): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = lang === "he" ? ("wikipedia-he" as const) : ("wikipedia-en" as const);
  const label = lang === "he" ? "ויקיפדיה (עברית)" : "Wikipedia (English)";
  try {
    const encoded = encodeURIComponent(query);
    const searchUrl =
      `https://${lang}.wikipedia.org/w/api.php?action=query&list=search&srsearch=${encoded}` +
      `&srlimit=3&format=json&origin=*`;
    const searchData = (await fetchJson<{
      query?: { search?: Array<{ title: string; snippet: string; pageid: number }> };
    }>(searchUrl)) as {
      query?: { search?: Array<{ title: string; snippet: string; pageid: number }> };
    };

    const hits = searchData.query?.search ?? [];
    if (!hits.length) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: "אין תוצאות",
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const pageIds = hits.map((h) => h.pageid).join("|");
    const detailUrl =
      `https://${lang}.wikipedia.org/w/api.php?action=query&prop=extracts|pageimages` +
      `&exintro=1&explaintext=1&piprop=thumbnail&pithumbsize=320` +
      `&pageids=${pageIds}&format=json&origin=*`;
    const detailData = (await fetchJson<{
      query?: {
        pages?: Record<
          string,
          { extract?: string; title?: string; thumbnail?: { source?: string } }
        >;
      };
    }>(detailUrl)) as {
      query?: {
        pages?: Record<
          string,
          { extract?: string; title?: string; thumbnail?: { source?: string } }
        >;
      };
    };

    const pages = detailData.query?.pages ?? {};
    const top = hits[0];
    const pageUrl = `https://${lang}.wikipedia.org/wiki/${encodeURIComponent(top.title.replace(/ /g, "_"))}`;

    const lines = hits.map((h, i) => {
      const url = `https://${lang}.wikipedia.org/wiki/${encodeURIComponent(h.title.replace(/ /g, "_"))}`;
      const page = pages[String(h.pageid)];
      const body =
        i === 0
          ? (page?.extract ?? h.snippet.replace(/<[^>]+>/g, "")).slice(0, 1200)
          : h.snippet.replace(/<[^>]+>/g, "");
      const thumb = page?.thumbnail?.source?.trim();
      const imageLine = thumb ? `  IMAGE: ${thumb}\n` : "";
      return i === 0
        ? `- ${h.title} (מלא):\n${body}\n${imageLine}  ${url}`
        : `- ${h.title}: ${body}\n${imageLine}  ${url}`;
    });

    return {
      provider,
      label,
      ok: true,
      text: lines.join("\n\n"),
      url: pageUrl,
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
