import { fetchJson } from "../fetchJson";
import type { SearchSourceResult } from "../types";
import { buildGitHubSearchQuery } from "../intents";

const USER_AGENT = "GROVEEMODEL/1.0 (browser chat; web search)";

export const fetchGitHubSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "github" as const;
  const label = "GitHub Repositories";
  const ghq = buildGitHubSearchQuery(query);
  if (!ghq) {
    return {
      provider,
      label,
      ok: false,
      text: "",
      error: "אין שאילתת GitHub מתאימה",
      latencyMs: Math.round(performance.now() - started),
    };
  }
  try {
    const url = `https://api.github.com/search/repositories?q=${encodeURIComponent(ghq)}&sort=stars&order=desc&per_page=6`;
    const data = await fetchJson<{
      items?: Array<{
        full_name: string;
        description: string | null;
        html_url: string;
        stargazers_count: number;
        language: string | null;
      }>;
    }>(url, {
      headers: {
        Accept: "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": USER_AGENT,
      },
    });

    const items = data.items ?? [];
    if (!items.length) {
      return {
        provider,
        label,
        ok: false,
        text: "",
        error: `אין מאגרים עבור: ${ghq}`,
        latencyMs: Math.round(performance.now() - started),
      };
    }

    const text = items
      .map(
        (item) =>
          `- ${item.full_name}${item.language ? ` [${item.language}]` : ""}` +
          `${item.description ? `: ${item.description}` : ""} (${item.html_url}) ★${item.stargazers_count}`,
      )
      .join("\n");

    return {
      provider,
      label,
      ok: true,
      text: `שאילתה: ${ghq}\n${text}`,
      url: `https://github.com/search?q=${encodeURIComponent(ghq)}&type=repositories`,
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
