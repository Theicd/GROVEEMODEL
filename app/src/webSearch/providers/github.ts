import { fetchJson } from "../fetchJson";
import type { SearchSourceResult } from "../types";
import { buildGitHubSearchQuery, isGitHubPopularQuery } from "../intents";

const USER_AGENT = "GROVEEMODEL/1.0 (browser chat; web search)";
const DESC_MAX = 140;

const trimDesc = (desc: string | null | undefined): string => {
  if (!desc?.trim()) return "";
  const clean = desc.replace(/\s+/g, " ").trim();
  return clean.length <= DESC_MAX ? clean : `${clean.slice(0, DESC_MAX - 1)}…`;
};

const formatRepoLine = (
  item: {
    full_name: string;
    description: string | null;
    html_url: string;
    stargazers_count: number;
    language: string | null;
  },
  index?: number,
): string => {
  const prefix = index != null ? `${index}. ` : "";
  const lang = item.language ? ` [${item.language}]` : "";
  const desc = trimDesc(item.description);
  const descPart = desc ? `: ${desc}` : "";
  return `${prefix}${item.full_name}${lang}${descPart} (${item.html_url}) ★${item.stargazers_count.toLocaleString("en-US")}`;
};

export const fetchGitHubSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "github" as const;
  const label = "GitHub Repositories";
  const ghq = buildGitHubSearchQuery(query);
  const popular = isGitHubPopularQuery(query);
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
    const perPage = popular ? 5 : 6;
    const url = `https://api.github.com/search/repositories?q=${encodeURIComponent(ghq)}&sort=stars&order=desc&per_page=${perPage}`;
    const data = await fetchJson<{
      items?: Array<{
        full_name: string;
        description: string | null;
        html_url: string;
        stargazers_count: number;
        language: string | null;
        pushed_at?: string;
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

    const top = items[0];
    const lines: string[] = [];
    if (popular) {
      lines.push(`ANSWER (GitHub top): ${top.full_name} ★${top.stargazers_count.toLocaleString("en-US")}`);
      lines.push(`סינון: ${ghq} · מיון: כוכבים (stars)`);
      lines.push(
        `הפרויקט הפופולרי ביותר בין מאגרים עם push אחרון לאחרונה: ${formatRepoLine(top)}`,
      );
      items.slice(1, 4).forEach((item, i) => lines.push(formatRepoLine(item, i + 2)));
    } else {
      lines.push(`שאילתה: ${ghq}`);
      items.forEach((item, i) => lines.push(formatRepoLine(item, i + 1)));
    }

    return {
      provider,
      label,
      ok: true,
      text: lines.join("\n"),
      url: `https://github.com/search?q=${encodeURIComponent(ghq)}&type=repositories&s=stars&o=desc`,
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
