import type { SearchIntent } from "./types";
import { isGeneralWebTopicQuery, isTopicalOverviewQuery } from "./intents";

/** Topic overview («מה חדש בגיימינג») — not bare world RSS. */
export const isTopicalOverviewRouting = (text: string): boolean =>
  isGeneralWebTopicQuery(text) || isTopicalOverviewQuery(text);

/** English search terms for GitHub / HN when query is Hebrew topical. */
export const topicalProviderQuery = (text: string): string => {
  const q = text.trim();
  if (/גיימינג|gaming|video\s*game|משחק(?:י)?(?:ם)?\b/i.test(q)) return "gaming video games esports";
  if (/רובוט|robotics/i.test(q)) return "robotics automation";
  if (/בינה\s+(?:ה)?מלאכותית|\bai\b|llm|machine\s+learning/i.test(q)) return "artificial intelligence LLM";
  if (/קריפטו|crypto|bitcoin|ביטקוין/i.test(q)) return "cryptocurrency bitcoin";
  if (/אקלים|climate|global\s+warming/i.test(q)) return "climate change environment";
  if (/סייבר|cyber|security/i.test(q)) return "cybersecurity";
  if (/חלל|space\b/i.test(q)) return "space exploration satellite";
  const latin = q.match(/[a-zA-Z][a-zA-Z0-9_.+\- ]{2,}/g)?.join(" ").trim();
  if (latin && latin.length >= 4) return latin.slice(0, 120);
  return "technology trends news";
};

/** ≥3 live APIs for topical «מה קורה / מה חדש» — no SearXNG-only dead-end. */
export const topicalEnrichmentIntents = (text: string): SearchIntent[] => {
  const q = text.trim();
  const out = new Set<SearchIntent>(["hackernews", "github"]);

  if (/בינה\s+(?:ה)?מלאכותית|\bai\b|llm|machine\s+learning|transformer/i.test(q)) {
    out.add("huggingface");
  }
  if (/רובוט|robotics|מדע/i.test(q)) {
    out.add("arxiv");
  }
  if (/קריפטו|crypto|bitcoin|ביטקוין|ethereum/i.test(q)) {
    out.add("crypto");
  }
  if (/אקלים|climate|סופה|hurricane|אסון|disaster/i.test(q)) {
    out.add("disaster");
    out.add("news");
  }

  return [...out];
};
