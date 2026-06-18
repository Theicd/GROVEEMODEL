import { describe, expect, it } from "vitest";
import { buildSearchBrief, formatSearchBriefContext } from "./searchBrief";
import type { SearchSourceResult } from "./types";

describe("searchBrief github formatting", () => {
  it("includes numbered GitHub repo lines in facts", () => {
    const sources: SearchSourceResult[] = [
      {
        provider: "github",
        label: "GitHub Repositories",
        ok: true,
        text: "שאילתה: robotics\n1. org/repo [Rust]: demo (https://github.com/org/repo) ★500",
        latencyMs: 100,
      },
    ];
    const brief = buildSearchBrief(sources, ["github"], "robotics");
    expect(brief.facts.some((f) => f.includes("org/repo"))).toBe(true);
    const ctx = formatSearchBriefContext(brief, "robotics", 900, sources);
    expect(ctx).toContain("org/repo");
  });
});
