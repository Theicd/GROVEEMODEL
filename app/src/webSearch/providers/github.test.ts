import { describe, expect, it, vi, beforeEach } from "vitest";
import { fetchGitHubSearch } from "./github";

vi.mock("../fetchJson", () => ({
  fetchJson: vi.fn(),
}));

import { fetchJson } from "../fetchJson";

const mockFetch = vi.mocked(fetchJson);

describe("fetchGitHubSearch", () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  it("uses stars/pushed filter for B16 instead of bare GitHub", async () => {
    mockFetch.mockResolvedValue({
      items: [
        {
          full_name: "owner/repo",
          description: "A maintained project",
          html_url: "https://github.com/owner/repo",
          stargazers_count: 12000,
          language: "TypeScript",
        },
      ],
    });

    const result = await fetchGitHubSearch("מהו הפרויקט הפופולרי ביותר היום ב-GitHub?");
    expect(result.ok).toBe(true);
    expect(result.text).toContain("ANSWER (GitHub top): owner/repo");
    const calledUrl = decodeURIComponent(String(mockFetch.mock.calls[0]?.[0]));
    expect(calledUrl).toMatch(/stars:>500/);
    expect(calledUrl).not.toMatch(/q=GitHub(?:&|$)/i);
  });

  it("truncates long descriptions", async () => {
    mockFetch.mockResolvedValue({
      items: [
        {
          full_name: "a/b",
          description: "x".repeat(200),
          html_url: "https://github.com/a/b",
          stargazers_count: 1,
          language: null,
        },
      ],
    });

    const result = await fetchGitHubSearch("open source llm chat");
    expect(result.ok).toBe(true);
    expect(result.text?.length ?? 0).toBeLessThan(400);
  });
});
