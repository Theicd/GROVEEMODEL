/** Split `owner/repo: description` for cleaner GitHub SERP titles. */
export function formatGithubTitleLine(title: string): { repo: string; description: string } {
  const idx = title.indexOf(":");
  if (idx > 0) {
    return {
      repo: title.slice(0, idx).trim(),
      description: title.slice(idx + 1).trim(),
    };
  }
  return { repo: title.trim(), description: "" };
}
