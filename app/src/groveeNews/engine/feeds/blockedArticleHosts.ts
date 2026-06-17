// @ts-nocheck
/** Article hosts that block scraping — skip page fetch / og:image (RSS image still OK). */
const BLOCKED_HOSTS = new Set([
  "thekitchn.com",
  "www.thekitchn.com",
  "bonappetit.com",
  "www.bonappetit.com",
  "eater.com",
  "www.eater.com",
]);

export function isBlockedArticleHost(url: string): boolean {
  try {
    const host = new URL(url).hostname.toLowerCase();
    return BLOCKED_HOSTS.has(host);
  } catch {
    return false;
  }
}
