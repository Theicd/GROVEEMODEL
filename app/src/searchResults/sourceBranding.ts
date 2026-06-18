/** Favicon + display name for SERP rows. */

export function hostFromUrl(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return "";
  }
}

export function faviconForUrl(url: string): string {
  const host = hostFromUrl(url);
  if (!host) return "";
  return `https://www.google.com/s2/favicons?domain=${encodeURIComponent(host)}&sz=32`;
}

export function displayPath(url: string): string {
  // Kept for callers — prefer displayBreadcrumb in SERP UI.
  try {
    const u = new URL(url);
    const host = u.hostname.replace(/^www\./, "");
    const path = u.pathname.replace(/\/$/, "");
    const short = `${host}${path}`;
    return short.length > 56 ? `${short.slice(0, 55)}…` : short;
  } catch {
    return url.length > 56 ? `${url.slice(0, 55)}…` : url;
  }
}

export function sourceLabelForHost(host: string): string {
  const map: Record<string, string> = {
    "ynet.co.il": "ynet",
    "www.jpost.com": "JPost",
    "www.timesofisrael.com": "Times of Israel",
    "www.bbc.com": "BBC",
    "techcrunch.com": "TechCrunch",
    "github.com": "GitHub",
    "arxiv.org": "arXiv",
    "news.ycombinator.com": "Hacker News",
  };
  return map[host] ?? host;
}
