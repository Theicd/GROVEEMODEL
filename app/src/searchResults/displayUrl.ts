/** Google-style breadcrumb: `example.com › blog › article…` */
export function displayBreadcrumb(url: string, maxLen = 56): string {
  try {
    const u = new URL(url);
    const host = u.hostname.replace(/^www\./, "");
    const segments = u.pathname.split("/").filter(Boolean).map((s) => {
      try {
        return decodeURIComponent(s);
      } catch {
        return s;
      }
    });

    let crumb = host;
    for (const seg of segments) {
      const piece = seg.length > 28 ? `${seg.slice(0, 26)}…` : seg;
      const next = `${crumb} › ${piece}`;
      if (next.length > maxLen) {
        return `${crumb} › …`;
      }
      crumb = next;
    }

    if (crumb.length > maxLen) {
      return `${crumb.slice(0, maxLen - 1)}…`;
    }
    return crumb;
  } catch {
    const trimmed = url.trim();
    return trimmed.length > maxLen ? `${trimmed.slice(0, maxLen - 1)}…` : trimmed;
  }
}

export function googleTranslatePageUrl(pageUrl: string, targetLang: string): string {
  const tl = targetLang.trim().toLowerCase() || "en";
  return `https://translate.google.com/translate?sl=auto&tl=${encodeURIComponent(tl)}&u=${encodeURIComponent(pageUrl)}`;
}
