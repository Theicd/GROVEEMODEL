const normalize = (text: string): string =>
  text
    .toLowerCase()
    .replace(/\s+/g, " ")
    .replace(/[★*•·]/g, "")
    .trim();

/** Drop snippets that repeat the title or full provider line. */
export function cleanDisplaySnippet(title: string, snippet: string, url?: string): string {
  let s = snippet.trim();
  if (!s) return "";

  const t = title.trim();
  if (url) {
    s = s.replace(url, "").trim();
  }

  if (!s || s === t) return "";

  if (t && s.startsWith(t)) {
    const rest = s.slice(t.length).replace(/^[\s:—\-–|]+/, "").trim();
    return rest.length >= 12 ? rest : "";
  }

  if (t && s.includes(t)) {
    const rest = s.slice(s.indexOf(t) + t.length).replace(/^[\s:—\-–|]+/, "").trim();
    if (rest.length >= 24) s = rest;
    else if (normalize(s) === normalize(t)) return "";
  }

  const ns = normalize(s);
  const nt = normalize(t);
  if (!ns || ns === nt) return "";
  if (nt.length > 20 && ns.startsWith(nt.slice(0, Math.min(nt.length, 48)))) {
    const rest = s.slice(t.length).replace(/^[\s:—\-–|]+/, "").trim();
    if (rest.length >= 24) return rest;
    if (s.length <= t.length + 40) return "";
  }

  return s;
}
