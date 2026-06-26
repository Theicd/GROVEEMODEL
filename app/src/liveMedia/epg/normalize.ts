/** Strip quality tags and noise from IPTV channel titles for EPG matching. */
export function normalizeChannelTitle(raw: string): string {
  return raw
    .replace(/#EXTINF:.*,/gi, "")
    .replace(/group-title="[^"]*"/gi, "")
    .replace(/like Gecko\)[^,]*,\s*/gi, "")
    .replace(/\s*\(\d+p\)\s*/gi, " ")
    .replace(/\s*\[[^\]]*\]\s*/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export function normalizeForMatch(raw: string): string {
  return normalizeChannelTitle(raw)
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9\u0590-\u05ff]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export function stripTvgFeed(tvgId: string): string {
  const at = tvgId.indexOf("@");
  return at >= 0 ? tvgId.slice(0, at) : tvgId;
}
