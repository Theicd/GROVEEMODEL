/** Parse YouTube video IDs and build embed/thumbnail URLs for SERP hits. */

const YOUTUBE_HOST =
  /^(?:www\.|m\.|music\.)?youtube\.com$|^youtu\.be$/i;

export function parseYouTubeVideoId(url: string): string | null {
  try {
    const u = new URL(url.trim());
    const host = u.hostname.replace(/^www\./, "");
    if (host === "youtu.be") {
      const id = u.pathname.replace(/^\//, "").split("/")[0];
      return id && id.length >= 6 ? id : null;
    }
    if (YOUTUBE_HOST.test(u.hostname)) {
      const v = u.searchParams.get("v");
      if (v && v.length >= 6) return v;
      const embed = u.pathname.match(/^\/embed\/([^/?]+)/);
      if (embed?.[1]) return embed[1];
      const shorts = u.pathname.match(/^\/shorts\/([^/?]+)/);
      if (shorts?.[1]) return shorts[1];
    }
    if (u.searchParams.get("v") && /invidious|yewtu/i.test(host)) {
      const v = u.searchParams.get("v");
      if (v && v.length >= 6) return v;
    }
  } catch {
    /* invalid URL */
  }
  return null;
}

export function isYouTubeUrl(url: string): boolean {
  if (!url.trim()) return false;
  if (parseYouTubeVideoId(url)) return true;
  try {
    const host = new URL(url.trim()).hostname.toLowerCase();
    return host.includes("youtube.com") || host === "youtu.be";
  } catch {
    return /youtube\.com|youtu\.be/i.test(url);
  }
}

export function youtubeThumbnail(videoId: string): string {
  return `https://i.ytimg.com/vi/${videoId}/hqdefault.jpg`;
}

export function youtubeEmbedUrl(videoId: string): string {
  return `https://www.youtube-nocookie.com/embed/${videoId}`;
}

export function youtubeWatchUrl(videoId: string): string {
  return `https://www.youtube.com/watch?v=${videoId}`;
}
