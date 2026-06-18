import { translateTexts } from "../../groveeNews/engine/translate/googleTranslate";
import { fetchJson } from "../fetchJson";
import { buildMediaSearchQuery } from "../intents";
import type { MediaSerpHit, SearchSourceResult } from "../types";

const DEFAULT_PIXABAY_KEY = "25540812-faf2b76d586c1787d2dd02736";
const DEFAULT_PER_PAGE = 12;

const pixabayApiKey = (): string =>
  (import.meta.env.VITE_PIXABAY_API_KEY as string | undefined)?.trim() || DEFAULT_PIXABAY_KEY;

const perPage = (): number => {
  const n = Number((import.meta.env.VITE_PIXABAY_PER_PAGE as string | undefined)?.trim());
  return Number.isFinite(n) && n > 0 ? Math.min(n, 24) : DEFAULT_PER_PAGE;
};

export const pixabaySearchQuery = async (query: string): Promise<string> => {
  const cleaned = buildMediaSearchQuery(query);
  if (!cleaned) return "";
  if (!/[\u0590-\u05FF]/.test(cleaned)) return cleaned;
  try {
    const { texts } = await translateTexts([cleaned], "en", "he");
    return texts[0]?.trim() || cleaned;
  } catch {
    return cleaned;
  }
};

type PixabayImageHit = {
  id: number;
  pageURL: string;
  previewURL: string;
  webformatURL: string;
  largeImageURL?: string;
  user: string;
  tags: string;
  imageWidth?: number;
  imageHeight?: number;
};

type PixabayVideoSize = {
  url: string;
  width?: number;
  height?: number;
  thumbnail?: string;
};

type PixabayVideoHit = {
  id: number;
  pageURL: string;
  picture_id?: string;
  webformatURL?: string;
  user: string;
  tags: string;
  duration: number;
  videos: {
    large?: PixabayVideoSize;
    medium?: PixabayVideoSize;
    small?: PixabayVideoSize;
    tiny?: PixabayVideoSize;
  };
};

export const pixabayVideoThumbnail = (video: PixabayVideoHit): string => {
  const v = video.videos;
  const fromApi =
    v?.medium?.thumbnail?.trim() ||
    v?.small?.thumbnail?.trim() ||
    v?.large?.thumbnail?.trim() ||
    v?.tiny?.thumbnail?.trim();
  if (fromApi) return fromApi;
  const pid = video.picture_id?.trim();
  if (pid) return `https://i.vimeocdn.com/video/${pid}_295x166.jpg`;
  return video.webformatURL?.trim() || "";
};

const mapImageHit = (img: PixabayImageHit): MediaSerpHit => ({
  id: `pixabay-img-${img.id}`,
  mediaType: "image",
  title: img.tags?.split(",")[0]?.trim() || `Pixabay · ${img.user}`,
  url: img.pageURL,
  playUrl: img.largeImageURL || img.webformatURL,
  downloadUrl: img.largeImageURL || img.webformatURL,
  thumbnail: img.previewURL,
  snippet: img.tags,
  author: img.user,
  licenseUrl: img.pageURL,
  tags: img.tags,
  width: img.imageWidth,
  height: img.imageHeight,
  source: "Pixabay",
});

const mapVideoHit = (video: PixabayVideoHit): MediaSerpHit | null => {
  const medium = video.videos?.medium?.url || video.videos?.small?.url;
  if (!medium) return null;
  const thumb = pixabayVideoThumbnail(video);
  return {
    id: `pixabay-vid-${video.id}`,
    mediaType: "video",
    title: video.tags?.split(",")[0]?.trim() || `Pixabay · ${video.user}`,
    url: video.pageURL,
    playUrl: medium,
    downloadUrl: video.videos?.large?.url || medium,
    thumbnail: thumb,
    snippet: video.tags,
    author: video.user,
    licenseUrl: video.pageURL,
    tags: video.tags,
    durationSec: video.duration,
    width: video.videos?.medium?.width,
    height: video.videos?.medium?.height,
    source: "Pixabay",
  };
};

const formatMediaText = (hits: MediaSerpHit[], query: string, label: string): string => {
  const lines = [`שאילתה: ${query} · ${label}`];
  hits.forEach((h, i) => {
    const extra = h.mediaType === "video" && h.durationSec ? ` · ${h.durationSec}s` : "";
    lines.push(`${i + 1}. ${h.title}${extra} (${h.licenseUrl || h.url})`);
  });
  return lines.join("\n");
};

const emptyResult = (
  provider: "pixabay-images" | "pixabay-videos",
  label: string,
  error: string,
  started: number,
): SearchSourceResult => ({
  provider,
  label,
  ok: false,
  text: "",
  error,
  latencyMs: Math.round(performance.now() - started),
});

export const fetchPixabayImagesSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "pixabay-images" as const;
  const label = "Pixabay · תמונות";
  const key = pixabayApiKey();
  if (!key) {
    return emptyResult(provider, label, "חסר VITE_PIXABAY_API_KEY", started);
  }

  const mq = await pixabaySearchQuery(query);
  if (!mq || mq.length < 2) {
    return emptyResult(provider, label, "אין שאילתת תמונות מתאימה", started);
  }

  try {
    const params = new URLSearchParams({
      key,
      q: mq,
      image_type: "photo",
      per_page: String(perPage()),
      lang: "en",
      safesearch: "true",
    });
    const data = await fetchJson<{ hits?: PixabayImageHit[] }>(
      `https://pixabay.com/api/?${params}`,
      undefined,
      { timeoutMs: 12_000 },
    );
    const hits = (data.hits ?? []).map(mapImageHit);
    if (!hits.length) {
      return emptyResult(provider, label, `לא נמצאו תמונות עבור: ${mq}`, started);
    }
    return {
      provider,
      label,
      ok: true,
      text: formatMediaText(hits, mq, "תמונות"),
      url: hits[0]?.licenseUrl,
      mediaHits: hits,
      latencyMs: Math.round(performance.now() - started),
    };
  } catch (err) {
    return emptyResult(
      provider,
      label,
      err instanceof Error ? err.message : "שגיאה בחיפוש תמונות",
      started,
    );
  }
};

export const fetchPixabayVideosSearch = async (query: string): Promise<SearchSourceResult> => {
  const started = performance.now();
  const provider = "pixabay-videos" as const;
  const label = "Pixabay · וידאו";
  const key = pixabayApiKey();
  if (!key) {
    return emptyResult(provider, label, "חסר VITE_PIXABAY_API_KEY", started);
  }

  const mq = await pixabaySearchQuery(query);
  if (!mq || mq.length < 2) {
    return emptyResult(provider, label, "אין שאילתת וידאו מתאימה", started);
  }

  try {
    const params = new URLSearchParams({
      key,
      q: mq,
      per_page: String(perPage()),
      lang: "en",
      safesearch: "true",
    });
    const data = await fetchJson<{ hits?: PixabayVideoHit[] }>(
      `https://pixabay.com/api/videos/?${params}`,
      undefined,
      { timeoutMs: 14_000 },
    );
    const hits = (data.hits ?? []).map(mapVideoHit).filter((h): h is MediaSerpHit => h != null);
    if (!hits.length) {
      return emptyResult(provider, label, `לא נמצאו סרטונים עבור: ${mq}`, started);
    }
    return {
      provider,
      label,
      ok: true,
      text: formatMediaText(hits, mq, "וידאו"),
      url: hits[0]?.licenseUrl,
      mediaHits: hits,
      latencyMs: Math.round(performance.now() - started),
    };
  } catch (err) {
    return emptyResult(
      provider,
      label,
      err instanceof Error ? err.message : "שגיאה בחיפוש וידאו",
      started,
    );
  }
};
