import { buildYouTubeSearchQuery, buildMediaSearchQuery, buildMoviesSearchQuery } from "../intents";

import { fetchJson } from "../fetchJson";

import {

  youtubeEmbedUrl,

  youtubeThumbnail,

  youtubeWatchUrl,

} from "../youtubeUrls";

import type { MediaSerpHit, SearchSourceResult } from "../types";

import { searchAllYouTubeMedia } from "./youtubeMedia";



type InvidiousThumbnail = { quality?: string; url?: string; width?: number };



type InvidiousSearchHit = {

  type?: string;

  title?: string;

  videoId?: string;

  playlistId?: string;

  author?: string;

  authorId?: string;

  lengthSeconds?: number;

  videoCount?: number;

  videoThumbnails?: InvidiousThumbnail[];

  playlistThumbnails?: InvidiousThumbnail[];

  authorThumbnails?: InvidiousThumbnail[];

};



const DEFAULT_INSTANCES = [
  "https://invidious.privacyredirect.com",
  "https://inv.nadeko.net",
  "https://inv.tux.pizza",
  "https://yewtu.be",
  "https://invidious.nerdvpn.de",
];



const MAX_PER_TYPE = 8;



const getInvidiousInstances = (): string[] => {

  const env = (import.meta.env.VITE_INVIDIOUS_INSTANCES as string | undefined)?.trim();

  const fromEnv = env

    ? env

        .split(",")

        .map((s) => s.trim().replace(/\/$/, ""))

        .filter(Boolean)

    : [];

  return [...new Set([...fromEnv, ...DEFAULT_INSTANCES])];

};



export const invidiousSearchQuery = (query: string): string => {

  const cleaned =

    buildYouTubeSearchQuery(query) ||

    buildMediaSearchQuery(query) ||

    buildMoviesSearchQuery(query) ||

    query.trim();

  return cleaned.length >= 2 ? cleaned : "";

};



export function invidiousEmbedUrl(instance: string, videoId: string): string {

  return `${instance.replace(/\/$/, "")}/embed/${videoId}`;

}



export function pickInvidiousThumbnail(thumbnails: InvidiousThumbnail[] | undefined): string {

  if (!thumbnails?.length) return "";

  const ranked = [...thumbnails].sort((a, b) => (b.width ?? 0) - (a.width ?? 0));

  const medium = thumbnails.find((t) => t.quality === "medium")?.url;

  return medium?.trim() || ranked[0]?.url?.trim() || "";

}



export function mapInvidiousHit(_instance: string, row: InvidiousSearchHit): MediaSerpHit | null {

  if (row.type !== "video" || !row.videoId?.trim() || !row.title?.trim()) return null;

  const videoId = row.videoId.trim();

  return {

    id: `invidious-${videoId}`,

    mediaType: "video",

    youtubeSubType: "video",

    title: row.title.trim(),

    url: youtubeWatchUrl(videoId),

    playUrl: youtubeEmbedUrl(videoId),

    thumbnail: pickInvidiousThumbnail(row.videoThumbnails) || youtubeThumbnail(videoId),

    snippet: row.author ? `ערוץ: ${row.author}` : "",

    author: row.author,

    source: "YouTube",

    durationSec: row.lengthSeconds,

  };

}



export function mapInvidiousPlaylist(_instance: string, row: InvidiousSearchHit): MediaSerpHit | null {

  const playlistId = row.playlistId?.trim();

  if (row.type !== "playlist" || !playlistId || !row.title?.trim()) return null;

  return {

    id: `invidious-pl-${playlistId}`,

    mediaType: "video",

    youtubeSubType: "playlist",

    title: row.title.trim(),

    url: `https://www.youtube.com/playlist?list=${playlistId}`,

    playUrl: "",

    thumbnail: pickInvidiousThumbnail(row.playlistThumbnails ?? row.videoThumbnails),

    snippet: row.videoCount ? `פלייליסט · ${row.videoCount} סרטונים` : "פלייליסט",

    author: row.author,

    source: "YouTube",

  };

}



export function mapInvidiousChannel(_instance: string, row: InvidiousSearchHit): MediaSerpHit | null {

  const authorId = row.authorId?.trim();

  if (row.type !== "channel" || !authorId || !row.author?.trim()) return null;

  return {

    id: `invidious-ch-${authorId}`,

    mediaType: "video",

    youtubeSubType: "channel",

    title: row.author.trim(),

    url: `https://www.youtube.com/channel/${authorId}`,

    playUrl: "",

    thumbnail: pickInvidiousThumbnail(row.authorThumbnails ?? row.videoThumbnails),

    snippet: "ערוץ YouTube",

    author: row.author,

    source: "YouTube",

  };

}



const mapRow = (instance: string, row: InvidiousSearchHit): MediaSerpHit | null => {

  if (row.type === "playlist") return mapInvidiousPlaylist(instance, row);

  if (row.type === "channel") return mapInvidiousChannel(instance, row);

  return mapInvidiousHit(instance, row);

};



async function searchInvidiousType(

  instance: string,

  q: string,

  type: "video" | "playlist" | "channel",

): Promise<MediaSerpHit[]> {

  const url = `${instance}/api/v1/search?q=${encodeURIComponent(q)}&type=${type}`;

  const rows = await fetchJson<InvidiousSearchHit[]>(url, undefined, { timeoutMs: 9000 });

  return (rows ?? [])

    .slice(0, MAX_PER_TYPE)

    .map((row) => mapRow(instance, row))

    .filter((h): h is MediaSerpHit => h != null);

}



export async function searchInvidiousVideos(query: string): Promise<MediaSerpHit[]> {

  const q = invidiousSearchQuery(query);

  if (!q) return [];



  for (const instance of getInvidiousInstances()) {

    try {

      const [videos, playlists, channels] = await Promise.all([

        searchInvidiousType(instance, q, "video"),

        searchInvidiousType(instance, q, "playlist"),

        searchInvidiousType(instance, q, "channel"),

      ]);

      const merged = [...videos, ...playlists, ...channels];

      if (merged.length) return merged;

    } catch {

      continue;

    }

  }

  return [];

}



const emptyResult = (error: string, started: number): SearchSourceResult => ({

  provider: "invidious-videos",

  label: "YouTube",

  ok: false,

  text: "",

  error,

  latencyMs: Math.round(performance.now() - started),

});



export const fetchInvidiousVideosSearch = async (query: string): Promise<SearchSourceResult> => {

  const started = performance.now();

  const provider = "invidious-videos" as const;

  const label = "YouTube";



  const q = invidiousSearchQuery(query);

  if (!q) {

    return emptyResult("אין שאילתת YouTube מתאימה", started);

  }



  try {

    const hits = await searchAllYouTubeMedia(query);

    if (!hits.length) {

      return emptyResult(`לא נמצאו תוצאות YouTube עבור: ${q} (Invidious/Piped/SearXNG)`, started);

    }



    return {

      provider,

      label,

      ok: true,

      text: [`שאילתה: ${q} · YouTube`, ...hits.map((h, i) => `${i + 1}. ${h.title}`)].join("\n"),

      url: hits[0]?.url,

      mediaHits: hits,

      latencyMs: Math.round(performance.now() - started),

    };

  } catch (err) {

    return emptyResult(err instanceof Error ? err.message : "שגיאה בחיפוש YouTube", started);

  }

};


