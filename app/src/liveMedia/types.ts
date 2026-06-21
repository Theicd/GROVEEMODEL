/** Live TV / Radio — ported from GROVEE LIVE TV (IPTV OS). */

export type ChannelType = "tv" | "radio" | "youtube";
export type StreamStatus = "working" | "warning" | "offline" | "unknown";

export interface Channel {
  id: string;
  name: string;
  logo: string;
  country: string;
  language: string;
  /** All detected broadcast languages (ISO-ish codes). */
  languages?: string[];
  /** Where primary language came from. */
  languageSource?: "tvg" | "inferred" | "default";
  category: string;
  stream: string;
  source: string;
  type: ChannelType;
  status: StreamStatus;
  lastCheck: number;
  favorite: boolean;
  tags?: string[];
  tvgId?: string;
  groupTitle?: string;
  channelNumber?: number;
  addedAt: number;
  /** QA probe duration (ms). Lower is better. */
  loadTimeMs?: number;
  /** Derived ranking score after QA (0–100+). */
  qualityScore?: number;
}

export interface RadioStation {
  id: string;
  name: string;
  favicon: string;
  tags: string[];
  country: string;
  countrycode: string;
  language: string;
  languages?: string[];
  stream: string;
  type: "radio";
  status?: StreamStatus;
  lastCheck?: number;
  bitrate?: number;
  codec?: string;
  votes?: number;
  favorite: boolean;
  addedAt: number;
  loadTimeMs?: number;
  qualityScore?: number;
}

export interface Source {
  id: string;
  name: string;
  type: "iptv" | "radio" | "youtube" | "epg";
  url: string;
  enabled: boolean;
  autoRefresh: boolean;
  lastSync: number;
  channelCount: number;
  builtin?: boolean;
}

export interface SearchFilter {
  query: string;
  country?: string;
  language?: string;
  category?: string;
  type?: ChannelType | "all";
  onlyFavorites?: boolean;
  onlyWorking?: boolean;
}
