/** GROVEE desktop companion plugins — extensible registry (first: search-companion). */

export type PluginCategory = "search" | "media" | "tools" | "data";

export type PluginStatus = "unknown" | "offline" | "online" | "degraded";

export type PluginDownloadInfo = {
  url: string;
  filename: string;
  sizeHintHe: string;
};

export type PluginHealthResult = {
  status: PluginStatus;
  messageHe: string;
  version?: string;
  enginesReady?: string[];
  enginesFailed?: string[];
  latencyMs?: number;
};

export type PluginHealthState = PluginHealthResult & {
  checkedAt: number;
};

export type GroveePluginManifest = {
  id: string;
  nameHe: string;
  shortNameHe: string;
  descriptionHe: string;
  icon: string;
  category: PluginCategory;
  version: string;
  defaultBaseUrl: string;
  defaultPort: number;
  installStepsHe: string[];
  download?: {
    win?: PluginDownloadInfo;
  };
  upstream?: {
    name: string;
    version: string;
    url: string;
  };
};

export type GroveePlugin = GroveePluginManifest & {
  probeHealth: () => Promise<PluginHealthResult>;
  probeSearch?: (query?: string) => Promise<{ ok: boolean; messageHe: string; hitCount?: number }>;
  getBaseUrl: () => string;
  setBaseUrl: (url: string) => void;
  isActive: () => boolean;
  onDetectedOnline?: () => void;
};

export type PluginHealthSnapshot = Record<string, PluginHealthState | undefined>;
