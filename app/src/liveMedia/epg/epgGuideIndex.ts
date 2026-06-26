import { stripTvgFeed } from "./normalize";

type GuideRow = {
  channel: string | null;
  site: string;
};

let loadPromise: Promise<Set<string>> | null = null;

async function fetchGuideChannelIds(): Promise<Set<string>> {
  const res = await fetch("https://iptv-org.github.io/api/guides.json", { cache: "force-cache" });
  if (!res.ok) return new Set();
  const rows = (await res.json()) as GuideRow[];
  const ids = new Set<string>();
  for (const row of rows) {
    if (!row.channel) continue;
    ids.add(stripTvgFeed(row.channel));
    ids.add(row.channel);
  }
  return ids;
}

export function getIptvOrgEpgChannelIds(): Promise<Set<string>> {
  if (!loadPromise) loadPromise = fetchGuideChannelIds();
  return loadPromise;
}

export function resetEpgGuideIndexForTests(): void {
  loadPromise = null;
}
